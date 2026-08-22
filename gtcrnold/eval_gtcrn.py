import h5py
import torch
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
from pystoi import stoi

# Import your model class
from models.gtcrn import GTCRN 

# --- CONFIGURATION (Must match training hyperparameters) ---
FS = 16000
N_FFT = 512      # stft_length
BLOCK_SHIFT = 256 # stft_shift
CIRM_K = 1.0     # cirm_comp_K
CIRM_C = 1.0     # cirm_comp_C
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./scratch/" 

from scipy.signal import correlate

def align_signals(ref, est):
    """
    Finds the optimal lag to align the estimated signal with the reference.
    """
    # Convert to numpy and flatten for correlation
    ref_np = ref.detach().cpu().numpy().flatten()
    est_np = est.detach().cpu().numpy().flatten()
    
    # Compute cross-correlation
    corr = correlate(ref_np, est_np, mode='full')
    lag = np.argmax(corr) - (len(est_np) - 1)
    
    # Apply the shift
    if lag > 0:
        # est is delayed; shift ref forward or est back
        return ref[:, lag:], est[:, :-lag]
    elif lag < 0:
        # ref is delayed; shift est forward or ref back
        return ref[:, :lag], est[:, -lag:]
    
    return ref, est

# --- REPLICATED SI-SDR LOGIC FROM EnhancementExp ---
def compute_global_si_sdr(est_clean_td, clean_td):
    """Exactly as defined in EnhancementExp"""
    def si_sdr_inner(s, s_hat):
        # alpha = <s_hat, s> / <s, s>
        alpha = torch.einsum('cs,cs->c', s_hat, s) / (torch.einsum('cs,cs->c', s, s) + 1e-14)
        scaled_ref = torch.unsqueeze(alpha, dim=1) * s
        # Metric is 10 * log10(Signal Power / Error Power)
        sdr = 10 * torch.log10(torch.einsum('cs,cs->c', scaled_ref, scaled_ref) / (
                    torch.einsum('cs,cs->c', scaled_ref - s_hat, scaled_ref - s_hat) + 1e-14))
        return sdr
    return si_sdr_inner(clean_td, est_clean_td)

def get_complex_masks_from_stacked(real_mask):
    """Replicated mask decoding from EnhancementExp"""
    compressed_complex_speech_mask = real_mask[:, 0, ...] + (1j) * real_mask[:, 1, ...]
    # tanh inversion
    complex_speech_mask = (-1 / CIRM_C) * torch.log(
        (CIRM_K - CIRM_K * compressed_complex_speech_mask) / (
                CIRM_K + CIRM_K * compressed_complex_speech_mask + 1e-8))
    return complex_speech_mask

def apply_neural_steer(stft, theta_t_deg):
    """Steering logic for 4-mic system provided by your previous iteration"""
    c = 343.0
    dist = [0.0283, 0.0400, 0.0283]
    theta_1_list = [225.0, 180.0, 135.0]
    f = np.linspace(0, FS / 2, N_FFT // 2 + 1)
    theta_t_deg = theta_t_deg % 360
    stft_list = [stft[i] for i in range(4)]
    
    if (135 > theta_t_deg >= 45):
        stft_list = [stft_list[1], stft_list[2], stft_list[3], stft_list[0]]
        theta_t_deg -= 90
    elif (225 > theta_t_deg >= 135):
        stft_list = [stft_list[2], stft_list[3], stft_list[0], stft_list[1]]
        theta_t_deg -= 180
    elif (315 > theta_t_deg >= 225):
        stft_list = [stft_list[3], stft_list[0], stft_list[1], stft_list[2]]
        theta_t_deg -= 270

    theta_t = np.deg2rad(theta_t_deg)
    steered_stfts = [stft_list[0]]
    for i in range(3):
        d = dist[i]
        theta_1 = np.deg2rad(theta_1_list[i])
        Y_m = stft_list[i+1]
        delta_psi = 2 * np.pi * f * (d / c) * (np.cos(theta_1) - np.cos(theta_1 + theta_t))
        a_k = torch.from_numpy(np.exp(1j * delta_psi)).to(DEVICE)
        steered_stfts.append(Y_m * a_k[:, None])
    return torch.stack(steered_stfts, dim=0)

def run_evaluation(ckpt_path, h5_path):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize Model
    model = GTCRN(n_channels=4).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    model.eval()

    # REPLICATED WINDOW: Square root Hann window
    window = torch.sqrt(torch.hann_window(N_FFT)).to(DEVICE)
    
    results = {"stoi": [], "estoi": [], "si_sdr": []}

    with h5py.File(h5_path, 'r') as f:
        audio_ds = f['test'] 
        theta_ds = f['theta'][:]
        n_samples = audio_ds.shape[0]

        for i in tqdm(range(n_samples), desc="Evaluating"):
            # 1. Load data & Mix
            reverb_target = torch.from_numpy(audio_ds[i, 0]).to(DEVICE) 
            reverb_noise = torch.from_numpy(audio_ds[i, 1]).to(DEVICE)  
            mix_audio_4ch = reverb_target + reverb_noise
            
            # Ground truth clean ref (Mic 0)
            clean_ref_td_raw = torch.from_numpy(audio_ds[i, 2, 0, :]).to(DEVICE).unsqueeze(0) 
            target_theta = theta_ds[i]

            # 2. STFT
            stft_mix = torch.stft(
                mix_audio_4ch, N_FFT, BLOCK_SHIFT, 
                window=window, center=True, return_complex=True
            )
            
            # 3. Spatial Steering and Inference
            steered_stft = apply_neural_steer(stft_mix, target_theta)
            model_in = torch.cat((steered_stft.real.unsqueeze(0), 
                                  steered_stft.imag.unsqueeze(0)), dim=1).to(torch.float32)

            with torch.no_grad():
                stacked_mask = model(model_in)
                speech_mask = get_complex_masks_from_stacked(stacked_mask)
                # Filter Reference Channel (Steered Index 0)
                est_clean_stft = steered_stft[0, ...].unsqueeze(0) * speech_mask

            # 4. ISTFT
            est_clean_td = torch.istft(
                est_clean_stft, N_FFT, BLOCK_SHIFT, 
                window=window, center=True
            )

            # 5. Length Matching Logic (Match GTCRNExp.shared_step)
            output_len = est_clean_td.shape[-1]
            ref_clean_td = clean_ref_td_raw[..., :output_len]

            # 6. Metrics Calculation (Fixed Precision for einsum)
            est_clean_td = est_clean_td.to(torch.float32)
            ref_clean_td = ref_clean_td.to(torch.float32)

            si_sdr_val = compute_global_si_sdr(est_clean_td, ref_clean_td)
            results["si_sdr"].append(si_sdr_val.mean().item())

            # Convert to numpy for STOI
            est_np = est_clean_td.squeeze().cpu().numpy()
            ref_np = ref_clean_td.squeeze().cpu().numpy()
            results["stoi"].append(stoi(ref_np, est_np, FS, extended=False))
            results["estoi"].append(stoi(ref_np, est_np, FS, extended=True))

            # Optional: Save every 10th sample
            if i % 10 == 0:
                sf.write(os.path.join(SAVE_DIR, f"sample_{i}_noisy.wav"), 
                         (reverb_target[0] + reverb_noise[0]).cpu().numpy(), FS)
                sf.write(os.path.join(SAVE_DIR, f"sample_{i}_enhanced.wav"), est_np, FS)

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print(f"EVALUATION SUMMARY (N={n_samples})")
    print("="*40)
    for m in ["stoi", "estoi", "si_sdr"]:
        arr = np.array(results[m])
        if m == "si_sdr":
            print(f"{m.upper():<7}: {np.mean(arr):.4f} dB")
        else:
            print(f"{m.upper():<7}: {np.mean(arr):.4f} ± {np.std(arr):.4f}")

if __name__ == "__main__":
    CKPT_PATH = "./logs/gtcrn_ch4_sp2.ckpt"
    H5_PATH = "./prep/test_theta_0_interf_2.hdf5"
    run_evaluation(CKPT_PATH, H5_PATH)