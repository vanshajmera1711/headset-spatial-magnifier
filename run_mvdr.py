import os
import json
import numpy as np
import soundfile as sf
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "mvdr_output")
METRICS_PATH = os.path.join(DATA_DIR, "mvdr_metrics.json")

SAMPLE_RATE = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256   # 50% overlap
N_BINS = FFT_LENGTH // 2 + 1
REFERENCE_CHANNEL = 1  # left_front anchor channel

SPEED_OF_SOUND = 343.0  # m/s

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. GEOMETRY RECONSTRUCTION
# ==========================================
def reconstruct_mic_positions(centroid):
    """
    Reconstructs the 6-channel triangular mic array 3D coordinates from the centroid.
    """
    cx, cy, height = centroid
    mics = np.array([
        [cx - 0.08,  cy,        height + 0.03], # ch 0 - left_top
        [cx - 0.08,  cy + 0.02, height       ], # ch 1 - left_front
        [cx - 0.08,  cy - 0.02, height       ], # ch 2 - left_back
        [cx + 0.08,  cy - 0.02, height       ], # ch 3 - right_back
        [cx + 0.08,  cy + 0.02, height       ], # ch 4 - right_front
        [cx + 0.08,  cy,        height + 0.03]  # ch 5 - right_top
    ])
    return mics

# ==========================================
# 3. SI-SNR EVALUATION METRIC
# ==========================================
def compute_si_snr_numpy(estimate, reference, epsilon=1e-8):
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    
    min_len = min(len(estimate), len(reference))
    estimate = estimate[:min_len]
    reference = reference[:min_len]
    
    ref_pow = np.sum(reference ** 2)
    dot_prod = np.sum(estimate * reference)
    scale = dot_prod / (ref_pow + epsilon)
    
    target = scale * reference
    error = estimate - target
    
    target_pow = np.sum(target ** 2)
    error_pow = np.sum(error ** 2)
    
    return 10 * np.log10(target_pow / (error_pow + epsilon))

# ==========================================
# 4. CORE ENGINE WITH REAL BLIND MVDR
# ==========================================
def process_single_sample(sample_idx):
    try:
        # File paths
        meta_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_meta.json")
        mix_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_mix.wav")
        tgt_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_target_gt.wav")
        out_path = os.path.join(OUTPUT_FOLDER, f"enhanced_sample_{sample_idx}_mix.wav")
        
        if not (os.path.exists(meta_path) and os.path.exists(mix_path) and os.path.exists(tgt_path)):
            return {"sample_idx": sample_idx, "status": "missing_files"}
        
        # Load Metadata & Reconstruct Array Physics
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        centroid = meta["headset_centroid"]
        target_pos = np.array(meta["target_position"])
        mic_positions = reconstruct_mic_positions(centroid)
        
        # Load Audio (Crop to 64000 samples to strip convolution tails cleanly)
        mix, sr = sf.read(mix_path, frames=64000, dtype='float32')
        tgt, _  = sf.read(tgt_path, frames=64000, dtype='float32')
        
        # Calculate Input Metrics on Reference Channel
        mix_ref = mix[:, REFERENCE_CHANNEL]
        tgt_ref = tgt[:, REFERENCE_CHANNEL]
        sisnr_in = compute_si_snr_numpy(mix_ref, tgt_ref)
        
        # STFT transformation: (Channels, Bins, Frames)
        f, t, stft_mix = signal.stft(mix.T, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
        n_channels, n_bins, n_frames = stft_mix.shape
        
        # RELATIVE TIME DIFFERENCE OF ARRIVAL (TDOA) FORMULATION
        absolute_distances = np.linalg.norm(mic_positions - target_pos, axis=1) # (6,)
        absolute_delays = absolute_distances / SPEED_OF_SOUND # (6,)
        
        # Lock relative delays to anchor channel 1
        relative_delays = absolute_delays - absolute_delays[REFERENCE_CHANNEL] # (6,)
        
        # Frequency vector corresponding to the STFT bins
        freqs = np.arange(n_bins) * (SAMPLE_RATE / FFT_LENGTH) # (257,)
        
        # Construct the phase matching matrix using relative time offsets
        steering_vector = np.exp(-1j * 2 * np.pi * freqs[None, :] * relative_delays[:, None])
        
        # Blind Noise Covariance Tracking initialization
        R_noise = np.zeros((n_channels, n_channels, n_bins), dtype=np.complex64)
        for b in range(n_bins):
            R_noise[:, :, b] = np.eye(n_channels) * 1e-3
            
        power_fast = np.zeros(n_bins)
        power_slow = np.zeros(n_bins)
        
        # Process Frame-by-Frame Causally (Bins fully vectorized)
        enhanced_spec = np.zeros((n_bins, n_frames), dtype=np.complex64)
        I_batch = np.tile(np.eye(n_channels)[None, :, :], (n_bins, 1, 1))
        
        for t_idx in range(n_frames):
            X_t = stft_mix[:, :, t_idx] # Shape: (6, 257)
            
            # Fast/Slow energy tracking for VAD proxy
            current_energy = np.abs(X_t[REFERENCE_CHANNEL, :]) ** 2
            if t_idx == 0:
                power_fast = current_energy
                power_slow = current_energy
            else:
                power_fast = 0.4 * current_energy + (1 - 0.4) * power_fast
                power_slow = 0.98 * current_energy + (1 - 0.98) * power_slow
                
            # Vectorized outer-product across all bins: shape (6, 6, 257)
            X_outer = np.einsum('if,jf->ijf', X_t, np.conj(X_t))
            
            # Mask generation: Update noise space only during low-energy moments
            noise_mask = (power_fast <= 1.05 * power_slow).astype(np.float32)
            
            # Update R_noise via broadcasting
            alpha_tensor = 0.95 * noise_mask + 1.0 * (1 - noise_mask)
            R_noise = alpha_tensor[None, None, :] * R_noise + (1 - alpha_tensor[None, None, :]) * X_outer
            
            # Rearrange dimensions for batched matrix operation: (6, 6, 257) -> (257, 6, 6)
            R_batch = np.moveaxis(R_noise, 2, 0)
            
            # ACTIVE BLIND MVDR PROCESSING TRACKS COVARIANCE SHIFT
            R_batch = R_batch + 1e-3 * I_batch  # Apply stable static diagonal floor
            
            # Batched inversion: handles all 257 bins simultaneously in C
            R_inv_batch = np.linalg.pinv(R_batch) # Shape: (257, 6, 6)
            
            # Batched weight computation formulation
            d_batch = np.moveaxis(steering_vector, 1, 0)[:, :, None] # Shape: (257, 6, 1)
            numerator_batch = R_inv_batch @ d_batch                  # Shape: (257, 6, 1)
            
            d_H_batch = np.conj(np.moveaxis(d_batch, 2, 1))          # Shape: (257, 1, 6)
            denominator_batch = d_H_batch @ numerator_batch          # Shape: (257, 1, 1)
            
            w_batch = np.squeeze(numerator_batch / (denominator_batch + 1e-8), axis=-1) # (257, 6)
            
            # Apply Filter to all bins at once
            enhanced_spec[:, t_idx] = np.einsum('fi,if->f', np.conj(w_batch), X_t)
                
        # ISTFT reconstruction
        _, enhanced_waveform = signal.istft(enhanced_spec, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
        enhanced_waveform = enhanced_waveform[:len(tgt_ref)]
        
        # Calculate Output Metrics
        sisnr_out = compute_si_snr_numpy(enhanced_waveform, tgt_ref)
        
        # Save Enhanced File
        sf.write(out_path, enhanced_waveform, SAMPLE_RATE)
        
        return {
            "sample_idx": sample_idx,
            "status": "success",
            "sisnr_input": float(sisnr_in),
            "sisnr_output": float(sisnr_out),
            "sisnr_improvement": float(sisnr_out - sisnr_in)
        }
        
    except Exception as e:
        return {"sample_idx": sample_idx, "status": f"failed: {str(e)}"}

# ==========================================
# 5. MULTIPROCESSING ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    NUM_SAMPLES = 1000
    sample_indices = list(range(NUM_SAMPLES))
    
    num_workers = os.cpu_count()
    print(f"Executing Blind MVDR Engine via parallel pools across {num_workers} cores...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(tqdm(
            executor.map(process_single_sample, sample_indices),
            total=NUM_SAMPLES,
            desc="MVDR (Causal, blind)"
        ))
        
    # Aggregate and serialize metrics
    valid_scenes = 0
    total_in = 0.0
    total_out = 0.0
    per_file_log = {}
    
    for res in list_of_results:
        if res["status"] == "success":
            valid_scenes += 1
            total_in += res["sisnr_input"]
            total_out += res["sisnr_output"]
            
            per_file_log[f"sample_{res['sample_idx']}"] = {
                "sisnr_input": float(res["sisnr_input"]),
                "sisnr_output": float(res["sisnr_output"]),
                "sisnr_improvement": float(res["sisnr_improvement"])
            }
            
    if valid_scenes > 0:
        mean_in = total_in / valid_scenes
        mean_out = total_out / valid_scenes
        mean_imp = mean_out - mean_in
    else:
        mean_in, mean_out, mean_imp = 0.0, 0.0, 0.0
        
    summary_data = {
        "scenes_processed": valid_scenes,
        "mean_sisnr_input_db": round(float(mean_in), 2),
        "mean_sisnr_output_db": round(float(mean_out), 2),
        "mean_sisnr_improvement_db": round(float(mean_imp), 2),
        "per_file_metrics": per_file_log
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print("\n" + "─" * 40 + "\n Summary \n" + "─" * 40)
    print(f"  Mean SI-SNR input       : {mean_in:.2f} dB")
    print(f"  Mean SI-SNR output      : {mean_out:.2f} dB")
    print(f"  Mean SI-SNR improvement : {mean_imp:.2f} dB")
    print(f"  Samples processed       : {valid_scenes}")
    print(f"\nMetrics saved → {METRICS_PATH}")
    print(f"Enhanced audio saved → {OUTPUT_FOLDER}")