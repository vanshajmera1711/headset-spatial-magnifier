import json
import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_directivity(w, mics, reference_channel, sample_rate, fft_length, n_bins, plot_freq_hz):
    """
    w: MVDR weights (n_bins, n_channels)
    plots the beamformer response across azimuth angles in the horizontal plane
    """
    angles = np.linspace(0, 2 * np.pi, 360)
    radius = 1.0  # 1 meter distance for the candidate sources
    
    # pick a single frequency bin to plot
    freq_bin = int(plot_freq_hz / (sample_rate / fft_length))
    w_at_freq = w[freq_bin]  # (n_channels,)
    
    responses = []

    for angle in angles:
        # candidate source position at this angle in the horizontal plane
        candidate_pos = np.array([radius * np.sin(angle), radius * np.cos(angle), 0.0])
        
        # build steering vector for this candidate direction
        abs_distances = np.linalg.norm(mics - candidate_pos, axis=1)
        abs_delays = abs_distances / SPEED_OF_SOUND
        rel_delays = abs_delays - abs_delays[reference_channel]
        rel_amplitude = abs_distances[reference_channel] / abs_distances
        
        freqs = np.arange(n_bins) * (sample_rate / fft_length)
        f = freqs[freq_bin]
        
        d_candidate = rel_amplitude * np.exp(-1j * 2 * np.pi * f * rel_delays)  # (n_channels,)
        
        # beamformer response = w^H @ d
        response = np.abs(np.conj(w_at_freq) @ d_candidate)
        responses.append(response)

    responses = np.array(responses)
    responses_db = 20 * np.log10(responses / (np.max(responses) + 1e-8))

    # polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.plot(angles, responses_db)
    ax.set_theta_zero_location('N')   # 0 degrees at top (front of headset)
    ax.set_theta_direction(-1)         # clockwise
    ax.set_rlim(-40, 0)               # dynamic range in dB
    ax.set_title(f'MVDR Directivity at {plot_freq_hz} Hz', va='bottom')
    plt.tight_layout()
    plt.savefig('directivity.png', dpi=150)
    plt.show()


# input and output directories 

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset_reverb")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "mvdr_output_reverb")
METRICS_PATH = os.path.join(DATA_DIR, "mvdr_metrics_reverb.json")


SAMPLE_RATE = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256   # 50% overlap
N_BINS = FFT_LENGTH // 2 + 1
REFERENCE_CHANNEL = 1  # left_front anchor channel

SPEED_OF_SOUND = 343.0  # m/s

os.makedirs(OUTPUT_FOLDER, exist_ok=True)



#compute si snr 

def compute_si_snr(reference,estimate,epsilon=1e-8):
    estimate=estimate-np.mean(estimate) #here we are mean centering both the quantities to ensure that the changes in volume levels caused by the mvdr does not affect the si_snr calculation 
    reference=reference-np.mean(reference)

    min_len=min(len(reference),len(estimate)) #here we are taking min len to ensure we dont get an error in case the lengths are different when we are multiplying 
    estimate=estimate[:min_len]
    reference=reference[:min_len]

    ref_pow=np.sum(reference**2)
    dot_prod=np.sum(reference*estimate)
    scale = dot_prod/(ref_pow+epsilon)

    target=scale*reference
    error=estimate-target

    target_pow=np.sum(target**2)
    error_pow=np.sum(error**2)

    si_snr=10*np.log10(target_pow/(error_pow+epsilon) )
    return si_snr

#actual mvdr processing

def simple_mvdr(sample_idx):
    try:
        # File paths
        mix_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_mix.wav")
        tgt_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_target_gt.wav")
        out_path = os.path.join(OUTPUT_FOLDER, f"enhanced_sample_{sample_idx}_mix.wav")

        if not (os.path.exists(mix_path) and os.path.exists(tgt_path)):
            return {"sample_idx": sample_idx, "status": "missing_files"}
        
        #headset geometry and mic positions
        
        cx,cy,height=[0,0,0]
        mics=np.array([
            [cx-0.08,cy,height+0.03],#left top
            [cx-0.08,cy+0.02,height],#left front
            [cx-0.08,cy-0.02,height],#left back
            [cx+0.08,cy,height+0.03],#right top
            [cx+0.08,cy+0.02,height],#right front
            [cx+0.08,cy-0.02,height]#right back 
        ]) 
        target_pos = np.array([0.0, 0.12, -0.10])#10 cm ahead and 12 cm below the headset center so around the mouth position of the user

        #load audio 
        mix, sr = sf.read(mix_path, frames=64000, dtype='float32') #fixing the frames to 64k to ensure there are no extra trails being read and causing issues in the matrix length    
        tgt, _  = sf.read(tgt_path, frames=64000, dtype='float32')
        mix_ref = mix[:, REFERENCE_CHANNEL]
        tgt_ref = tgt[:, REFERENCE_CHANNEL]
        sisnr_in = compute_si_snr(mix_ref, tgt_ref) #calculating the i/p sisnr using mix and target reference

        #stft here 
        f,t,stft_op=signal.stft(mix.T, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT) 
        n_channels, n_bins, n_frames = stft_op.shape

        X_outer_all = np.einsum('ibf,jbf->ijbf', stft_op, np.conj(stft_op)) #taking the autocorrelation of the stft output of each frame 
        #this gives n_Channel,n_channel,n_bins,n_frames as the shape of the output   
        Rxx = np.mean(X_outer_all, axis=-1) #computing rxx here by averaging the outer products across frames
        #we averaged x_outer_all across the frames so here we have n_channel,n_channel,n_bins as the shape of the output

        #steering vector calculation 
        #here we do not have doa and neither are we using a ula so we cant directly use e^-j*k*m-1*d*sin theta so instead we are replacing it with delays and distances which we can calculate
        absolute_distances = np.linalg.norm(mics - target_pos, axis=1) 
        absolute_delays = absolute_distances / SPEED_OF_SOUND 
        relative_delays = absolute_delays - absolute_delays[REFERENCE_CHANNEL]#shape is (n_channels,)
        relative_amplitude = absolute_distances[REFERENCE_CHANNEL] / absolute_distances

        freqs = np.arange(n_bins) * (SAMPLE_RATE / FFT_LENGTH) # arange will create bins from0 to 256 and then we multiply it with the frequency resolution to include the k factor in the steering vector calculation
        #shape is (n_bins,)
        steering_vector = (
            relative_amplitude[:, None]* np.exp(-1j * 2 * np.pi * freqs[None, :] * relative_delays[:, None])
            )#freqs*delays gives (channels, n_bins) and then relative amoplitude is (channels,) with none its (channels, 1) and this broadcastes with the earlier term gives us (channels, n_bins)
        Rxx_T = Rxx.transpose(2, 0, 1)   # → (n_bins, channels, channels) 
        #regularization to avoid blowing up of the matrix inversion
        epsilon_reg = 1e-3
        I = np.eye(n_channels)
        trace_per_bin = np.real(np.trace(Rxx_T, axis1=1, axis2=2))  # (n_bins,) trace tells us how loud or how much energy is present in each bin 
        Rxx_reg = Rxx_T + epsilon_reg * trace_per_bin[:, None, None] * I[None, :, :] #here we multiply the trace with the identity matrix to ensure that we are adding a small value to the diagonal elements of the Rxx matrix to avoid singularity and then we add it to the original Rxx_T matrix
        #calculating rxx inv to use in weights of the mvdr filter
        Rxx_inv = np.linalg.inv(Rxx_reg)  # (n_bins, channels, channels)

        #now we need d_H
        d = steering_vector.T[:, :, None]   # (n_bins, channels, 1)
        d_H = np.conj(d.transpose(0, 2, 1)) # (n_bins, 1, channels)

        #finally calculating the weights of the mvdr filter
        # numerator: Rxx_inv @ d → (n_bins, channels, 1)
        numerator = Rxx_inv @ d

        # denominator: scalar per bin → (n_bins, 1, 1)
        denominator = d_H @ numerator  # (n_bins, 1, 1)

        w = (numerator / denominator).squeeze(-1)
        if sample_idx == 0:
            for freq in [500, 1000, 2000, 4000]:
                compute_directivity(w, mics, REFERENCE_CHANNEL, SAMPLE_RATE, FFT_LENGTH, n_bins, plot_freq_hz=freq)

        #now applying the weights back to the stft output to get the enhanced signal

        # stft_op is (channels, n_bins, n_frames)
        # rearrange to (n_bins, channels, n_frames) for clean matmul
        X = stft_op.transpose(1, 0, 2)   # (n_bins, channels, n_frames)

        # w is (n_bins, channels) → (n_bins, 1, channels) for matmul against X
        # w^H @ X per bin: (n_bins, 1, channels) @ (n_bins, channels, n_frames) → (n_bins, 1, n_frames)
        enhanced_stft = (np.conj(w)[:, None, :] @ X).squeeze(1)  # (n_bins, n_frames)

        # ISTFT back to time domain
        _, enhanced_audio = signal.istft(
            enhanced_stft, 
            fs=SAMPLE_RATE, 
            nperseg=FFT_LENGTH, 
            noverlap=FFT_SHIFT
        )

        enhanced_audio = enhanced_audio.real.astype(np.float32)

        sf.write(out_path, enhanced_audio, SAMPLE_RATE)

        #enhanced op metrics 
        sisnr_out = compute_si_snr(enhanced_audio, tgt_ref)
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
# 4. MULTIPROCESSING ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    NUM_SAMPLES = 1000
    sample_indices = list(range(NUM_SAMPLES))

    num_workers = os.cpu_count()
    print(f"Executing Standalone Blind MVDR Engine via parallel pools across {num_workers} cores...")

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(tqdm(
            executor.map(simple_mvdr, sample_indices),
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

 

        



