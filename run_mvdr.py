import os
import json
import torch
import torchaudio
import soundfile as sf

# ==========================================
# 1. BULLETPROOF SOUNDFILE PATCHES 
# ==========================================
def soundfile_load_patch(filepath, frame_offset=0, num_frames=-1, convert=True, channels_first=True):
    data, samplerate = sf.read(filepath, start=frame_offset, frames=num_frames, dtype='float32', always_2d=True)
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.t()
    return tensor, samplerate

def soundfile_save_patch(filepath, src, sample_rate, channels_first=True, bits_per_sample=16, format=None, encoding=None):
    if channels_first and src.ndim > 1:
        data = src.t().numpy()
    else:
        data = src.numpy()
    sf.write(filepath, data, sample_rate, subtype=f'PCM_{bits_per_sample}')

torchaudio.load = soundfile_load_patch
torchaudio.save = soundfile_save_patch

# ==========================================
# 2. EXACT SI-SNR METRIC COMPUTATION
# ==========================================
def compute_si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    
    min_len = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :min_len]
    reference = reference[..., :min_len]
    
    reference_pow = reference.pow(2).sum(dim=-1, keepdim=True)
    mix_pow = (estimate * reference).sum(dim=-1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    target = scale * reference
    error = estimate - target

    target_pow = target.pow(2).sum(dim=-1)
    error_pow = error.pow(2).sum(dim=-1)

    si_snr_val = 10 * torch.log10(target_pow / (error_pow + epsilon))
    return si_snr_val.mean().item()

# ==========================================
# 3. CONFIGURATION & DIRECTORIES
# ==========================================
INPUT_FOLDER = r"C:\Users\Admin\room_simulation\data\simulated" 
OUTPUT_FOLDER = r"C:\Users\Admin\room_simulation\data\mvdr_output"
METRICS_FOLDER = r"C:\Users\Admin\room_simulation\data"

SAMPLE_RATE = 16000  
REFERENCE_CHANNEL = 1  # Aligned to our master evaluation channel 
N_FFT = 1024
N_HOP = 256

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(METRICS_FOLDER, exist_ok=True)

stft = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=N_HOP, power=None)
istft = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP)
psd_transform = torchaudio.transforms.PSD()
mvdr_transform = torchaudio.transforms.SoudenMVDR()

# ==========================================
# 4. FIXED PROCESSING LOOP FOR TARGET_GT STEMS
# ==========================================
all_files = os.listdir(INPUT_FOLDER)

# Gather only the true mixture files based on your directory structure
mixture_files = [f for f in all_files if f.endswith("_mix.wav") or (f.endswith("_mix") and not f.endswith(".json"))]

# If extensions are hidden in your OS view, fall back to matching the base string safely
if not mixture_files:
    mixture_files = [f for f in all_files if f.endswith("_mix")]

scenes_processed = 0
total_sisnr_in = 0.0
total_sisnr_out = 0.0
metrics_log = {}

print(f"Starting analysis on dataset mixtures inside: {INPUT_FOLDER}\n")

for filename in mixture_files:
    # Handle files whether they have extensions explicitly appended or hidden
    base_ext = ".wav" if filename.endswith(".wav") else ""
    file_path = os.path.join(INPUT_FOLDER, filename if base_ext else f"{filename}.wav")
    
    # Map directly to your explicit target_gt file naming convention
    target_filename = filename.replace("_mix", "_target_gt")
    target_path = os.path.join(INPUT_FOLDER, target_filename if base_ext else f"{target_filename}.wav")
    
    # Security check: verify both files exist on disk before moving forward
    if not (os.path.exists(file_path) and os.path.exists(target_path)):
        continue

    try:
        # Load audio signals
        waveform_mix, sr = torchaudio.load(file_path)
        waveform_clean, sr_c = torchaudio.load(target_path)
        
        if waveform_mix.shape[0] < 2:
            continue  # Requires multi-channel data

        # Dynamic Noise Field Isolation (Mix - Target = Combined Clutter)
        # Pads/truncates to ensure exact shape alignment before subtraction
        min_samples = min(waveform_mix.shape[-1], waveform_clean.shape[-1])
        waveform_mix = waveform_mix[:, :min_samples]
        waveform_clean = waveform_clean[:, :min_samples]
        
        total_noise_field = waveform_mix - waveform_clean

        # Isolate anchor reference channel arrays (Channel 1)
        clean_ref = waveform_clean[REFERENCE_CHANNEL:REFERENCE_CHANNEL+1]
        mix_ref = waveform_mix[REFERENCE_CHANNEL:REFERENCE_CHANNEL+1]
        
        # Calculate True Base Input SI-SNR
        sisnr_in = compute_si_snr(mix_ref, clean_ref)

        # Process MVDR
        waveform_mix_double = waveform_mix.to(torch.double)
        stft_mix = stft(waveform_mix_double)
        
        # Compute Power Spectral Density matrices using the isolated tracks
        stft_target = stft(waveform_clean.to(torch.double))
        stft_noise_field = stft(total_noise_field.to(torch.double))
        
        target_mag = stft_target.abs()[REFERENCE_CHANNEL]
        noise_mag = stft_noise_field.abs()[REFERENCE_CHANNEL]
        
        # Ideal Mask Assignment
        irm_speech = (target_mag > noise_mag).to(torch.double)
        irm_noise = (target_mag <= noise_mag).to(torch.double)
        
        # Compute spatial covariance matrices
        psd_speech = psd_transform(stft_mix, irm_speech)
        psd_noise = psd_transform(stft_mix, irm_noise)
        
        # Souden Beamformer Execution
        stft_souden = mvdr_transform(stft_mix, psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)
        waveform_souden = istft(stft_souden, length=waveform_mix.shape[-1])
        waveform_souden = waveform_souden.reshape(1, -1).to(torch.float32)
        
        # Calculate True Output SI-SNR
        sisnr_out = compute_si_snr(waveform_souden, clean_ref)
        sisnr_imp = sisnr_out - sisnr_in
        
        # Save Enhanced File
        output_path = os.path.join(OUTPUT_FOLDER, f"enhanced_{filename if base_ext else f'{filename}.wav'}")
        torchaudio.save(output_path, waveform_souden, SAMPLE_RATE)
        
        # Update metrics
        scenes_processed += 1
        total_sisnr_in += sisnr_in
        total_sisnr_out += sisnr_out
        
        metrics_log[filename] = {
            "sisnr_input": round(sisnr_in, 2),
            "sisnr_output": round(sisnr_out, 2),
            "sisnr_improvement": round(sisnr_imp, 2)
        }
        print(f"Processed scene {scenes_processed}: {filename} (Gain: {sisnr_imp:+.2f} dB)")

    except Exception as e:
        print(f"Error handling file {filename}: {str(e)}")


# ==========================================
# 5. GENERATE FINAL METRICS & CONSOLE PRINT
# ==========================================
if scenes_processed > 0:
    mean_in = total_sisnr_in / scenes_processed
    mean_out = total_sisnr_out / scenes_processed
    mean_imp = mean_out - mean_in
else:
    mean_in, mean_out, mean_imp = 0.0, 0.0, 0.0

metrics_json_path = os.path.join(METRICS_FOLDER, "mvdr_metrics.json")
summary_data = {
    "scenes_processed": scenes_processed,
    "mean_sisnr_input_db": round(mean_in, 2),
    "mean_sisnr_output_db": round(mean_out, 2),
    "mean_sisnr_improvement_db": round(mean_imp, 2),
    "per_file_metrics": metrics_log
}

with open(metrics_json_path, "w") as f:
    json.dump(summary_data, f, indent=4)

print("\n--- MVDR Results ---")
print(f"Scenes processed : {scenes_processed}")
print(f"Mean SI-SNR input : {mean_in:.2f} dB")
print(f"Mean SI-SNR output: {mean_out:.2f} dB")
print(f"Mean SI-SNR improvement: {mean_imp:.2f} dB")
print(f"Outputs saved to: data/mvdr_output")
print(f"Metrics saved to: data/mvdr_metrics.json")