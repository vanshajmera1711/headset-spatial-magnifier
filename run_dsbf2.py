import os
import json
import numpy as np
import soundfile as sf
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==========================================
# TRUE PHASE-ALIGNED DELAY-AND-SUM BEAMFORMER
# ==========================================
# Goal: isolate whether the STEERING VECTOR in run_mvdr.py is correct, by
# using the EXACT SAME geometry/steering-vector code, but applying a static
# delay-and-sum weight (no adaptive covariance, no nulling at all).
#
# If this gives a solid, interferer-count-INDEPENDENT improvement (roughly
# flat across num_interferers, in the few-dB range), the steering vector is
# fine and the MVDR shortfall is in the covariance/nulling logic.
#
# If this gives near-zero or negative improvement, the steering vector
# itself is wrong (sign, units, or geometry), independent of MVDR's
# adaptive logic.

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "dsbf_proper_output")
METRICS_PATH = os.path.join(DATA_DIR, "dsbf_proper_metrics.json")

SAMPLE_RATE = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256
N_BINS = FFT_LENGTH // 2 + 1
REFERENCE_CHANNEL = 1  # left_front anchor channel -- SAME as run_mvdr.py

SPEED_OF_SOUND = 343.0  # m/s -- SAME as run_mvdr.py

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# GEOMETRY RECONSTRUCTION -- IDENTICAL to run_mvdr.py
# ==========================================
def reconstruct_mic_positions(centroid):
    cx, cy, height = centroid
    mics = np.array([
        [cx - 0.08,  cy,        height + 0.03],  # ch 0 - left_top
        [cx - 0.08,  cy + 0.02, height       ],  # ch 1 - left_front
        [cx - 0.08,  cy - 0.02, height       ],  # ch 2 - left_back
        [cx + 0.08,  cy - 0.02, height       ],  # ch 3 - right_back
        [cx + 0.08,  cy + 0.02, height       ],  # ch 4 - right_front
        [cx + 0.08,  cy,        height + 0.03]   # ch 5 - right_top
    ])
    return mics


# ==========================================
# SI-SNR -- IDENTICAL to run_mvdr.py
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
# CORE ENGINE -- TRUE PHASE-ALIGNED DELAY-AND-SUM
# ==========================================
def process_single_sample(sample_idx):
    try:
        meta_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_meta.json")
        mix_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_mix.wav")
        tgt_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_target_gt.wav")
        out_path = os.path.join(OUTPUT_FOLDER, f"enhanced_sample_{sample_idx}_mix.wav")

        if not (os.path.exists(meta_path) and os.path.exists(mix_path) and os.path.exists(tgt_path)):
            return {"sample_idx": sample_idx, "status": "missing_files"}

        with open(meta_path, "r") as f:
            meta = json.load(f)

        centroid = meta["headset_centroid"]
        target_pos = np.array(meta["target_position"])
        num_interferers = meta.get("num_interferers", None)
        mic_positions = reconstruct_mic_positions(centroid)

        mix, sr = sf.read(mix_path, frames=64000, dtype='float32')
        tgt, _ = sf.read(tgt_path, frames=64000, dtype='float32')

        mix_ref = mix[:, REFERENCE_CHANNEL]
        tgt_ref = tgt[:, REFERENCE_CHANNEL]
        sisnr_in = compute_si_snr_numpy(mix_ref, tgt_ref)

        # STFT -- IDENTICAL params to run_mvdr.py
        f, t, stft_mix = signal.stft(mix.T, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
        n_channels, n_bins, n_frames = stft_mix.shape

        # STEERING VECTOR -- IDENTICAL math to run_mvdr.py
        absolute_distances = np.linalg.norm(mic_positions - target_pos, axis=1)
        absolute_delays = absolute_distances / SPEED_OF_SOUND
        relative_delays = absolute_delays - absolute_delays[REFERENCE_CHANNEL]
        freqs = np.arange(n_bins) * (SAMPLE_RATE / FFT_LENGTH)
        steering_vector = np.exp(-1j * 2 * np.pi * freqs[None, :] * relative_delays[:, None])
        if sample_idx < 5:
            print(f"\nsample {sample_idx}:")
            print(f"  centroid = {centroid}")
            print(f"  target_position = {target_pos}")
            print(f"  mic_positions =\n{mic_positions}")
            print(f"  absolute_distances (m) = {absolute_distances}")
            print(f"  absolute_delays (s) = {absolute_delays}")
            print(f"  relative_delays (s) = {relative_delays}")
        # steering_vector shape: (n_channels, n_bins)

        # ---- TRUE DELAY-AND-SUM WEIGHTS ----
        # w_das = conj(d) / ||d||^2 normalization is not needed for d with
        # unit-magnitude entries (pure phase steering vector): w = conj(d)/N
        # This applies the EXACT SAME phase compensation that MVDR's
        # constraint w^H d = 1 would apply if R were a scaled identity --
        # i.e. this is what MVDR degenerates to when noise is spatially white.
        w_das = np.conj(steering_vector) / n_channels  # shape (n_channels, n_bins)

        # Apply to every frame at once (static weights -- no adaptation)
        # stft_mix: (n_channels, n_bins, n_frames)
        enhanced_spec = np.einsum('cf,cft->ft', w_das, stft_mix)  # (n_bins, n_frames)

        _, enhanced_waveform = signal.istft(enhanced_spec, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
        enhanced_waveform = enhanced_waveform[:len(tgt_ref)]

        sisnr_out = compute_si_snr_numpy(enhanced_waveform, tgt_ref)

        sf.write(out_path, enhanced_waveform, SAMPLE_RATE)

        return {
            "sample_idx": sample_idx,
            "status": "success",
            "num_interferers": num_interferers,
            "sisnr_input": float(sisnr_in),
            "sisnr_output": float(sisnr_out),
            "sisnr_improvement": float(sisnr_out - sisnr_in)
        }

    except Exception as e:
        return {"sample_idx": sample_idx, "status": f"failed: {str(e)}"}


# ==========================================
# ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    NUM_SAMPLES = 1000
    sample_indices = list(range(NUM_SAMPLES))

    num_workers = os.cpu_count()
    print(f"Running TRUE phase-aligned delay-and-sum across {num_workers} cores...")
    print("(static steering-vector weights, no adaptive covariance / no nulling)\n")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list_of_results = list(tqdm(
            executor.map(process_single_sample, sample_indices),
            total=NUM_SAMPLES,
            desc="DSBF (phase-aligned, static)"
        ))

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
                "num_interferers": res["num_interferers"],
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

    print("\n" + "-" * 40 + "\n Summary (TRUE phase-aligned DSBF) \n" + "-" * 40)
    print(f"  Mean SI-SNR input       : {mean_in:.2f} dB")
    print(f"  Mean SI-SNR output      : {mean_out:.2f} dB")
    print(f"  Mean SI-SNR improvement : {mean_imp:.2f} dB")
    print(f"  Samples processed       : {valid_scenes}")
    print(f"\nMetrics saved -> {METRICS_PATH}")