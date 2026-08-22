import numpy as np
import soundfile as sf
import scipy.signal as signal

SAMPLE_RATE = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256
N_BINS = FFT_LENGTH // 2 + 1
REFERENCE_CHANNEL = 1
SPEED_OF_SOUND = 343.0


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


def run_offline(mix_path, tgt_path, max_frames_samples=64000, apply_peak_norm=True):
    cx, cy, height = 0.0, 0.0, 0.0
    mic_positions = np.array([
        [cx - 0.08, cy, height + 0.03],
        [cx - 0.08, cy + 0.02, height],
        [cx - 0.08, cy - 0.02, height],
        [cx + 0.08, cy - 0.02, height],
        [cx + 0.08, cy + 0.02, height],
        [cx + 0.08, cy, height + 0.03]
    ])
    target_pos = np.array([0.0, 0.12, -0.10])

    mix, sr = sf.read(mix_path, frames=max_frames_samples, dtype='float32')
    tgt, _ = sf.read(tgt_path, frames=max_frames_samples, dtype='float32')

    mix_ref = mix[:, REFERENCE_CHANNEL]
    tgt_ref = tgt[:, REFERENCE_CHANNEL]
    sisnr_in = compute_si_snr_numpy(mix_ref, tgt_ref)

    f, t, stft_mix = signal.stft(mix.T, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
    n_channels, n_bins, n_frames = stft_mix.shape

    absolute_distances = np.linalg.norm(mic_positions - target_pos, axis=1)
    absolute_delays = absolute_distances / SPEED_OF_SOUND
    relative_delays = absolute_delays - absolute_delays[REFERENCE_CHANNEL]
    relative_amplitude = absolute_distances[REFERENCE_CHANNEL] / absolute_distances
    freqs = np.arange(n_bins) * (SAMPLE_RATE / FFT_LENGTH)
    steering_vector = (
        relative_amplitude[:, None]
        * np.exp(-1j * 2 * np.pi * freqs[None, :] * relative_delays[:, None])
    )

    R_noise = np.zeros((n_channels, n_channels, n_bins), dtype=np.complex64)
    for b in range(n_bins):
        R_noise[:, :, b] = np.eye(n_channels) * 1e-5

    power_fast = np.zeros(n_bins)
    power_slow = np.zeros(n_bins)
    noise_mask_smoothed = np.zeros(n_bins, dtype=np.float32)
    w_smoothed = None
    beta = 0.65

    enhanced_spec = np.zeros((n_bins, n_frames), dtype=np.complex64)
    I_batch = np.tile(np.eye(n_channels)[None, :, :], (n_bins, 1, 1))

    # Debug capture for validation against streaming version
    debug_log = {"w_smoothed": [], "R_noise_trace": [], "enhanced_spec_frames": []}

    for t_idx in range(n_frames):
        X_t = stft_mix[:, :, t_idx]

        current_energy = np.abs(X_t[REFERENCE_CHANNEL, :]) ** 2
        EPSILON_FLOOR = 1e-10

        if t_idx == 0:
            power_fast = current_energy + EPSILON_FLOOR
            power_slow = current_energy + EPSILON_FLOOR
        else:
            power_fast = 0.4 * current_energy + (1 - 0.4) * power_fast
            power_slow = 0.98 * current_energy + (1 - 0.98) * power_slow
            power_slow = np.maximum(power_slow, EPSILON_FLOOR)
            power_fast = np.maximum(power_fast, EPSILON_FLOOR)

        X_outer = np.einsum('if,jf->ijf', X_t, np.conj(X_t))

        noise_mask_instant = (power_fast <= 1.02 * power_slow).astype(np.float32)

        if t_idx == 0:
            noise_mask_smoothed = noise_mask_instant
        else:
            noise_mask_smoothed = np.maximum(noise_mask_instant, 0.7 * noise_mask_smoothed)

        alpha_tensor = 0.95 * noise_mask_smoothed + 1.0 * (1 - noise_mask_smoothed)
        R_noise = alpha_tensor[None, None, :] * R_noise + (1 - alpha_tensor[None, None, :]) * X_outer

        R_batch = np.moveaxis(R_noise, 2, 0)

        eps = np.maximum(1e-5 * np.mean(np.abs(R_batch)), 1e-7)
        R_batch = R_batch + eps * I_batch

        R_inv_batch = np.linalg.pinv(R_batch)

        d_batch = np.moveaxis(steering_vector, 1, 0)[:, :, None]
        numerator_batch = R_inv_batch @ d_batch

        d_H_batch = np.conj(np.moveaxis(d_batch, 2, 1))
        denominator_batch = d_H_batch @ numerator_batch

        w_instant = np.squeeze(numerator_batch / (denominator_batch + 1e-8), axis=-1)

        if t_idx == 0:
            w_smoothed = w_instant
        else:
            w_smoothed = beta * w_smoothed + (1 - beta) * w_instant

        w_filtered_spec = np.einsum('fi,if->f', np.conj(w_smoothed), X_t)
        enhanced_spec[:, t_idx] = w_filtered_spec

        debug_log["w_smoothed"].append(w_smoothed.copy())
        debug_log["enhanced_spec_frames"].append(w_filtered_spec.copy())

    _, enhanced_waveform = signal.istft(enhanced_spec, fs=SAMPLE_RATE, nperseg=FFT_LENGTH, noverlap=FFT_SHIFT)
    enhanced_waveform_unnorm = enhanced_waveform[:len(tgt_ref)].copy()

    enhanced_waveform = enhanced_waveform_unnorm.copy()
    if apply_peak_norm:
        max_val = np.max(np.abs(enhanced_waveform))
        if max_val > 0:
            enhanced_waveform = (enhanced_waveform / max_val) * 0.89

    sisnr_out = compute_si_snr_numpy(enhanced_waveform, tgt_ref)

    return {
        "sisnr_input": float(sisnr_in),
        "sisnr_output": float(sisnr_out),
        "enhanced_waveform": enhanced_waveform,
        "enhanced_waveform_unnorm": enhanced_waveform_unnorm,
        "enhanced_spec": enhanced_spec,
        "stft_mix": stft_mix,
        "steering_vector": steering_vector,
        "debug_log": debug_log,
        "tgt_ref": tgt_ref,
    }


if __name__ == "__main__":
    res = run_offline("data/sample_0_mix.wav", "data/sample_0_target_gt.wav")
    print(f"SI-SNR input:  {res['sisnr_input']:.3f} dB")
    print(f"SI-SNR output: {res['sisnr_output']:.3f} dB")
    sf.write("data/offline_enhanced_unnorm.wav", res["enhanced_waveform_unnorm"], SAMPLE_RATE)
    sf.write("data/offline_enhanced.wav", res["enhanced_waveform"], SAMPLE_RATE)
    np.save("data/offline_enhanced_unnorm.npy", res["enhanced_waveform_unnorm"])
