import numpy as np
import soundfile as sf
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_baseline import run_offline, compute_si_snr_numpy
from streaming_mvdr import StreamingMVDR, build_steering_vector, FFT_LENGTH, FFT_SHIFT, SAMPLE_RATE

# Point these at your actual dataset location, e.g.:
# DATA_DIR = "C:/projects/headset-spatial-magnifier/data/generated_dataset"
DATA_DIR = os.environ.get("MVDR_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
MIX_PATH = os.path.join(DATA_DIR, "sample_0_mix.wav")
TGT_PATH = os.path.join(DATA_DIR, "sample_0_target_gt.wav")


def main():
    print("=" * 60)
    print("STEP 1: Run offline baseline (ground truth)")
    print("=" * 60)
    offline = run_offline(MIX_PATH, TGT_PATH, apply_peak_norm=False)
    print(f"Offline SI-SNR input:  {offline['sisnr_input']:.4f} dB")
    print(f"Offline SI-SNR output: {offline['sisnr_output']:.4f} dB")
    offline_wave = offline["enhanced_waveform_unnorm"]

    print()
    print("=" * 60)
    print("STEP 2: Run streaming version, fed in real-time-sized chunks")
    print("=" * 60)

    mix, sr = sf.read(MIX_PATH, frames=64000, dtype='float64')
    tgt, _ = sf.read(TGT_PATH, frames=64000, dtype='float64')
    tgt_ref = tgt[:, 1]

    steering_vector = build_steering_vector()
    bf = StreamingMVDR(steering_vector=steering_vector, apply_limiter=False)

    # Simulate a real audio callback delivering small chunks (e.g. 128 samples
    # at a time -- smaller than the hop size, to prove the class correctly
    # buffers partial hops, not just exact-hop-sized chunks).
    CALLBACK_CHUNK = 128
    n_samples = mix.shape[0]

    streaming_output_chunks = []
    pos = 0
    while pos < n_samples:
        chunk = mix[pos:pos + CALLBACK_CHUNK, :]
        pos += CALLBACK_CHUNK
        out = bf.process(chunk)
        if len(out) > 0:
            streaming_output_chunks.append(out)

    streaming_wave = np.concatenate(streaming_output_chunks) if streaming_output_chunks else np.zeros(0)
    print(f"Streaming output length: {len(streaming_wave)} samples")
    print(f"Offline output length:   {len(offline_wave)} samples")

    print()
    print("=" * 60)
    print("STEP 3: Compare, accounting for fixed algorithmic latency")
    print("=" * 60)
    latency = bf.latency_samples
    print(f"Expected algorithmic latency: {latency} samples ({latency/SAMPLE_RATE*1000:.1f} ms)")

    # The streaming output is causal and delayed; offline output is
    # acausal-corrected (scipy strips the front padding). Align by shifting
    # the streaming output back by `latency`.
    streaming_aligned = streaming_wave[latency:]

    min_len = min(len(streaming_aligned), len(offline_wave))
    diff = np.abs(streaming_aligned[:min_len] - offline_wave[:min_len])
    print(f"Comparing {min_len} aligned samples")
    print(f"Max abs diff:  {np.max(diff):.6e}")
    print(f"Mean abs diff: {np.mean(diff):.6e}")
    print(f"Median abs diff: {np.median(diff):.6e}")

    # Show where in the signal the largest errors occur (start, middle, end)
    print(f"\nFirst 10 samples -- streaming vs offline:")
    print("streaming:", streaming_aligned[:10])
    print("offline:  ", offline_wave[:10])

    print(f"\nMiddle 10 samples (idx {min_len//2}) -- streaming vs offline:")
    mid = min_len // 2
    print("streaming:", streaming_aligned[mid:mid+10])
    print("offline:  ", offline_wave[mid:mid+10])

    print()
    print("=" * 60)
    print("STEP 4: SI-SNR comparison (streaming vs offline-unnormalized)")
    print("=" * 60)
    # Compare streaming output's SI-SNR against the target, aligned for latency
    tgt_ref_aligned = tgt_ref[:len(streaming_aligned)]
    sisnr_streaming = compute_si_snr_numpy(streaming_aligned, tgt_ref_aligned)
    print(f"Streaming SI-SNR output (vs target, latency-aligned): {sisnr_streaming:.4f} dB")
    print(f"Offline SI-SNR output (unnormalized):                 {offline['sisnr_output']:.4f} dB")

    sf.write(os.path.join(DATA_DIR, "streaming_enhanced.wav"), streaming_wave, SAMPLE_RATE)
    sf.write(os.path.join(DATA_DIR, "streaming_enhanced_aligned.wav"), streaming_aligned, SAMPLE_RATE)


if __name__ == "__main__":
    main()
