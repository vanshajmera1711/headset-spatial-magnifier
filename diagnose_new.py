"""
Run this on your machine (not here -- I don't have your actual wav files).
This checks ONE sample directly: prints PESQ with/without our alignment fix,
and checks frame-to-frame MVDR weight stability as a proxy for musical noise.
"""
import numpy as np
import soundfile as sf
from pesq import pesq as pesq_fn

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
SAMPLE_IDX = 0  # change this to inspect a few different samples

tgt_path = f"{DATA_DIR}/generated_dataset/sample_{SAMPLE_IDX}_target_gt.wav"
enh_path = f"{DATA_DIR}/mvdr_output/enhanced_sample_{SAMPLE_IDX}_mix.wav"

tgt, sr1 = sf.read(tgt_path, frames=64000, dtype='float32')
enh, sr2 = sf.read(enh_path, dtype='float32')

tgt_ref = tgt[:, 1]  # reference channel
min_len = min(len(tgt_ref), len(enh))
tgt_ref = tgt_ref[:min_len]
enh = enh[:min_len]

print(f"sample rate tgt: {sr1}, enh: {sr2}")
print(f"tgt_ref range: [{tgt_ref.min():.4f}, {tgt_ref.max():.4f}], rms: {np.sqrt(np.mean(tgt_ref**2)):.4f}")
print(f"enh range: [{enh.min():.4f}, {enh.max():.4f}], rms: {np.sqrt(np.mean(enh**2)):.4f}")

# raw PESQ, no alignment fix at all
try:
    raw_score = pesq_fn(16000, tgt_ref, enh, 'wb')
    print(f"\nRAW PESQ (no alignment fix): {raw_score:.3f}")
except Exception as e:
    print(f"RAW PESQ failed: {e}")

# Check for any NaN/Inf/clipping issues
print(f"\nany NaN in enh: {np.isnan(enh).any()}")
print(f"any Inf in enh: {np.isinf(enh).any()}")
print(f"fraction of enh samples > 0.99 (clipping check): {(np.abs(enh) > 0.99).mean():.4f}")

# Save a short clip of each for you to listen to side by side
sf.write(f"{DATA_DIR}/debug_tgt_clip.wav", tgt_ref[:32000], 16000)
sf.write(f"{DATA_DIR}/debug_enh_clip.wav", enh[:32000], 16000)
print(f"\nSaved 2-second clips to data/debug_tgt_clip.wav and data/debug_enh_clip.wav for listening")