"""
Diagnostic script — run before MVDR to check dataset sanity.
Usage: python diagnose.py
"""
import os
import glob
import numpy as np
import soundfile as sf

INPUT_DIR = "data/simulated"
REF_MIC = 1

mix_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_mix.wav")))[:5]

for mix_path in mix_files:
    sample_id   = os.path.basename(mix_path).replace("_mix.wav", "")
    target_path = os.path.join(INPUT_DIR, f"{sample_id}_target_gt.wav")

    mix,    _ = sf.read(mix_path)
    target, _ = sf.read(target_path)

    mix_ref    = mix[:, REF_MIC]
    target_ref = target[:, REF_MIC]

    # Basic energy check
    mix_rms    = np.sqrt(np.mean(mix_ref**2))
    target_rms = np.sqrt(np.mean(target_ref**2))

    # What fraction of mix energy is target?
    ratio_db = 20 * np.log10(target_rms / (mix_rms + 1e-8))

    # Are mix and target correlated? (target should be present in mix)
    corr = np.corrcoef(mix_ref, target_ref)[0, 1]

    # Is target_ref a subset of mix_ref? Compute residual
    # mix = target + noise+interf  =>  residual = mix - target  should have lower energy than mix
    residual = mix_ref - target_ref
    residual_rms = np.sqrt(np.mean(residual**2))

    print(f"\n--- {sample_id} ---")
    print(f"  mix RMS:         {mix_rms:.4f}")
    print(f"  target RMS:      {target_rms:.4f}")
    print(f"  target/mix ratio:{ratio_db:.1f} dB")
    print(f"  correlation:     {corr:.4f}  (should be positive, ideally > 0.3)")
    print(f"  residual RMS:    {residual_rms:.4f}  (mix - target; should be < mix RMS if target is in mix)")
    print(f"  mix shape:       {mix.shape}")
    print(f"  target shape:    {target.shape}")
    print(f"  mix peak:        {np.max(np.abs(mix_ref)):.4f}")
    print(f"  target peak:     {np.max(np.abs(target_ref)):.4f}")
