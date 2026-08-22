"""
Extended evaluation suite for the headset spatial magnifier project.

Adds two categories of analysis beyond plain SI-SNR:

  1. PERCEPTUAL / SEPARATION METRICS
     - PESQ          (perceived quality, MOS-like)
     - STOI          (intelligibility)
     - SDR/SIR/SAR   (overall distortion / interferer suppression / artifacts)

  2. GEOMETRIC BREAKDOWNS (uses meta.json -- no new data generation needed)
     - SI-SNR improvement vs. closest-interferer angular separation from target
     - SI-SNR improvement vs. interferer azimuth (relative to head axis)
     - SI-SNR improvement vs. interferer distance from headset centroid
     - SI-SNR improvement vs. num_interferers (already had this, kept for completeness)

Run from: C:/projects/headset-spatial-magnifier

    pip install pesq pystoi mir_eval matplotlib --break-system-packages
    python evaluate_extended.py

Notes:
  - PESQ requires 8k or 16k sample rate (we're at 16k, fine) and is picky
    about exact sample alignment -- this script handles cropping safely.
  - mir_eval's bss_eval_sources is the simplest way to get SDR/SIR/SAR
    without hand-rolling the projection math. It expects (n_sources, n_samples)
    arrays of TIME-DOMAIN reference signals. For our 1-target setup we treat
    "everything that isn't the target" (interferers + noise, reconstructed as
    mix - target_gt) as a second "source" so SIR/SAR are still meaningful.
"""

import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset_reverb")
ENHANCED_FOLDER = os.path.join(DATA_DIR, "mvdr_output")   # enhanced_sample_N_mix.wav
METRICS_OUT = os.path.join(DATA_DIR, "extended_metrics.json")
PLOTS_DIR = os.path.join(DATA_DIR, "evaluation_plots")

SAMPLE_RATE = 16000
REFERENCE_CHANNEL = 1
CROP_LEN = 64000

os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Optional imports -- degrade gracefully if not installed, rather than
# crashing the whole script. We tell you exactly what to pip install.
# --------------------------------------------------------------------------
HAVE_PESQ = True
try:
    from pesq import pesq as pesq_fn
except ImportError:
    HAVE_PESQ = False

HAVE_STOI = True
try:
    from pystoi import stoi as stoi_fn
except ImportError:
    HAVE_STOI = False

HAVE_MIR_EVAL = True
try:
    from mir_eval.separation import bss_eval_sources
except ImportError:
    HAVE_MIR_EVAL = False

if not HAVE_PESQ:
    print("WARNING: 'pesq' not installed -- skipping PESQ. "
          "Install with: pip install pesq --break-system-packages")
if not HAVE_STOI:
    print("WARNING: 'pystoi' not installed -- skipping STOI. "
          "Install with: pip install pystoi --break-system-packages")
if not HAVE_MIR_EVAL:
    print("WARNING: 'mir_eval' not installed -- skipping SDR/SIR/SAR. "
          "Install with: pip install mir_eval --break-system-packages")


# ==========================================
# GEOMETRY HELPERS
# ==========================================
def reconstruct_mic_positions(centroid):
    cx, cy, height = centroid
    mics = np.array([
        [cx - 0.08,  cy,        height + 0.03],
        [cx - 0.08,  cy + 0.02, height       ],
        [cx - 0.08,  cy - 0.02, height       ],
        [cx + 0.08,  cy - 0.02, height       ],
        [cx + 0.08,  cy + 0.02, height       ],
        [cx + 0.08,  cy,        height + 0.03]
    ])
    return mics


def azimuth_relative_to_head(centroid, position):
    """
    Azimuth angle (degrees, 0-180) of `position` relative to the headset
    centroid, measured against the head's left-right (x) axis.
    0 deg  = directly left/right (broadside -- easiest to localize/null)
    90 deg = directly front/back (on-axis -- hardest, matches target's
             fixed position, minimal inter-aural difference)

    We use atan2 on the horizontal plane (x = left-right, y = front-back),
    ignoring height (z), since that's the dominant cue for a horizontally
    arranged binaural-style array.
    """
    dx = position[0] - centroid[0]
    dy = position[1] - centroid[1]
    # angle from the x-axis (left-right): 0 = broadside, 90 = on-axis (front/back)
    angle_from_x_axis = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-8))
    return angle_from_x_axis


def angular_separation(centroid, pos_a, pos_b):
    """
    Angular separation (degrees, 0-180) between two positions as seen
    from the headset centroid -- i.e. how far apart the target and an
    interferer appear, angularly, from the array's point of view.
    """
    va = np.array(pos_a) - np.array(centroid)
    vb = np.array(pos_b) - np.array(centroid)
    va = va / (np.linalg.norm(va) + 1e-8)
    vb = vb / (np.linalg.norm(vb) + 1e-8)
    cos_angle = np.clip(np.dot(va, vb), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def distance(centroid, position):
    return float(np.linalg.norm(np.array(position) - np.array(centroid)))


def compute_pesq_aligned(reference, estimate, sample_rate, max_lag=80):
    """
    PESQ is sensitive to small constant time shifts and level mismatches
    that SI-SNR (scale-invariant) and STOI (correlation-windowed) largely
    absorb but PESQ's perceptual model does not. The STFT/ISTFT round trip
    in run_mvdr.py can introduce a few samples of group delay, which is
    enough to tank PESQ scores even when the audio is genuinely listenable.

    This fixes both before scoring:
      1. Cross-correlation to find and correct the best integer-sample lag
         between estimate and reference (search window: +/- max_lag samples,
         generous for a few-sample STFT/ISTFT group delay).
      2. Least-squares level matching (same scale-fix SI-SNR already does
         internally) so PESQ isn't penalizing a pure gain mismatch.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    # ---- 1. Find best lag via cross-correlation over a small window ----
    # Only search +/- max_lag samples -- this is meant to correct a few
    # samples of STFT/ISTFT group delay, not do general-purpose alignment.
    best_lag = 0
    best_score = -np.inf
    search_ref = reference[: min(len(reference), 16000)]   # first 1s @16kHz is enough to find lag
    search_est = estimate[: min(len(estimate), 16000)]
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = search_ref[lag:]
            b = search_est[: len(a)]
        else:
            b = search_est[-lag:]
            a = search_ref[: len(b)]
        if len(a) < 100:
            continue
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        score = np.dot(a, b) / denom
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag >= 0:
        ref_aligned = reference[best_lag:]
        est_aligned = estimate[: len(ref_aligned)]
    else:
        est_aligned = estimate[-best_lag:]
        ref_aligned = reference[: len(est_aligned)]

    min_len = min(len(ref_aligned), len(est_aligned))
    ref_aligned = ref_aligned[:min_len]
    est_aligned = est_aligned[:min_len]

    # ---- 2. Least-squares level match (same scale-fix SI-SNR uses) ----
    ref_pow = np.sum(ref_aligned ** 2) + 1e-8
    scale = np.sum(est_aligned * ref_aligned) / ref_pow
    if abs(scale) > 1e-8:
        est_aligned = est_aligned / scale  # bring estimate to reference's level

    # ---- 3. Normalize to a safe peak level for PESQ's expected range ----
    peak = max(np.max(np.abs(ref_aligned)), np.max(np.abs(est_aligned)), 1e-8)
    if peak > 1.0:
        ref_aligned = ref_aligned / peak
        est_aligned = est_aligned / peak

    return float(pesq_fn(sample_rate, ref_aligned.astype(np.float32), est_aligned.astype(np.float32), 'wb'))


# ==========================================
# MAIN EVALUATION LOOP
# ==========================================
def evaluate_single_sample(idx):
    """Evaluate one sample. Returns a result dict, or a dict with 'skip': True."""
    try:
        meta_path = os.path.join(INPUT_FOLDER, f"sample_{idx}_meta.json")
        mix_path = os.path.join(INPUT_FOLDER, f"sample_{idx}_mix.wav")
        tgt_path = os.path.join(INPUT_FOLDER, f"sample_{idx}_target_gt.wav")
        enh_path = os.path.join(ENHANCED_FOLDER, f"enhanced_sample_{idx}_mix.wav")

        if not all(os.path.exists(p) for p in [meta_path, mix_path, tgt_path, enh_path]):
            return {"skip": True, "sample_idx": idx, "reason": "missing_files"}

        with open(meta_path, "r") as f:
            meta = json.load(f)

        centroid = meta["headset_centroid"]
        target_pos = meta["target_position"]
        interferer_positions = meta.get("interferer_positions", [])
        num_interferers = meta.get("num_interferers", len(interferer_positions))

        # ---- Geometric features ----
        if len(interferer_positions) > 0:
            separations = [angular_separation(centroid, target_pos, ip) for ip in interferer_positions]
            closest_separation = min(separations)
            azimuths = [azimuth_relative_to_head(centroid, ip) for ip in interferer_positions]
            mean_interferer_azimuth = float(np.mean(azimuths))
            distances = [distance(centroid, ip) for ip in interferer_positions]
            closest_interferer_distance = min(distances)
        else:
            closest_separation = None
            mean_interferer_azimuth = None
            closest_interferer_distance = None

        target_distance = distance(centroid, target_pos)

        # ---- Load audio ----
        mix, _ = sf.read(mix_path, frames=CROP_LEN, dtype='float32')
        tgt, _ = sf.read(tgt_path, frames=CROP_LEN, dtype='float32')
        enh, _ = sf.read(enh_path, dtype='float32')

        min_len = min(len(tgt), len(enh), len(mix))
        tgt_ref = tgt[:min_len, REFERENCE_CHANNEL]
        mix_ref = mix[:min_len, REFERENCE_CHANNEL]
        enh = enh[:min_len] if enh.ndim == 1 else enh[:min_len, 0]

        # interferer+noise reference = mix - target (residual), for SIR/SAR
        residual_ref = mix_ref - tgt_ref

        result = {
            "skip": False,
            "sample_idx": idx,
            "num_interferers": num_interferers,
            "target_distance_m": round(target_distance, 4),
            "closest_interferer_angular_sep_deg": round(closest_separation, 2) if closest_separation is not None else None,
            "mean_interferer_azimuth_deg": round(mean_interferer_azimuth, 2) if mean_interferer_azimuth is not None else None,
            "closest_interferer_distance_m": round(closest_interferer_distance, 4) if closest_interferer_distance is not None else None,
        }

        if HAVE_PESQ:
                try:
                    # Bypass the custom alignment wrapper to test raw file alignment
                    from pesq import pesq
                    pesq_score = pesq(SAMPLE_RATE, tgt_ref, enh, 'wb')
                    result["pesq"] = pesq_score
                except Exception:
                    result["pesq"] = None
            # ---- STOI ----
        if HAVE_STOI:
            try:
                result["stoi"] = float(stoi_fn(tgt_ref, enh, SAMPLE_RATE, extended=False))
            except Exception:
                result["stoi"] = None

        # ---- SDR / SIR / SAR via mir_eval ----
        if HAVE_MIR_EVAL:
            try:
                # Standard approach for a single estimated source: treat the
                # mixture of (target, residual) as the 2-source ground truth
                # and the single enhanced output as the estimate of source 0
                # (target); mir_eval supports this directly via bss_eval_sources.
                ref_sources = np.stack([tgt_ref, residual_ref], axis=0)
                est_sources = np.stack([enh, mix_ref - enh], axis=0)
                sdr, sir, sar, _ = bss_eval_sources(ref_sources, est_sources)
                result["sdr"] = float(sdr[0])
                result["sir"] = float(sir[0])
                result["sar"] = float(sar[0])
            except Exception:
                result["sdr"] = None
                result["sir"] = None
                result["sar"] = None

        return result

    except Exception as e:
        return {"skip": True, "sample_idx": idx, "reason": str(e)}


def main():
    sample_files = [f for f in os.listdir(ENHANCED_FOLDER) if f.startswith("enhanced_sample_")]
    sample_indices = sorted(
        int(f.replace("enhanced_sample_", "").replace("_mix.wav", ""))
        for f in sample_files
    )

    print(f"Found {len(sample_indices)} enhanced samples to evaluate.\n")

    num_workers = os.cpu_count()
    print(f"Running extended evaluation across {num_workers} cores...\n")

    per_sample_results = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(
            executor.map(evaluate_single_sample, sample_indices),
            total=len(sample_indices),
            desc="Extended eval (PESQ/STOI/SDR/SIR/SAR + geometry)"
        ):
            if res.get("skip"):
                skipped += 1
                continue
            per_sample_results.append(res)

    print(f"\nEvaluated {len(per_sample_results)} samples successfully. ({skipped} skipped)\n")

    with open(METRICS_OUT, "w") as f:
        json.dump(per_sample_results, f, indent=2)
    print(f"Saved per-sample extended metrics -> {METRICS_OUT}\n")

    # ====================================================
    # SUMMARY STATS
    # ====================================================
    def safe_mean(key):
        vals = [r[key] for r in per_sample_results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    print("=" * 60)
    print(" Aggregate metrics")
    print("=" * 60)
    for key in ["pesq", "stoi", "sdr", "sir", "sar"]:
        m = safe_mean(key)
        if m is not None:
            print(f"  Mean {key.upper():6s}: {m:.3f}")
    print("=" * 60)

    # ====================================================
    # PLOTS
    # ====================================================
    make_plots(per_sample_results)
    print(f"\nPlots saved to: {PLOTS_DIR}")


def make_plots(results):
    # Filter to samples that actually have geometric + sdr data
    has_geom = [r for r in results if r["closest_interferer_angular_sep_deg"] is not None]

    if not has_geom:
        print("No samples with interferer geometry found -- skipping geometric plots.")
        return

    # Pick whichever quality metric is available, prefer SDR, fallback SIR, fallback PESQ
    metric_key = None
    for candidate in ["sdr", "sir", "pesq", "stoi"]:
        if any(r.get(candidate) is not None for r in has_geom):
            metric_key = candidate
            break

    if metric_key is None:
        print("No quality metric available for plotting (install pesq/pystoi/mir_eval).")
        return

    def scatter_plot(x_key, x_label, filename, title):
        xs = [r[x_key] for r in has_geom if r.get(metric_key) is not None]
        ys = [r[metric_key] for r in has_geom if r.get(metric_key) is not None]
        if not xs:
            return
        plt.figure(figsize=(7, 5))
        plt.scatter(xs, ys, alpha=0.4, s=18)
        # trendline
        if len(xs) > 5:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 100)
            plt.plot(x_line, p(x_line), color='red', linewidth=2, label=f"trend (slope={z[0]:.4f})")
            plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(metric_key.upper())
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=140)
        plt.close()

    scatter_plot(
        "closest_interferer_angular_sep_deg",
        "Angular separation: target vs. closest interferer (deg)",
        "metric_vs_angular_separation.png",
        f"{metric_key.upper()} vs. Target-Interferer Angular Separation"
    )

    scatter_plot(
        "mean_interferer_azimuth_deg",
        "Mean interferer azimuth from head axis (deg, 0=broadside, 90=on-axis)",
        "metric_vs_interferer_azimuth.png",
        f"{metric_key.upper()} vs. Interferer Azimuth"
    )

    scatter_plot(
        "closest_interferer_distance_m",
        "Closest interferer distance from headset (m)",
        "metric_vs_interferer_distance.png",
        f"{metric_key.upper()} vs. Closest Interferer Distance"
    )

    # Boxplot-style by num_interferers
    by_count = {}
    for r in has_geom:
        if r.get(metric_key) is None:
            continue
        by_count.setdefault(r["num_interferers"], []).append(r[metric_key])

    if by_count:
        counts = sorted(by_count.keys())
        data = [by_count[c] for c in counts]
        plt.figure(figsize=(7, 5))
        plt.boxplot(data, labels=[str(c) for c in counts])
        plt.xlabel("num_interferers")
        plt.ylabel(metric_key.upper())
        plt.title(f"{metric_key.upper()} by Number of Interferers")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "metric_vs_num_interferers_boxplot.png"), dpi=140)
        plt.close()


if __name__ == "__main__":
    main()