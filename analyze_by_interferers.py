"""
Diagnostic: breaks down mvdr_metrics.json sisnr_improvement by num_interferers,
using the meta.json files in generated_dataset.

This does NOT change the beamformer at all — it's pure analysis to test the
hypothesis that R_noise is learning ambient noise structure but failing to
null directional interferers (which would show up as improvement dropping
as num_interferers increases).

Run from: C:/projects/headset-spatial-magnifier
    python analyze_by_interferers.py
"""

import os
import json
from collections import defaultdict

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset")
METRICS_PATH = os.path.join(DATA_DIR, "mvdr_metrics.json")


def main():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    per_file = metrics["per_file_metrics"]

    # group sample improvements by num_interferers
    groups = defaultdict(list)
    missing_meta = 0

    for sample_key, vals in per_file.items():
        # sample_key looks like "sample_123"
        sample_idx = sample_key.replace("sample_", "")
        meta_path = os.path.join(INPUT_FOLDER, f"sample_{sample_idx}_meta.json")

        if not os.path.exists(meta_path):
            missing_meta += 1
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        n_interf = meta.get("num_interferers", None)
        if n_interf is None:
            continue

        groups[n_interf].append({
            "sample_idx": sample_idx,
            "sisnr_input": vals["sisnr_input"],
            "sisnr_output": vals["sisnr_output"],
            "sisnr_improvement": vals["sisnr_improvement"],
        })

    if missing_meta:
        print(f"WARNING: {missing_meta} samples had no matching meta.json (skipped)\n")

    print("=" * 70)
    print(" SI-SNR improvement broken down by num_interferers")
    print("=" * 70)
    print(f"{'num_interferers':>16} | {'count':>6} | {'mean_in':>9} | {'mean_out':>9} | {'mean_improve':>13}")
    print("-" * 70)

    overall_n = 0
    overall_improve_sum = 0.0

    for n_interf in sorted(groups.keys()):
        rows = groups[n_interf]
        count = len(rows)
        mean_in = sum(r["sisnr_input"] for r in rows) / count
        mean_out = sum(r["sisnr_output"] for r in rows) / count
        mean_improve = sum(r["sisnr_improvement"] for r in rows) / count

        overall_n += count
        overall_improve_sum += sum(r["sisnr_improvement"] for r in rows)

        print(f"{n_interf:>16} | {count:>6} | {mean_in:>9.3f} | {mean_out:>9.3f} | {mean_improve:>13.3f}")

    print("-" * 70)
    if overall_n > 0:
        print(f"{'OVERALL':>16} | {overall_n:>6} | {'':>9} | {'':>9} | {overall_improve_sum/overall_n:>13.3f}")
    print("=" * 70)

    # Also: correlation-style check — sort all samples by num_interferers,
    # print min/max improvement per group to see spread, not just mean
    print("\nSpread per group (min / median / max improvement):")
    for n_interf in sorted(groups.keys()):
        vals = sorted(r["sisnr_improvement"] for r in groups[n_interf])
        n = len(vals)
        median = vals[n // 2]
        print(f"  num_interferers={n_interf}: min={vals[0]:.2f}  median={median:.2f}  max={vals[-1]:.2f}  (n={n})")


if __name__ == "__main__":
    main()