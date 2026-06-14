import json
import os
import numpy as np

METRICS_PATH = r"C:\Users\Admin\room_simulation\data\dsbf_metrics.json"

if not os.path.exists(METRICS_PATH):
    print(f"Error: Could not find {METRICS_PATH}")
else:
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    per_file = data["per_file_metrics"]
    improvements = [v["sisnr_improvement"] for v in per_file.values()]
    
    # 1. Distribution Analysis
    improvements = np.array(improvements)
    print("--- SI-SNR Improvement Distribution ---")
    print(f"Max Gain     : {np.max(improvements):.2f} dB")
    print(f"Min Gain     : {np.min(improvements):.2f} dB")
    print(f"Median Gain  : {np.median(improvements):.2f} dB")
    print(f"Std Deviation: {np.std(improvements):.2f} dB")
    
    # 2. Performance Buckets
    degraded = improvements[improvements < 0]
    small_gain = improvements[(improvements >= 0) & (improvements < 2)]
    high_gain = improvements[improvements >= 2]
    
    print(f"\n--- Scene Performance Breakdown ---")
    print(f"Degraded Quality (< 0 dB) : {len(degraded)} files ({(len(degraded)/len(improvements))*100:.1f}%)")
    print(f"Modest Gain (0-2 dB)      : {len(small_gain)} files ({(len(small_gain)/len(improvements))*100:.1f}%)")
    print(f"Significant Gain (> 2 dB) : {len(high_gain)} files ({(len(high_gain)/len(improvements))*100:.1f}%)")

    # 3. Identify Top/Bottom 3 Scenes for Manual Listening
    # Sorting by improvement value
    sorted_files = sorted(per_file.items(), key=lambda x: x[1]['sisnr_improvement'])
    
    print(f"\n--- Worst 3 Scenes (Check for high reverb/noise) ---")
    for name, metrics in sorted_files[:3]:
        print(f" {name}: {metrics['sisnr_improvement']} dB")
        
    print(f"\n--- Best 3 Scenes (Ideal spatial separation) ---")
    for name, metrics in sorted_files[-3:]:
        print(f" {name}: {metrics['sisnr_improvement']} dB")