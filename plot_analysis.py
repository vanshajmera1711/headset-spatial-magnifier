import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH CONFIGURATIONS & DYNAMIC FOLDER
# ==========================================
METRICS_JSON_PATH = r"C:\Users\Admin\room_simulation\data\mvdr_metrics.json"
BASE_DATA_FOLDER = r"C:\Users\Admin\room_simulation\data"

if not os.path.exists(METRICS_JSON_PATH):
    print(f"Error: Missing metrics file at {METRICS_JSON_PATH}.")
    print("Please execute your 'run_mvdr.py' script first to generate the metric JSON data.")
    exit()

# Create a unique timestamped folder name (Format: analysis_YYYYMMDD_HHMMSS)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_FOLDER = os.path.join(BASE_DATA_FOLDER, f"analysis_{timestamp}")

# Ensure the new directory exists on disk
os.makedirs(RUN_OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PLOT_PATH = os.path.join(RUN_OUTPUT_FOLDER, "si_snr_analysis.png")

# ==========================================
# 2. PARSE METRICS LOG DATA
# ==========================================
with open(METRICS_JSON_PATH, "r") as f:
    metrics_data = json.load(f)

per_file_logs = metrics_data.get("per_file_metrics", {})

if not per_file_logs:
    print("Error: The 'per_file_metrics' log inside your JSON file is completely empty.")
    exit()

input_sisnr = []
output_sisnr = []

for filename, scores in per_file_logs.items():
    input_sisnr.append(scores["sisnr_input"])
    output_sisnr.append(scores["sisnr_output"])

# ==========================================
# 3. GENERATE MATPLOTLIB AXES
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

# Scatter plot of individual audio scenes
ax.scatter(input_sisnr, output_sisnr, color='#1f77b4', alpha=0.8, edgecolors='black', s=100, label='Simulated Audio Scenes')

# Determine limits for the diagonal identity threshold line
all_coordinates = input_sisnr + output_sisnr
axis_min = min(all_coordinates) - 2
axis_max = max(all_coordinates) + 2

# Draw the 0 dB identity baseline (y = x)
ax.plot([axis_min, axis_max], [axis_min, axis_max], color='#d62728', linestyle='--', linewidth=2, label='Zero Gain Line (Output = Input)')

# Axis decoration and framing
ax.set_xlabel('Input SI-SNR (dB)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Output SI-SNR (dB)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Multi-Channel Souden MVDR Evaluation\nOutput SI-SNR vs. Input SI-SNR', fontsize=14, fontweight='bold', pad=15)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=11, frameon=True)

ax.set_xlim(min(input_sisnr) - 1.5, max(input_sisnr) + 1.5)
ax.set_ylim(min(output_sisnr) - 1.5, max(output_sisnr) + 1.5)

# ==========================================
# 4. EXPORT FILE ARCHIVE
# ==========================================
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH)
plt.close()  # Clean up memory allocation

print(f"\n[SUCCESS] Diagnostic plot archived without overwriting!")
print(f" Saved to: {OUTPUT_PLOT_PATH}")