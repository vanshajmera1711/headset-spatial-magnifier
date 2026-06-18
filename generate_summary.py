import json

metrics_json_path = r"C:\projects\headset-spatial-magnifier\data\mvdr_metrics_blind.json"

with open(metrics_json_path, "r") as f:
    data = json.load(f)

print("\n" + "="*40)
print("       BLIND REAL-TIME MVDR SUMMARY")
print("="*40)
print(f"Scenes processed        : {data['scenes_processed']}")
print(f"Mean SI-SNR input       : {data['mean_sisnr_input_db']:.2f} dB")
print(f"Mean SI-SNR output      : {data['mean_sisnr_output_db']:.2f} dB")
print(f"Mean SI-SNR improvement : {data['mean_sisnr_improvement_db']:.2f} dB")
print("="*40)
