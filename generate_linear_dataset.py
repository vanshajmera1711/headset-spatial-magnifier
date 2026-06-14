import os
import json
import torch
import soundfile as sf

def soundfile_save_patch(filepath, src, sample_rate, channels_first=True, bits_per_sample=16, format=None, encoding=None):
    if channels_first and src.ndim > 1:
        data = src.t().numpy()
    else:
        data = src.numpy()
    sf.write(filepath, data, sample_rate, subtype=f'PCM_{bits_per_sample}')

import torchaudio
torchaudio.save = soundfile_save_patch

# ==========================================================
# 2. DEFINE SCENARIOS & SPATIAL STEERING VECTORS
# ==========================================================
OUTPUT_DIR = r"C:\Users\Admin\room_simulation\data\simulated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
DURATION_SECONDS = 3
duration_samples = SAMPLE_RATE * DURATION_SECONDS

# Standard target speech steering vector (arrives straight-on, center mouth)
steering_speech = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

# Define 4 distinct physical noise angle scenarios
SCENARIOS = {
    "Easy_90_Degrees": {
        "steering": torch.tensor([[1.0], [-0.9], [0.8], [-0.7], [0.6], [-0.5]]),
        "desc": "Noise source far to the left. Wide angular separation."
    },
    "Medium_45_Degrees": {
        "steering": torch.tensor([[1.0], [0.4], [0.9], [0.3], [0.8], [0.2]]),
        "desc": "Noise source at a diagonal. Moderate separation."
    },
    "Hard_10_Degrees": {
        "steering": torch.tensor([[1.0], [0.95], [0.99], [0.94], [0.98], [0.93]]),
        "desc": "Noise source right next to target speech. Narrow separation."
    },
    "Brutal_Overlapped": {
        "steering": torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]),
        "desc": "Noise source directly behind speaker. Zero spatial separation."
    }
}

# 10 volume steps per scenario
input_snr_gradient = [-10, -5, -2, 0, 2, 5, 8, 12, 15, 20]

print("Generating expanded dataset (4 Scenarios x 10 Volume Steps = 40 Scenes)...")

# ==========================================================
# 3. GENERATION LOOP
# ==========================================================
file_idx = 0
for scenario_name, config in SCENARIOS.items():
    print(f"\nProcessing Scenario: {scenario_name} ({config['desc']})")
    
    # Ensure source audio signals are distinct for every single scenario
    torch.manual_seed(file_idx + 100)
    mono_speech = torch.randn(1, duration_samples) * 0.1
    
    torch.manual_seed(file_idx + 500)
    mono_noise = torch.randn(1, duration_samples) * 0.1
    
    # Project to 6 channels using the scenario's spatial signature
    base_target_spatial = steering_speech @ mono_speech
    base_noise_spatial = config["steering"] @ mono_noise
    
    for snr_db in input_snr_gradient:
        alpha = 10 ** (-snr_db / 20.0)
        scaled_noise_field = base_noise_spatial * alpha
        
        simulated_mixture = base_target_spatial + scaled_noise_field
        
        # Consistent naming scheme for your evaluation script
        mix_path = os.path.join(OUTPUT_DIR, f"sample_{file_idx}_mix.wav")
        target_path = os.path.join(OUTPUT_DIR, f"sample_{file_idx}_target_gt.wav")
        meta_path = os.path.join(OUTPUT_DIR, f"sample_{file_idx}_meta.json")
        
        torchaudio.save(mix_path, simulated_mixture, SAMPLE_RATE)
        torchaudio.save(target_path, base_target_spatial, SAMPLE_RATE)
        
        metadata = {
            "sample_index": file_idx,
            "scenario": scenario_name,
            "theoretical_input_snr_db": snr_db
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        file_idx += 1

print(f"\n[SUCCESS] Generated {file_idx} baseline test files!")