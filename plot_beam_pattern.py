import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
INPUT_FOLDER = os.path.join(DATA_DIR, "generated_dataset")
SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 16000
FFT_LENGTH = 512

# 1. Load your ACTUAL live matrix dumps and sample metadata
try:
    R_noise_true = np.load(os.path.join(DATA_DIR, "true_R_noise.npy")) 
    steering_true = np.load(os.path.join(DATA_DIR, "true_steering.npy"))
    
    with open(os.path.join(INPUT_FOLDER, "sample_0_meta.json"), "r") as f:
        meta = json.load(f)
except FileNotFoundError:
    print("Error: Missing necessary files. Run your full simulation script first.")
    exit()

# Extract your exact uncentered geometry from metadata
centroid = np.array(meta["headset_centroid"])
target_pos = np.array(meta["target_position"])

def reconstruct_mic_positions(centroid):
    cx, cy, height = centroid
    return np.array([
        [cx - 0.08,  cy,        height + 0.03], # ch 0
        [cx - 0.08,  cy + 0.02, height       ], # ch 1
        [cx - 0.08,  cy - 0.02, height       ], # ch 2
        [cx + 0.08,  cy - 0.02, height       ], # ch 3
        [cx + 0.08,  cy + 0.02, height       ], # ch 4
        [cx + 0.08,  cy,        height + 0.03]  # ch 5
    ])

mic_positions = reconstruct_mic_positions(centroid)

TARGET_BINS = [16, 32, 96] # 500Hz, 1000Hz, 3000Hz
angles = np.linspace(0, 2 * np.pi, 360)
scan_radius = 2.0  # 2 meters out from the headset
n_channels = 6

fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})

for i, bin_idx in enumerate(TARGET_BINS):
    freq = bin_idx * (SAMPLE_RATE / FFT_LENGTH)
    
    R_bin = R_noise_true[:, :, bin_idx]
    d_bin = steering_true[:, bin_idx]
    
    # Invert and compute weights using the updated lower safety floor
    eps = np.maximum(1e-6 * np.mean(np.abs(R_bin)), 1e-8)
    R_inv = np.linalg.pinv(R_bin + eps * np.eye(n_channels))
    w_mvdr = (R_inv @ d_bin) / (np.conj(d_bin) @ R_inv @ d_bin + 1e-5)
    
    response = []
    for theta in angles:
        # --- TRUE UNCENTERED SPHERICAL SCANNING ---
        # Places the scanning speakers at the true coordinates relative to your headset centroid
        scan_pos = np.array([
            centroid[0] + scan_radius * np.cos(theta),
            centroid[1] + scan_radius * np.sin(theta),
            centroid[2]
        ])
        
        # Calculate true distances from the scanning positions to each physical mic capsule
        scan_dist = np.linalg.norm(mic_positions - scan_pos, axis=1)
        
        # Calculate time-of-flight delays and lock them to reference channel 1
        scan_delays = scan_dist / SPEED_OF_SOUND
        scan_rel_delays = scan_delays - scan_delays[1]
        
        # True Near-Field Amplitude scaling term (Inverse-Square Law drop)
        scan_rel_amp = scan_dist[1] / scan_dist
        
        # Build the actual scanning template vector
        d_scan = scan_rel_amp * np.exp(-1j * 2 * np.pi * freq * scan_rel_delays)
        
        # Compute the true spatial response response scalar
        response.append(np.abs(np.conj(w_mvdr) @ d_scan))
        
    response_db = 20 * np.log10(np.array(response) + 1e-5)
    response_db -= np.max(response_db)
    
    ax = axs[i]
    ax.plot(angles, response_db, color='crimson', linewidth=2)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(-35, 5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"TRUE Data: {int(freq)} Hz", va='bottom', fontsize=12)

plt.suptitle("Actual Asymmetric Live MVDR Beams (Calculated from Data Matrices)", fontsize=14, weight='bold', y=1.05)
plt.tight_layout()
plt.show()