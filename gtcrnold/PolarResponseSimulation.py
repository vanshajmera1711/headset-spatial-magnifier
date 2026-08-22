import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- IMPORT YOUR MODEL ---
# Ensure this matches your folder structure
from models.gtcrn import GTCRN 

# --- CONFIGURATION ---
AUDIO_FILE = "../LibriSpeech/test-clean/260/123288/260-123288-0027.flac"       # A clean, mono speech file (3-5 seconds is best)
MODEL_FILE = "./logs/gtcrn_ch4_sp2.ckpt"
N_CHANNELS = 4
ANGLES = np.arange(0, 360, 2)  # Test every 5 degrees
FS = 16000
N_FFT = 512
BLOCK_SHIFT = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MICROPHONE GEOMETRY ---
# Diagonal distance = 4 cm -> Radius = 2 cm (0.02 m)
MIC_RADIUS = 0.02

def load_model():
    print(f"Loading GTCRN on {DEVICE}...")
    try:
        model = GTCRN(n_channels=N_CHANNELS).to(DEVICE)
        ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
        
        # Handle 'state_dict' wrapper if present
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Remove 'model.' prefix if saved by Lightning
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def simulate_at_angle(audio, angle_deg, distance=1.5):
    """
    Simulates a source at a specific angle in a 6x6m room.
    Mic Array: 4-mic square, 4cm diagonal.
    """
    
    # 1. Setup Room (6m x 6m x 3m) - Low reverb (absorption=0.2)
    room = pra.ShoeBox([6, 6, 3], fs=FS, max_order=0, absorption=0.2)
    
    # 2. Setup Mic Array (Center of Room)
    cx, cy, cz = 3.0, 3.0, 1.5
    d = MIC_RADIUS

    # Square Array (Counter-Clockwise)
    # Mic 1: Top-Right
    # Mic 2: Top-Left
    # Mic 3: Bottom-Left
    # Mic 4: Bottom-Right
    mics = np.array([
        [cx+d, cy, cz], 
        [cx, cy+d, cz], 
        [cx-d, cy, cz],
        [cx, cy-d, cz]
    ]).T
    
    room.add_microphone_array(mics)
    
    # 3. Add Source
    rad = np.radians(angle_deg)
    src_x = cx + distance * np.cos(rad)
    src_y = cy + distance * np.sin(rad)
    
    room.add_source([src_x, src_y, cz], signal=audio)
    
    # 4. Simulate
    room.simulate()
    
    # Return [Samples, Mics]
    sim_out = room.mic_array.signals.T
    
    # Normalize (Crucial for consistent testing)
    sim_out = sim_out / (np.max(np.abs(sim_out)) + 1e-8) * 0.9
    return sim_out

def apply_neural_steer(stft_list, distances, theta_1_list, theta_t_deg, sr=16000, n_fft=512):
    """
    Applies independent steering logic for a 4-mic system where each pair 
    may have a different reference (trained) angle.
    
    Args:
        stft_list: List of 4 complex arrays, each (K, T). [cite: 154]
        distances: List of 3 distances [d_01, d_02, d_03] in meters. 
        theta_1_list: List of 3 trained reference angles [theta_1_01, theta_1_02, theta_1_03] in degrees.
        theta_2_deg: The single target angle (theta_2) we want to steer the whole array toward. 
        sr: 16000 (Sampling Rate). [cite: 165]
        n_fft: 320 (FFT size). [cite: 157]
    """
    c = 343.0  # Speed of sound in m/s [cite: 93]
    theta_t_deg = theta_t_deg % 360
    
    # f: Frequency bins (K,) where K = 161 for n_fft=320 [cite: 157]
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)

    if (theta_t_deg >= 315) or (theta_t_deg < 45):
        pass
    elif (135 > theta_t_deg >= 45):
        stft_list = [stft_list[1], stft_list[2], stft_list[3], stft_list[0]]
        theta_t_deg = theta_t_deg - 90
    elif (225 > theta_t_deg >= 135):
        stft_list = [stft_list[2], stft_list[3], stft_list[0], stft_list[1]]
        theta_t_deg = theta_t_deg - 180
    elif (315 > theta_t_deg >= 225):
        stft_list = [stft_list[3], stft_list[0], stft_list[1], stft_list[2]]
        theta_t_deg = theta_t_deg - 270


    theta_t = np.deg2rad(theta_t_deg) # Common target angle in radians
    
    # Y_m=1: Reference microphone (Channel 0) remains unchanged [cite: 100, 103]
    steered_stfts = [stft_list[0]]
    
    # Process each pair (0,1), (0,2), (0,3)
    for i in range(3):
        d = distances[i] # Distance for this specific pair 
        theta_1 = np.deg2rad(theta_1_list[i]) # Unique reference angle for this pair 
        Y_m = stft_list[i+1] # The STFT of the current microphone (m=2, 3, or 4) [cite: 108]
        
        # delta_psi: Phase shift to align theta_2 with this pair's theta_1 [cite: 105]
        # Formula: delta_psi = 2 * pi * f * (d/c) * (cos(theta_1) - cos(theta_2)) 
        delta_psi = 2 * np.pi * f * (d / c) * (np.cos(theta_1) - np.cos(theta_1+theta_t))
        
        # a_k: Complex steering vector (K,)
        a_k = np.exp(1j * delta_psi)
        
        # Y_tilde: Applying the phase shift to the spectrum 
        # Shape: (K, T) via broadcasting
        Y_tilde = Y_m * a_k[:, np.newaxis]
        
        steered_stfts.append(Y_tilde)
        
    return steered_stfts

def process_audio(model, audio_multi, target):
    """Runs the model inference on the simulated audio."""
    
    # Prepare Input [Batch=1, Channels, Freq, Time]
    inp = torch.tensor(audio_multi, dtype=torch.float32).to(DEVICE) #[T*F, C]
    # print("\n")
    # print(inp.shape)

    # STFT
    window = torch.hann_window(N_FFT).to(DEVICE)
    stft = torch.stft(
        inp.T, n_fft=N_FFT, hop_length=BLOCK_SHIFT, 
        win_length=N_FFT, window=window,
        center=True, return_complex=True
    ) #[C, F+1, T]
    # print("\n")
    
    dist = [0.0283, 0.0400, 0.0283]
    theta_1 = [225.0, 180.0, 135.0]
    # stft = torch.roll(stft, shifts=3, dims=0)
    steered_stft = apply_neural_steer(stft, dist, theta_1, target)
    steered_stft = torch.from_numpy(np.stack(steered_stft, axis=0))

    

    # Stack for Model
    stft = steered_stft.unsqueeze(0) 
    # print(stft.shape)
    model_in = torch.cat((stft.real, stft.imag), dim=1)
    
    # Inference
    with torch.no_grad():
        model_in = model_in.to(torch.float32)
        mask_out = model(model_in)
        
    # Decode Mask (cIRM)
    # Using Channel 0 as Target
    mask_real = mask_out[:, 0, ...]
    mask_imag = mask_out[:, 1, ...]
    
    K, C = 1.0, 1.0 # Hyperparams (Must match training)
    comp_mask = mask_real + 1j * mask_imag
    
    # Inverse cIRM formula
    term = (K - K * comp_mask) / (K + K * comp_mask + 1e-8)
    mask = (-1 / C) * torch.log(term + 0j)
    
    # Apply Mask to Reference Mic (Mic 0)
    ref_mic_stft = stft[:, 0, ...] 
    est_stft = ref_mic_stft * mask
    
    # ISTFT
    est_audio = torch.istft(
        est_stft, n_fft=N_FFT, hop_length=BLOCK_SHIFT, 
        win_length=N_FFT, window=window,
        center=True
    )
    
    return est_audio.squeeze().cpu().numpy()

def calculate_energy(signal):
    """Calculates RMS Energy in dB."""
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms + 1e-8)

if __name__ == "__main__":
    # 1. Load Source Audio
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: '{AUDIO_FILE}' not found. Please provide a mono wav file.")
        exit()
        
    clean_audio, _ = sf.read(AUDIO_FILE)
    if len(clean_audio.shape) > 1: clean_audio = clean_audio[:, 0]

    target = 300
    
    # 2. Load Model
    model = load_model()
    
    results = []
    print(f"\n--- Starting Polar Evaluation (Mic Radius: {MIC_RADIUS*100:.1f} cm) ---")
    
    for angle in tqdm(ANGLES):
        # A. Simulate
        sim_audio = simulate_at_angle(clean_audio, angle)
        
        # B. Process
        enhanced_audio = process_audio(model, sim_audio, target)
        
        # C. Measure
        energy_db = calculate_energy(enhanced_audio)
        results.append(energy_db)

    # --- PLOTTING ---
    print("\nGenerating Polar Plot...")
    
    # Normalize results: Max volume = 0 dB (Relative Attenuation)
    results = np.array(results)
    results_norm = results - np.max(results)
    
    # Wrap data to close the circle (0 == 360)
    angles_rad = np.radians(np.append(ANGLES, ANGLES[0]))
    values = np.append(results_norm, results_norm[0])
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Plot Data
    ax.plot(angles_rad, values, linewidth=2, label='Model Output', color='blue')
    ax.fill(angles_rad, values, alpha=0.3, color='blue')
    
    # Reference Line (0 dB)
    ax.plot(angles_rad, np.zeros_like(values), linestyle='--', color='gray', alpha=0.5, label='Reference (0 dB)')
    
    # Styling
    ax.set_theta_zero_location('E') # 0 degrees = East (Right)
    ax.set_theta_direction(1)       # Counter-Clockwise
    ax.set_rlabel_position(45)
    ax.set_title("Target Speaker Extraction Pattern (dB)", va='bottom', fontweight='bold')
    
    # Limit radial axis (e.g., down to -40dB) for clarity
    plt.ylim(bottom=-40, top=2)

    plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.1))
    # plt.tight_layout()    # remove this — it changes margins per-plot
    plt.savefig(
        f"polar_response_{target}.png",
        dpi=150,
        bbox_inches=None,    # fixed canvas, NOT 'tight'
        pad_inches=0.3
    )
    # plt.show()
    
    print("Success! Graph saved")