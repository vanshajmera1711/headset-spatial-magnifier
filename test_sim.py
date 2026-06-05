import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os

def create_headset_magnifier_array(centroid_x, centroid_y, height):
    """
    Generates a headset array layout: 
    2 real mics on Left Ear, 2 real mics on Right Ear, and 2 Virtual expansion points.
    Structure: [V_Left, R_Left_Front, R_Left_Back, R_Right_Back, R_Right_Front, V_Right]
    """
    # Physical head and headset spacing constants (in meters)
    head_width = 0.16    # 16 cm typical distance between left and right ears
    ear_mic_gap = 0.015  # 1.5 cm distance between the front and back mic on an ear
    virtual_gap = 0.05   # 5 cm virtual expansion outward to extend array aperture
    
    half_head = head_width / 2.0
    half_ear  = ear_mic_gap / 2.0
    
    # --- LEFT SIDE COORDINATES ---
    v_left_outer = [centroid_x - half_head - virtual_gap, centroid_y, height]
    real_left_front = [centroid_x - half_head, centroid_y + half_ear, height]
    real_left_back  = [centroid_x - half_head, centroid_y - half_ear, height]
    
    # --- RIGHT SIDE COORDINATES ---
    real_right_back  = [centroid_x + half_head, centroid_y - half_ear, height]
    real_right_front = [centroid_x + half_head, centroid_y + half_ear, height]
    v_right_outer = [centroid_x + half_head + virtual_gap, centroid_y, height]
    
    # Compile into the pyroomacoustics layout matrix: Shape (3, 6) -> (X, Y, Z) rows
    mic_matrix = np.array([
        v_left_outer,      # Channel 0: Virtual Target Left
        real_left_front,   # Channel 1: Real Mic Left Front
        real_left_back,    # Channel 2: Real Mic Left Back
        real_right_back,   # Channel 3: Real Mic Right Back
        real_right_front,  # Channel 4: Real Mic Right Front
        v_right_outer      # Channel 5: Virtual Target Right
    ]).T
    
    return mic_matrix

def run_headset_simulation():
    print("--- Starting Headset Spatial-Magnifier Simulation ---")
    
    # 1. Define Room Geometry (Length=6.5m, Width=5.5m, Height=2.8m)
    room_dim = np.array([6.5, 5.5, 2.8])
    rt60 = 0.35  # Target reverberation time in seconds
    fs = 16000   # 16kHz audio standard for speech models
    
    # Invert Sabine's formula to get the necessary wall absorption coefficient
    volume = np.prod(room_dim)
    surface_area = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[0]*room_dim[2])
    absorption = 0.161 * volume / (surface_area * rt60)
    
    # Instantiate Shoebox Room
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=12)
    
    # 2. Attach Headset Microphone Array (Centered at standing height 1.4m)
    c_x, c_y, c_z = 3.25, 2.75, 1.4
    mic_matrix = create_headset_magnifier_array(c_x, c_y, c_z)
    room.add_microphone_array(pra.MicrophoneArray(mic_matrix, room.fs))
    
    # 3. Create Synth Mock Signals (White noise arrays representing clean inputs)
    duration_samples = 4 * fs  # 4 seconds long
    np.random.seed(101)        # Hardcoded seed for spatial phase reproducibility
    
    target_signal = np.random.randn(duration_samples) * 0.1
    interferer_signal = np.random.randn(duration_samples) * 0.1
    
    # 4. Place Acoustic Sources inside the Space (Ensuring wall & array clearance)
    target_coords = np.array([2.1, 4.2, 1.5])      # Spatially separated to the front-left
    interferer_coords = np.array([4.8, 1.9, 1.3])  # Spatially separated to the back-right
    
    room.add_source(target_coords, signal=target_signal)
    room.add_source(interferer_coords, signal=interferer_signal)
    
    # 5. Compute RIR Vectors via Ray-Tracing Engine
    print("Simulating wall acoustic reflections and generating RIR arrays...")
    room.compute_rir()
    room.simulate()
    
    # 6. Extract Raw Output Mixture
    mixture = room.mic_array.signals
    num_mics = mic_matrix.shape[1]
    
    # 7. Manually Convolve Target RIR to Isolate Clean Reverberant Target Ground Truth
    target_gt = np.zeros_like(mixture)
    for m in range(num_mics):
        # Target was added first, so its source index is 0
        target_rir = room.rir[m][0]
        convolved = np.convolve(target_signal, target_rir)
        
        # Safe dynamic trimming/padding to perfectly match the mixture length
        if len(convolved) >= mixture.shape[1]:
            target_gt[m, :] = convolved[:mixture.shape[1]]
        else:
            target_gt[m, :len(convolved)] = convolved

    # 8. Multi-Channel Amplitude Normalization (Protects against clipping)
    max_peak = np.max(np.abs(mixture))
    if max_peak > 0:
        scale_factor = 0.9 / max_peak
        mixture *= scale_factor
        target_gt *= scale_factor
        
    # 9. Serialize Output Signals to Audio Tensors
    os.makedirs("headset_test_output", exist_ok=True)
    sf.write("headset_test_output/headset_mixture.wav", mixture.T, fs)
    sf.write("headset_test_output/headset_target_ground_truth.wav", target_gt.T, fs)
    
    print("\n[SUCCESS] Headset room simulation finalized!")
    print(f"Matrix Output Dimensions: {mixture.shape} ({mixture.shape[0]} Channels, {mixture.shape[1]} Time Frames)")
    print("Saved audio files successfully in directory: './headset_test_output'")

if __name__ == "__main__":
    run_headset_simulation()