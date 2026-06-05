import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
import random
import json
import glob
import librosa

def create_headset_magnifier_array(centroid_x, centroid_y, height):
    """
    Generates a headset array layout: 
    2 real mics on Left Ear, 2 real mics on Right Ear, and 2 Virtual expansion points.
    """
    head_width = 0.16    # 16 cm between left and right ears
    ear_mic_gap = 0.015  # 1.5 cm between front and back mic on an ear
    virtual_gap = 0.05   # 5 cm virtual expansion outward
    
    half_head = head_width / 2.0
    half_ear  = ear_mic_gap / 2.0
    
    v_left_outer = [centroid_x - half_head - virtual_gap, centroid_y, height]
    real_left_front = [centroid_x - half_head, centroid_y + half_ear, height]
    real_left_back  = [centroid_x - half_head, centroid_y - half_ear, height]
    
    real_right_back  = [centroid_x + half_head, centroid_y - half_ear, height]
    real_right_front = [centroid_x + half_head, centroid_y + half_ear, height]
    v_right_outer = [centroid_x + half_head + virtual_gap, centroid_y, height]
    
    return np.array([v_left_outer, real_left_front, real_left_back, real_right_back, real_right_front, v_right_outer]).T

def parse_dns_dataset(dns_speech_dir, dns_noise_dir):
    """
    Crawls directories to find all valid audio paths.
    """
    speech_files = glob.glob(os.path.join(dns_speech_dir, "**/*.wav"), recursive=True)
    noise_files = glob.glob(os.path.join(dns_noise_dir, "**/*.wav"), recursive=True)
    return speech_files, noise_files

def load_and_rescale_audio(file_path, target_len_samples, target_fs=16000):
    """
    Loads an audio file, resamples if needed, and forces it to a target sample length.
    """
    data, fs = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if fs != target_fs:
        data = librosa.resample(data, orig_sr=fs, target_sr=target_fs)
    if len(data) >= target_len_samples:
        return data[:target_len_samples]
    else:
        return np.pad(data, (0, target_len_samples - len(data)), 'constant')

def compute_active_rms(signal):
    eps = 1e-10
    return np.sqrt(np.mean(signal**2) + eps)

def simulate_single_scene(sample_idx, speech_files, noise_files, output_dir):
    fs = 16000
    duration_samples = 4 * fs  # Force to exactly 4 seconds
    
    # 1. Sample Randomized Room Configurations
    room_L = random.uniform(3.5, 8.0)
    room_W = random.uniform(3.5, 6.0)
    room_H = random.uniform(2.4, 3.5)
    room_dim = np.array([room_L, room_W, room_H])
    
    rt60 = random.uniform(0.2, 0.7)
    volume = np.prod(room_dim)
    surface_area = 2 * (room_dim[0]*room_dim[1] + room_dim[1]*room_dim[2] + room_dim[0]*room_dim[2])
    absorption = 0.161 * volume / (surface_area * rt60)
    absorption = min(max(absorption, 0.05), 0.5)
    
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption), max_order=10)
    
    # 2. Place Headset Centroid Randomly
    c_x = random.uniform(room_L * 0.3, room_L * 0.7)
    c_y = random.uniform(room_W * 0.3, room_W * 0.7)
    c_z = random.uniform(1.0, 1.5)
    
    mic_matrix = create_headset_magnifier_array(c_x, c_y, c_z)
    room.add_microphone_array(pra.MicrophoneArray(mic_matrix, room.fs))
    
    # 3. Choose Audio Assets Disjointly
    num_interferers = random.randint(1, 3)
    chosen_speech = random.sample(speech_files, 1 + num_interferers)
    target_path = chosen_speech[0]
    interf_paths = chosen_speech[1:]
    noise_path = random.choice(noise_files)

    s_target = load_and_rescale_audio(target_path, duration_samples, fs)
    s_interfs = [load_and_rescale_audio(p, duration_samples, fs) for p in interf_paths]
    s_noise = load_and_rescale_audio(noise_path, duration_samples, fs)
    
    # 4. Generate Random Coordinates for Sources
    def get_valid_coords():
        while True:
            coords = np.array([random.uniform(0.5, room_L-0.5), 
                               random.uniform(0.5, room_W-0.5), 
                               random.uniform(1.0, 1.8)])
            if np.linalg.norm(coords - np.array([c_x, c_y, c_z])) > 0.6:
                return coords

    target_coords = get_valid_coords()
    interf_coords_list = []
    for s_interf in s_interfs:
        ic = get_valid_coords()
        interf_coords_list.append(ic)
        room.add_source(ic, signal=s_interf)
    noise_coords = get_valid_coords()

    room.add_source(target_coords, signal=s_target)
    for i, s_interf in enumerate(s_interfs):
        room.add_source(interf_coords_list[i], signal=s_interf)
    room.add_source(noise_coords, signal=s_noise)
    
    # 5. Run Acoustic Math
    room.compute_rir()
    room.simulate()
    
    mixture = room.mic_array.signals
    num_mics = mic_matrix.shape[1]
    noise_source_idx = 1 + num_interferers
    
    # 6. Isolate Ground-Truth Components
    target_spatial = np.zeros_like(mixture)
    interf_spatial = np.zeros_like(mixture)
    noise_spatial = np.zeros_like(mixture)
    
    for m in range(num_mics):
        # --- Safe Target Convolution ---
        conv_t = np.convolve(s_target, room.rir[m][0])
        if len(conv_t) >= mixture.shape[1]:
            target_spatial[m, :] = conv_t[:mixture.shape[1]]
        else:
            target_spatial[m, :len(conv_t)] = conv_t

        # --- Safe Interferer Convolution (sum all interferers) ---
        for i, s_interf in enumerate(s_interfs):
            conv_i = np.convolve(s_interf, room.rir[m][1 + i])
            if len(conv_i) >= mixture.shape[1]:
                interf_spatial[m, :] += conv_i[:mixture.shape[1]]
            else:
                interf_spatial[m, :len(conv_i)] += conv_i

        # --- Safe Noise Convolution ---
        conv_n = np.convolve(s_noise, room.rir[m][noise_source_idx])
        if len(conv_n) >= mixture.shape[1]:
            noise_spatial[m, :] = conv_n[:mixture.shape[1]]
        else:
            noise_spatial[m, :len(conv_n)] = conv_n
        
    # 7. Balance Content (SIR/SNR levels)
    target_sir_db = random.uniform(0.0, 6.0)
    target_snr_db = random.uniform(5.0, 15.0)
    
    ref_mic = 1
    rms_t = compute_active_rms(target_spatial[ref_mic, :])
    rms_i = compute_active_rms(interf_spatial[ref_mic, :])
    rms_n = compute_active_rms(noise_spatial[ref_mic, :])
    
    scale_i = (rms_t / (10 ** (target_sir_db / 20))) / rms_i
    scale_n = (rms_t / (10 ** (target_snr_db / 20))) / rms_n
    
    interf_spatial *= scale_i
    noise_spatial *= scale_n
    
    final_mixture = target_spatial + interf_spatial + noise_spatial
    
    # 8. Global Normalization
    max_peak = np.max(np.abs(final_mixture))
    if max_peak > 0:
        scale_factor = 0.9 / max_peak
        final_mixture *= scale_factor
        target_spatial *= scale_factor
        
    # 9. Write outputs
    os.makedirs(output_dir, exist_ok=True)
    sf.write(f"{output_dir}/sample_{sample_idx}_mix.wav", final_mixture.T, fs)
    sf.write(f"{output_dir}/sample_{sample_idx}_target_gt.wav", target_spatial.T, fs)
    
    meta_data = {
        "room_size_lwh": room_dim.tolist(),
        "rt60_seconds": rt60,
        "sir_db": target_sir_db,
        "snr_db": target_snr_db,
        "headset_centroid": [c_x, c_y, c_z],
        "target_position": target_coords.tolist(),
        "num_interferers": num_interferers,
        "interferer_positions": [ic.tolist() for ic in interf_coords_list]
    }
    with open(f"{output_dir}/sample_{sample_idx}_meta.json", "w") as f:
        json.dump(meta_data, f, indent=4)