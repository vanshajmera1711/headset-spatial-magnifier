import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
import random
import json
import glob
import librosa
from tqdm import tqdm


def create_safe_headphone_physical_array(centroid_x, centroid_y, height):
    """
    Generates a 6-channel physical array safe for real-world headphone shells.
    Dimensions: 4.0 cm horizontal base (front-back) x 3.0 cm vertical height (upward apex).
    """
    head_width = 0.16        # 16.0 cm standard human head width
    cup_base_width = 0.04    # 4.0 cm horizontal distance between front and back mics
    cup_apex_height = 0.03   # 3.0 cm vertical height up to the top apex mic

    half_head = head_width / 2.0
    half_base = cup_base_width / 2.0

    # --- LEFT EAR-CUP SAFE TRIANGLE (X = -8.0 cm) ---
    real_left_front = [centroid_x - half_head, centroid_y + half_base, height]
    real_left_back  = [centroid_x - half_head, centroid_y - half_base, height]
    real_left_top   = [centroid_x - half_head, centroid_y,             height + cup_apex_height]
    
    # --- RIGHT EAR-CUP SAFE TRIANGLE (X = +8.0 cm) ---
    real_right_back = [centroid_x + half_head, centroid_y - half_base, height]
    real_right_front= [centroid_x + half_head, centroid_y + half_base, height]
    real_right_top  = [centroid_x + half_head, centroid_y,             height + cup_apex_height]

    return np.array([
        real_left_top,    # Channel 0
        real_left_front,  # Channel 1
        real_left_back,   # Channel 2
        real_right_back,  # Channel 3
        real_right_front, # Channel 4
        real_right_top    # Channel 5
    ]).T


def parse_dns_dataset(dns_speech_dir, dns_noise_dir):
    """
    Crawls directories to find all valid audio paths.
    """
    speech_files = glob.glob(os.path.join(dns_speech_dir, "**/*.wav"), recursive=True)
    noise_files  = glob.glob(os.path.join(dns_noise_dir,  "**/*.wav"), recursive=True)
    return speech_files, noise_files


def load_and_rescale_audio(file_path, target_len_samples, target_fs=16000):
    """
    Loads an audio file, resamples if needed, and forces it to a target sample length.
    """
    data, fs = sf.read(file_path)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample from DNS native 48 kHz → 16 kHz
    if fs != target_fs:
        data = librosa.resample(data, orig_sr=fs, target_sr=target_fs)

    if len(data) >= target_len_samples:
        return data[:target_len_samples]
    else:
        return np.pad(data, (0, target_len_samples - len(data)), 'constant')


def compute_active_rms(signal):
    eps = 1e-10
    return np.sqrt(np.mean(signal**2) + eps)


def simulate_single_scene(sample_idx, speech_files, noise_files, output_dir, add_reverb=True):
    fs = 16000
    duration_samples = 4 * fs  # Force to exactly 4 seconds (64,000 samples)

    # =========================================================================
    # 1. RANDOMIZED ROOM CONFIGURATIONS MATCHING YOUR BTP TABLE I
    # =========================================================================
    room_W = random.uniform(2.5, 5.0)  # Width range from paper Table I
    room_L = random.uniform(3.0, 9.0)  # Length range from paper Table I
    room_H = random.uniform(2.2, 3.5)  # Height range from paper Table I
    room_dim = np.array([room_L, room_W, room_H])

    # =========================================================================
    # 2. ENERGIZE ACOUSTIC ENVIRONMENT INTERFACE (REVERBERATION SWITCH)
    # =========================================================================
    if add_reverb:
        # Randomize RT60 (Reverberation Time) between 0.2 to 0.6 seconds (Standard room/office)
        rt60_target = random.uniform(0.2, 0.6)
        try:
            # Dynamically invert Sabine's equation to select realistic absorption materials
            e_material = pra.Material.from_rt60(rt60_target, room_dim, fs=fs)
            # Track up to 15 reflection generations to build a comprehensive late decay tail
            room = pra.ShoeBox(room_dim, fs=fs, materials=e_material, max_order=15)
        except Exception:
            # Safe boundary ceiling fallback parameters
            rt60_target = 0.25
            room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(0.25), max_order=10)
    else:
        rt60_target = 0.0
        # ANECHOIC OVERRIDE: max_order=0 turns off all reflections
        room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(1.0), max_order=0)

    # =========================================================================
    # 3. POSITION HEADSET CENTROID
    # =========================================================================
    c_x = random.uniform(room_L * 0.2, room_L * 0.8)
    c_y = random.uniform(room_W * 0.2, room_W * 0.8)
    c_z = random.uniform(1.2, 1.5)  # Safe sitting ear level height range

    # Generate the 6-Channel Real Triangular Array (Safe Headphone Spacing: 4.0cm x 3.0cm)
    mic_matrix = create_safe_headphone_physical_array(c_x, c_y, c_z)
    room.add_microphone_array(pra.MicrophoneArray(mic_matrix, room.fs))

    # Choose Audio Assets Disjointly
    num_interferers = random.randint(1, 3)
    chosen_speech   = random.sample(speech_files, 1 + num_interferers)
    target_path     = chosen_speech[0]
    interf_paths    = chosen_speech[1:]
    noise_path      = random.choice(noise_files)

    s_target = load_and_rescale_audio(target_path, duration_samples, fs)
    s_interfs = [load_and_rescale_audio(p, duration_samples, fs) for p in interf_paths]
    s_noise   = load_and_rescale_audio(noise_path,  duration_samples, fs)

    # =========================================================================
    # 4. SET WEARER TARGET COORDINATES & RANDOMIZE DISTANT INTERFERERS
    # =========================================================================
    target_coords = np.array([
        c_x, 
        c_y + 0.12, 
        c_z - 0.10
    ])

    def get_distant_interferer_coords():
        while True:
            coords = np.array([
                random.uniform(0.2, room_L - 0.2), 
                random.uniform(0.2, room_W - 0.2),
                random.uniform(1.2, 1.8)           
            ])
            if np.linalg.norm(coords - np.array([c_x, c_y, c_z])) > 0.5:
                return coords

    interf_coords_list = [get_distant_interferer_coords() for _ in s_interfs]
    noise_coords       = get_distant_interferer_coords()

    # =========================================================================
    # 5. REGISTER SOURCES & RUN SIMULATION
    # =========================================================================
    room.add_source(target_coords, signal=s_target)
    for i, s_interf in enumerate(s_interfs):
        room.add_source(interf_coords_list[i], signal=s_interf)
    room.add_source(noise_coords, signal=s_noise)

    # Run the main simulation to get the multi-source combined mixture
    room.compute_rir()
    room.simulate()
    mixture = room.mic_array.signals
    num_mics = mic_matrix.shape[1]

    # =========================================================================
    # 6. VERSION-AGNOSTIC GROUND-TRUTH ISOLATION (DIRECT-PATH PATHWAY ONLY)
    # =========================================================================
    target_spatial = np.zeros_like(mixture)
    interf_spatial = np.zeros_like(mixture)
    noise_spatial  = np.zeros_like(mixture)

    target_src_idx = 0
    interf_src_idxs = list(range(1, 1 + num_interferers))
    noise_src_idx   = 1 + num_interferers

    for m in range(num_mics):
        # Isolate Target Speech Ground-Truth via Direct-Path RIR filter convolution
        conv_t = np.convolve(s_target, room.rir[m][target_src_idx])
        out_len_t = min(len(conv_t), mixture.shape[1])
        target_spatial[m, :out_len_t] = conv_t[:out_len_t]

        # Isolate Combined Interferers Ground-Truth
        for i, s_interf in enumerate(s_interfs):
            conv_i = np.convolve(s_interf, room.rir[m][interf_src_idxs[i]])
            out_len_i = min(len(conv_i), mixture.shape[1])
            interf_spatial[m, :out_len_i] += conv_i[:out_len_i]

        # Isolate Ambient Noise Ground-Truth
        conv_n = np.convolve(s_noise, room.rir[m][noise_src_idx])
        out_len_n = min(len(conv_n), mixture.shape[1])
        noise_spatial[m, :out_len_n] = conv_n[:out_len_n]

    # =========================================================================
    # 7. BALANCE CONTENT (SIR/SNR tracked at Channel 1 reference microphone)
    # =========================================================================
    target_sir_db  = random.uniform(0.0, 6.0)
    target_snr_db  = random.uniform(5.0, 15.0)
    ref_mic = 1

    rms_t = compute_active_rms(target_spatial[ref_mic, :])
    rms_i = compute_active_rms(interf_spatial[ref_mic, :])
    rms_n = compute_active_rms(noise_spatial[ref_mic, :])

    scale_i = (rms_t / (10 ** (target_sir_db / 20))) / rms_i
    scale_n = (rms_t / (10 ** (target_snr_db / 20))) / rms_n

    interf_spatial *= scale_i
    noise_spatial  *= scale_n

    final_mixture = target_spatial + interf_spatial + noise_spatial

    # 8. Global Normalization to safeguard against digital clipping
    max_peak = np.max(np.abs(final_mixture))
    if max_peak > 0:
        scale_factor   = 0.9 / max_peak
        final_mixture *= scale_factor
        target_spatial *= scale_factor

    # 9. Write outputs structures natively as 6-channel wave files
    os.makedirs(output_dir, exist_ok=True)
    sf.write(f"{output_dir}/sample_{sample_idx}_mix.wav",       final_mixture.T,  fs)
    sf.write(f"{output_dir}/sample_{sample_idx}_target_gt.wav", target_spatial.T, fs)

    meta_data = {
        "room_size_lwh":       room_dim.tolist(),
        "rt60_seconds":        round(rt60_target, 3), 
        "sir_db":              target_sir_db,
        "snr_db":              target_snr_db,
        "headset_centroid":    [c_x, c_y, c_z],
        "target_position":     target_coords.tolist(),
        "num_interferers":     num_interferers,
        "interferer_positions": [ic.tolist() for ic in interf_coords_list],
        "noise_position":      noise_coords.tolist(),
    }
    with open(f"{output_dir}/sample_{sample_idx}_meta.json", "w") as f:
        json.dump(meta_data, f, indent=4)

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # SEED REPLICABILITY LOCK MECHANISM
    # ─────────────────────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)

    # MASTER SWITCH: True for Echoic/Reverberant Run, False for Original Anechoic Baseline
    ADD_REVERB = True  
    
    DNS_SPEECH_DIR = "C:/projects/headset-spatial-magnifier/data/datasets_fullband/clean_fullband/mnt/dnsv5/clean/vctk_wav48_silence_trimmed/mnt/input/clean_fullband/vctk_wav48_silence_trimmed"
    DNS_NOISE_DIR  = "C:/projects/headset-spatial-magnifier/data/datasets_fullband/noise_fullband"
    
    if ADD_REVERB:
        OUTPUT_DIR = "C:/projects/headset-spatial-magnifier/data/generated_dataset_reverb"
    else:
        OUTPUT_DIR = "C:/projects/headset-spatial-magnifier/data/generated_dataset"
        
    NUM_SAMPLES = 1000 
    speech_files, noise_files = parse_dns_dataset(DNS_SPEECH_DIR, DNS_NOISE_DIR)
    
    # Force lexicographical sort to establish matching file arrays across platforms
    speech_files.sort()
    noise_files.sort()
    
    print(f"Executing Controlled Array Simulation. Reverb Active: {ADD_REVERB}")
    print(f"Tracking pool indices: {len(speech_files)} speech items, {len(noise_files)} noise tracks.")

    for i in tqdm(range(NUM_SAMPLES), desc="Synthesizing Acoustic Audio Scenes", unit="sample"):
        simulate_single_scene(i, speech_files, noise_files, OUTPUT_DIR, add_reverb=ADD_REVERB)

    print(f"Generation Sequence Terminated. Target Path: {OUTPUT_DIR}")