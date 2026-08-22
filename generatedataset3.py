import glob
import json
import os
import random
import sys

import h5py
import librosa
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm

def convert_wav_folder_to_hdf5(
    wav_dir, output_h5_path, meta_json_path=None, val_split=0.2, target_len=64000
):
    print(f"\nPacking WAV dataset from '{wav_dir}' into '{output_h5_path}'...")

    mix_files = sorted(glob.glob(os.path.join(wav_dir, "*_mix.wav")))

    if len(mix_files) == 0:
        print("No mixture files found to pack!")
        return

    stacked_samples_list = []
    all_metadata_list = []

    for idx, mix_path in enumerate(tqdm(mix_files, desc="Reading WAVs")):
        sample_prefix = mix_path.replace("_mix.wav", "")
        target_path = f"{sample_prefix}_target_gt.wav"
        meta_path = f"{sample_prefix}_meta.json"

        # Read audio [samples, channels] -> Transpose to [channels, samples]
        mix_audio, _ = sf.read(mix_path)
        target_audio, _ = sf.read(target_path)

        if mix_audio.ndim == 2:
            mix_audio = mix_audio.T
            target_audio = target_audio.T
        elif mix_audio.ndim == 1:
            mix_audio = np.expand_dims(mix_audio, axis=0)
            target_audio = np.expand_dims(target_audio, axis=0)

        # Force exact sample length (64,000 samples)
        def fix_length(arr, expected_len):
            curr_len = arr.shape[1]
            if curr_len > expected_len:
                return arr[:, :expected_len]
            elif curr_len < expected_len:
                return np.pad(
                    arr, ((0, 0), (0, expected_len - curr_len)), mode="constant"
                )
            return arr

        mix_audio = fix_length(mix_audio, target_len)
        target_audio = fix_length(target_audio, target_len)

        # Derive residual noise array [channels, samples]
        noise_audio = mix_audio - target_audio

        # Stack into 3D sample matrix: [3, channels, samples]
        # index 0: mixture | index 1: target | index 2: noise
        stacked_sample = np.stack([mix_audio, target_audio, noise_audio], axis=0)
        stacked_samples_list.append(stacked_sample)

        meta_data = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_data = json.load(f)

        meta_data["sample_idx"] = idx
        meta_data["n_samples"] = target_len

        all_metadata_list.append(meta_data)

    # 4D Array -> Shape: (N, 3, channels, length) e.g., (N, 3, 3, 64000)
    full_dataset_array = np.array(stacked_samples_list, dtype=np.float32)

    # Perform Train / Validation Split
    num_samples = len(mix_files)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    train_data = full_dataset_array[:num_train]
    val_data = full_dataset_array[num_train:]

    train_meta_raw = all_metadata_list[:num_train]
    val_meta_raw = all_metadata_list[num_train:]

    if len(val_data) == 0:
        val_data = train_data
        val_meta_raw = train_meta_raw

    # Convert metadata to string-keyed dictionary: {"0": meta0, "1": meta1, ...}
    train_meta_dict = {str(i): meta for i, meta in enumerate(train_meta_raw)}
    val_meta_dict = {str(i): meta for i, meta in enumerate(val_meta_raw)}

    with h5py.File(output_h5_path, "w") as h5f:
        # Train and Val stage datasets with shape (N, 3, 3, 64000)
        h5f.create_dataset("train", data=train_data, compression="gzip", chunks=True)
        h5f.create_dataset("val", data=val_data, compression="gzip", chunks=True)

        h5f.attrs["sr"] = 16000
        h5f.attrs["num_samples"] = num_samples

    if meta_json_path:
        meta_out = {
            "train": train_meta_dict,
            "val": val_meta_dict
        }
        with open(meta_json_path, "w") as f:
            json.dump(meta_out, f, indent=4)

    print(f"\n[+] HDF5 Dataset regenerated with stacked 4D Shape: {full_dataset_array.shape}")
    print(f"    - Train Array Shape: {train_data.shape}")
    print(f"    - Val Array Shape:   {val_data.shape}")
def create_safe_headphone_physical_array(centroid_x, centroid_y, height):
    head_width = 0.16
    cup_base_width = 0.04
    cup_apex_height = 0.03

    half_head = head_width / 2.0
    half_base = cup_base_width / 2.0

    real_left_front = [centroid_x - half_head, centroid_y + half_base, height]
    real_left_back = [centroid_x - half_head, centroid_y - half_base, height]
    real_left_top = [
        centroid_x - half_head,
        centroid_y,
        height + cup_apex_height,
    ]

    return np.array([
        real_left_top,  # Channel 0
        real_left_front,  # Channel 1
        real_left_back,  # Channel 2
    ]).T


def parse_dns_dataset(dns_speech_dir, dns_noise_dir):
    speech_files = glob.glob(
        os.path.join(dns_speech_dir, "**/*.wav"), recursive=True
    )
    noise_files = glob.glob(
        os.path.join(dns_noise_dir, "**/*.wav"), recursive=True
    )
    return speech_files, noise_files


def load_and_rescale_audio(file_path, target_len_samples, target_fs=16000):
    data, fs = sf.read(file_path)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if fs != target_fs:
        data = librosa.resample(data, orig_sr=fs, target_sr=target_fs)

    if len(data) >= target_len_samples:
        return data[:target_len_samples]
    else:
        return np.pad(data, (0, target_len_samples - len(data)), "constant")


def compute_active_rms(signal):
    eps = 1e-10
    return np.sqrt(np.mean(signal**2) + eps)


def simulate_single_scene(sample_idx, speech_files, noise_files, output_dir):
    fs = 16000
    duration_samples = 4 * fs  # 64,000 samples

    room_W = random.uniform(2.5, 5.0)
    room_L = random.uniform(3.0, 9.0)
    room_H = random.uniform(2.2, 3.5)
    room_dim = np.array([room_L, room_W, room_H])

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(1.0), max_order=0
    )

    c_x = random.uniform(room_L * 0.2, room_L * 0.8)
    c_y = random.uniform(room_W * 0.2, room_W * 0.8)
    c_z = random.uniform(1.2, 1.5)

    mic_matrix = create_safe_headphone_physical_array(c_x, c_y, c_z)
    room.add_microphone_array(pra.MicrophoneArray(mic_matrix, room.fs))

    num_interferers = random.randint(1, 3)
    chosen_speech = random.sample(speech_files, 1 + num_interferers)
    target_path = chosen_speech[0]
    interf_paths = chosen_speech[1:]
    noise_path = random.choice(noise_files)

    s_target = load_and_rescale_audio(target_path, duration_samples, fs)
    s_interfs = [
        load_and_rescale_audio(p, duration_samples, fs) for p in interf_paths
    ]
    s_noise = load_and_rescale_audio(noise_path, duration_samples, fs)

    target_coords = np.array([c_x, c_y + 0.12, c_z - 0.10])

    def get_distant_interferer_coords():
        while True:
            coords = np.array([
                random.uniform(0.2, room_L - 0.2),
                random.uniform(0.2, room_W - 0.2),
                random.uniform(1.2, 1.8),
            ])
            if np.linalg.norm(coords - np.array([c_x, c_y, c_z])) > 0.5:
                return coords

    interf_coords_list = [get_distant_interferer_coords() for _ in s_interfs]
    noise_coords = get_distant_interferer_coords()

    room.add_source(target_coords, signal=s_target)
    for i, s_interf in enumerate(s_interfs):
        room.add_source(interf_coords_list[i], signal=s_interf)
    room.add_source(noise_coords, signal=s_noise)

    room.compute_rir()
    room.simulate()
    mixture = room.mic_array.signals
    num_mics = mic_matrix.shape[1]

    target_spatial = np.zeros_like(mixture)
    interf_spatial = np.zeros_like(mixture)
    noise_spatial = np.zeros_like(mixture)

    target_src_idx = 0
    interf_src_idxs = list(range(1, 1 + num_interferers))
    noise_src_idx = 1 + num_interferers

    for m in range(num_mics):
        conv_t = np.convolve(s_target, room.rir[m][target_src_idx])
        out_len_t = min(len(conv_t), mixture.shape[1])
        target_spatial[m, :out_len_t] = conv_t[:out_len_t]

        for i, s_interf in enumerate(s_interfs):
            conv_i = np.convolve(s_interf, room.rir[m][interf_src_idxs[i]])
            out_len_i = min(len(conv_i), mixture.shape[1])
            interf_spatial[m, :out_len_i] += conv_i[:out_len_i]

        conv_n = np.convolve(s_noise, room.rir[m][noise_src_idx])
        out_len_n = min(len(conv_n), mixture.shape[1])
        noise_spatial[m, :out_len_n] = conv_n[:out_len_n]

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

    max_peak = np.max(np.abs(final_mixture))
    if max_peak > 0:
        scale_factor = 0.9 / max_peak
        final_mixture *= scale_factor
        target_spatial *= scale_factor

    os.makedirs(output_dir, exist_ok=True)
    sf.write(f"{output_dir}/sample_{sample_idx}_mix.wav", final_mixture.T, fs)
    sf.write(
        f"{output_dir}/sample_{sample_idx}_target_gt.wav", target_spatial.T, fs
    )

    meta_data = {
        "room_size_lwh": room_dim.tolist(),
        "rt60_seconds": 0.0,
        "sir_db": target_sir_db,
        "snr_db": target_snr_db,
        "headset_centroid": [c_x, c_y, c_z],
        "target_position": target_coords.tolist(),
        "num_interferers": num_interferers,
        "interferer_positions": [ic.tolist() for ic in interf_coords_list],
        "noise_position": noise_coords.tolist(),
    }
    with open(f"{output_dir}/sample_{sample_idx}_meta.json", "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    DNS_SPEECH_DIR = "C:/projects/headset-spatial-magnifier/data/datasets_fullband/clean_fullband/mnt/dnsv5/clean/vctk_wav48_silence_trimmed/mnt/input/clean_fullband/vctk_wav48_silence_trimmed"
    DNS_NOISE_DIR = (
        "C:/projects/headset-spatial-magnifier/data/datasets_fullband/noise_fullband"
    )
    OUTPUT_DIR = (
        "C:/projects/headset-spatial-magnifier/data/generated_dataset3"
    )
    NUM_SAMPLES = 2000

    speech_files, noise_files = parse_dns_dataset(
        DNS_SPEECH_DIR, DNS_NOISE_DIR
    )
    print(
        f"Found {len(speech_files)} speech files, {len(noise_files)} noise files"
    )

    for i in tqdm(range(NUM_SAMPLES), desc="Generating", unit="sample"):
        simulate_single_scene(i, speech_files, noise_files, OUTPUT_DIR)

    print("All samples generated.")
   
    HDF5_OUTPUT = "C:/projects/headset-spatial-magnifier/data/prep_mix_ch3_dataset3.hdf5"
    META_OUTPUT = "C:/projects/headset-spatial-magnifier/data/prep_mix_meta_ch3_dataset3.json"

    convert_wav_folder_to_hdf5(
        OUTPUT_DIR, HDF5_OUTPUT, META_OUTPUT, val_split=0.2
    )
    print("[+] Dataset generation and metadata writing complete!")
    sys.stdout.flush()
    sys.exit(0)