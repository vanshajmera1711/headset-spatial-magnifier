import os
import numpy as np
import scipy.io.wavfile as wav

def build_local_mock_assets():
    print("--- Generating Local Mock Speech & Noise Assets ---")
    
    speech_dir = os.path.join("data", "dns_clean_speech")
    noise_dir = os.path.join("data", "dns_noise")
    
    os.makedirs(speech_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    fs = 16000
    duration = 5.0  # 5 seconds each
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Synthesize Mock Speaker 1 (Male Vocal Fundamental Frequency Range ~120Hz + harmonics)
    # We modulate it with a low-frequency envelope so it sounds varying like human speech vowels
    envelope1 = 0.5 * (1.0 + np.sin(2 * np.pi * 0.7 * t))
    speaker1 = envelope1 * (np.sin(2 * np.pi * 130 * t) + 0.5 * np.sin(2 * np.pi * 260 * t))
    speaker1 = speaker1 * 0.2
    
    # 2. Synthesize Mock Speaker 2 (Female Vocal Fundamental Frequency Range ~210Hz + harmonics)
    envelope2 = 0.5 * (1.0 + np.sin(2 * np.pi * 1.1 * t))
    speaker2 = envelope2 * (np.sin(2 * np.pi * 210 * t) + 0.5 * np.sin(2 * np.pi * 420 * t))
    speaker2 = speaker2 * 0.2
    
    # 3. Synthesize Mock Ambient Noise (Low-frequency air conditioning / fan hum)
    # Combining pinkish-filtered noise with a steady transformer hum at 50Hz
    random_white = np.random.randn(len(t))
    # Low-pass filter the noise using a rolling mean to simulate room rumble
    noise_rumble = np.convolve(random_white, np.ones(50)/50, mode='same')
    noise_ambient = noise_rumble + 0.1 * np.sin(2 * np.pi * 50 * t)
    noise_ambient = noise_ambient * 0.15

    # Write files to their target locations as standard 16-bit PCM WAVs
    wav.write(os.path.join(speech_dir, "speaker_1.wav"), fs, (speaker1 * 32767).astype(np.int16))
    wav.write(os.path.join(speech_dir, "speaker_2.wav"), fs, (speaker2 * 32767).astype(np.int16))
    wav.write(os.path.join(noise_dir, "noise_ambient.wav"), fs, (noise_ambient * 32767).astype(np.int16))
    
    print("[SUCCESS] Locally generated speech and noise mocks staged cleanly inside data folders!\n")
    return True

if __name__ == "__main__":
    build_local_mock_assets()