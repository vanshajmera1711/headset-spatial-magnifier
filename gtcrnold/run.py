import datetime
import os
import sounddevice as sd
import numpy as np
import torch
import soundfile as sf
from multiprocessing import Process, Queue
import time

# --- IMPORT YOUR MODEL ---
from models.gtcrn import GTCRN 

# --- CONFIGURATION ---
FS = 16000
BLOCK_SHIFT = 256
N_FFT = 512
N_MICS = 4
CIRM_C = 1
CIRM_K = 1
model_file = "./logs/gtcrn_ch2_sp2.ckpt"

# --- INPUT GAIN ---
INPUT_GAIN = 1.0 

# Stream Settings
CONTEXT_FRAMES = 10  
DEVICE = "cpu"      

# File Names
FILE_NOISY = "./streaming_output/recording_noisy.wav"
FILE_CLEAN = "./streaming_output/recording_clean.wav"

class AudioProcessor(Process):
    def __init__(self, in_queue):
        super().__init__()
        self.in_queue = in_queue
        self.window = torch.sqrt(torch.hann_window(N_FFT)).to(DEVICE)

        # --- UNIQUE FILENAME GENERATION ---
        # Get current time: YearMonthDay_HourMinuteSecond
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure the output folder exists
        os.makedirs("./streaming_output", exist_ok=True)
        
        # Create unique paths and store them in self
        self.file_noisy = f"./streaming_output/recording_noisy_{timestamp}.wav"
        self.file_clean = f"./streaming_output/recording_clean_{timestamp}.wav"

    def get_stft(self, td_signal, stft_length=N_FFT, stft_shift=BLOCK_SHIFT, return_complex=True, device="cpu"):
        """
        Replicates training STFT behavior exactly.

        Args:
            td_signal: Tensor of shape [C, T] or [T]
            stft_length: FFT size (e.g., 512)
            stft_shift: hop size (e.g., 256)
            return_complex: whether to return PyTorch complex tensor or split real/imag
            device: CPU or CUDA

        Returns:
            If td_signal was:
                [T]       → STFT: [F, T] or [F, T, 2]
                [C, T]    → STFT: [C, F, T] or [C, F, T, 2]
        """

        # training window
        window = torch.sqrt(torch.hann_window(stft_length)).to(device)

        # ---------------------------------------------------
        # Case 1: single-channel [T]
        # ---------------------------------------------------
        if td_signal.ndim == 1:
            return torch.stft(
                td_signal,
                n_fft=stft_length,
                hop_length=stft_shift,
                win_length=stft_length,
                window=window,
                center=True,
                onesided=True,
                return_complex=return_complex
            )

        # ---------------------------------------------------
        # Case 2: multi-channel or batch [C, T]
        # ---------------------------------------------------
        elif td_signal.ndim == 2:
            C, T = td_signal.shape

            # reshape to [C, T]
            reshaped = td_signal.reshape(C, T)

            stfts = torch.stft(
                reshaped,
                n_fft=stft_length,
                hop_length=stft_shift,
                win_length=stft_length,
                window=window,
                center=True,
                onesided=True,
                return_complex=return_complex
            )

            # return shapes:
            #   complex=True:     [C, F, TT]
            #   complex=False:    [C, F, TT, 2]
            return stfts

        else:
            raise RuntimeError("td_signal must be [T] or [C, T].")


    def get_complex_masks(self, real_mask):
        compressed_complex_speech_mask = real_mask[:, 0, ...] + (1j) * real_mask[:, 1, ...]
        complex_speech_mask = (-1 / CIRM_C) * torch.log((CIRM_K - CIRM_K * compressed_complex_speech_mask) / (CIRM_K + CIRM_K * compressed_complex_speech_mask))
        complex_noise_mask = (1 - torch.real(complex_speech_mask)) - (1j) * torch.imag(complex_speech_mask)
        return complex_speech_mask, complex_noise_mask
    
    def get_output(self, stft, stft_length=N_FFT, stft_shift=BLOCK_SHIFT, device="cpu"):
        """
        Convert a single-channel STFT back to waveform.
        stft shape: [1, F, T] (complex)   OR   [1, F, T, 2] (real/imag)
        Returns: [T]
        """

        # sqrt-Hann window (training accurate)
        window = torch.sqrt(torch.hann_window(stft_length)).to(device)

        # Remove batch dimension: [1, F, T] → [F, T]
        stft = stft.squeeze(0)
        # print(stft.shape)

        # Run ISTFT (handles both complex and real/imag)
        td = torch.istft(
            stft,
            n_fft=stft_length,
            hop_length=stft_shift,
            win_length=stft_length,
            window=window,
            center=True,
            onesided=True,
            return_complex=False
        )

        return td     # shape: [T]

    def run(self):
        torch.set_num_threads(1)
        print(f"Loading GTCRN on {DEVICE}...")
        
        try:
            model = GTCRN(n_channels=N_MICS).to(DEVICE)
            ckpt = torch.load(model_file, map_location=DEVICE)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)
            model.eval()
            # model = torch.jit.script(model)
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODEL: {e}")
            return

        # Initialize Buffers
        buffer_len = (CONTEXT_FRAMES + 2) * BLOCK_SHIFT
        audio_buffer = np.zeros((buffer_len, N_MICS), dtype=np.float32)
        # ola_buffer = np.zeros(BLOCK_SHIFT)
        
        counter = 0

        print(f"OPENING FILES:\n 1. {self.file_noisy} (4-Channel Raw)\n 2. {self.file_clean} (1-Channel Enhanced)")
        
        # --- UPDATE 1: Set noisy file to 4 channels ---
        with sf.SoundFile(self.file_noisy, mode='w', samplerate=FS, channels=N_MICS) as f_noisy, \
             sf.SoundFile(self.file_clean, mode='w', samplerate=FS, channels=1) as f_clean:
            
            while True:
                start_time = time.perf_counter()
                indata = self.in_queue.get()
                if indata is None: break
                
                # Apply Gain
                new_audio = indata[:,1:N_MICS+1]
                
                #print(f"The shape of recorded data is {new_audio.shape}")
                
                # --- UPDATE 2: Write all 4 columns (channels) ---
                f_noisy.write(new_audio)

                # --- PROCESS ---
                # 1. Update Buffer
                audio_buffer = np.roll(audio_buffer, -BLOCK_SHIFT, axis=0)
                audio_buffer[-BLOCK_SHIFT:, :] = new_audio
                
                # 2. STFT
                input_tensor = torch.from_numpy(audio_buffer).float().T.to(DEVICE)
                input_stft = self.get_stft(input_tensor)
                
                # 3. Model Inference
                noisy_stft = input_stft.unsqueeze(0)  #[1,4,F,T]
                stacked_in = torch.cat((torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1)   #[1,8,F,T]
                # print(f"stacked_stft SHAPE IS {stacked_in.shape}")
                
                with torch.no_grad():
                    stacked_mask = model(stacked_in)
                    # print(f"stacked_mask SHAPE IS {stacked_mask.shape}")    #[1,2,F,T]
                
                # 4. Decode Mask
                DELAY = 2 
                mask_slice = stacked_mask[..., -DELAY-1:-DELAY+1]     #[1,2,F,2]
                # print(f"mask_slice SHAPE IS {mask_slice.shape}")
                speech_mask, noise_mask = self.get_complex_masks(mask_slice)        #speech_mask(complex) [1,F,2]
                # print (f"speech_mask SHAPE IS {speech_mask.shape}")

                ref_noisy_slice = noisy_stft[:, 0, :, -DELAY-1:-DELAY+1]      #[1,F,2]
                # print (f"ref_noisy_slice SHAPE IS {ref_noisy_slice.shape}") 
                est_clean_stft = ref_noisy_slice * speech_mask              #[1,F,2]
                # print (f"est_clean_stft SHAPE IS {est_clean_stft.shape}")
                # print ("###########################\n###########################\n###########################\n")
                
                # 5. Inverse STFT & Overlap-Add
                est_frame_td = self.get_output(est_clean_stft)
                frame_numpy = est_frame_td.numpy()
                
                out_chunk = frame_numpy[:BLOCK_SHIFT] #+ ola_buffer
                # ola_buffer = frame_numpy[BLOCK_SHIFT:]
                
                # Write Clean Output
                out_final = np.clip(out_chunk, -1.0, 1.0)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000
                print (f"The latency is {latency:.2f}", )

                f_clean.write(out_final)
                
                # Stats
                counter += 1
                if counter % 60 == 0:
                    in_peak = np.max(np.abs(new_audio))
                    out_peak = np.max(np.abs(out_final))
                    print(f"Stats: In Peak={in_peak:.3f} | Out Peak={out_peak:.3f}")

def audio_callback(indata, outdata, frames, time, status):
    if status: print(status)
    in_queue.put(indata.copy())
    outdata.fill(0)

if __name__ == "__main__":
    print(sd.query_devices())
    try:
        input_device = int(input("Input Device ID: "))
    except:
        input_device = sd.default.device[0]
    
    # We use a large queue so the recording never skips even if processing lags
    in_queue = Queue(maxsize=1000)

    processor = AudioProcessor(in_queue)
    # FIX 1: Daemonize the process so it is killed automatically if the main script ends
    processor.daemon = True 
    processor.start()

    print("\n--- RECORDING STARTED ---")
    print(f"Gain set to {INPUT_GAIN}x.")
    print("Press Ctrl+C to stop and save files.")

    try:
        with sd.Stream(device=(input_device, sd.default.device[1]),
                       channels=(6, 1), 
                       samplerate=FS, blocksize=BLOCK_SHIFT,
                       callback=audio_callback):
            while True: 
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Attempt to signal the processor to stop gracefully
        try:
            in_queue.put_nowait(None)
        except:
            pass
            
        # FIX 2: Wait 1 second for graceful exit, then force kill
        processor.join(timeout=1.0)
        
        if processor.is_alive():
            print("Processor stuck. Forcing shutdown...")
            processor.terminate()
            processor.join() # Wait for the termination to finish
            
        print("Done.")