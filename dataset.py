import torch
from torch.utils.data import Dataset
import soundfile as sf
import os
import glob

class HeadsetSpatialMagnifierDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # Locate all available spatial mixture components
        self.mix_files = sorted(glob.glob(os.path.join(dataset_dir, "*_mix.wav")))
        
    def __len__(self):
        return len(self.mix_files)
        
    def __getitem__(self, idx):
        mix_path = self.mix_files[idx]
        target_path = mix_path.replace("_mix.wav", "_target_gt.wav")
        
        # Read the multi-channel files
        mix_sig, _ = sf.read(mix_path)
        target_sig, _ = sf.read(target_path)
        
        # Convert to PyTorch Tensors and transpose to (Channels, Samples)
        mix_tensor = torch.FloatTensor(mix_sig).T
        target_tensor = torch.FloatTensor(target_sig).T
        
        # Strict alignment target length: 4 seconds * 16000 Hz = 64000 samples
        target_len = 64000
        
        # --- DYNAMIC ENVELOPE FIX ---
        # Trim or pad the mixture tensor
        if mix_tensor.shape[1] >= target_len:
            mix_tensor = mix_tensor[:, :target_len]
        else:
            padding = target_len - mix_tensor.shape[1]
            mix_tensor = torch.nn.functional.pad(mix_tensor, (0, padding))
            
        # Trim or pad the target tensor
        if target_tensor.shape[1] >= target_len:
            target_tensor = target_tensor[:, :target_len]
        else:
            padding = target_len - target_tensor.shape[1]
            target_tensor = torch.nn.functional.pad(target_tensor, (0, padding))
        
        # --- THE SPATIAL-MAGNIFIER SEPARATION ---
        # Network Inputs (X): Extract channels 1, 2, 3, 4 (The 4 Real Headset Mics)
        x_inputs = mix_tensor[1:5, :]
        
        # Network Target Labels (Y): Extract channels 0 and 5 (The 2 Virtual Mics)
        y_targets = target_tensor[[0, 5], :]
        
        return x_inputs, y_targets
        # Read the multi-channel files: Output shapes are (Samples, Channels)
        mix_sig, _ = sf.read(mix_path)
        target_sig, _ = sf.read(target_path)
        
        # Convert to PyTorch Tensors and transpose to standard (Channels, Samples)
        mix_tensor = torch.FloatTensor(mix_sig).T
        target_tensor = torch.FloatTensor(target_sig).T
        
        # --- THE SPATIAL-MAGNIFIER SEPARATION ---
        # Network Inputs (X): Extract channels 1, 2, 3, 4 (The 4 Real Headset Mics)
        x_inputs = mix_tensor[1:5, :]
        
        # Network Target Labels (Y): Extract channels 0 and 5 (The 2 Virtual Mics)
        y_targets = target_tensor[[0, 5], :]
        
        return x_inputs, y_targets