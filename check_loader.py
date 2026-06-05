import torch
from dataset import HeadsetSpatialMagnifierDataset

def test_data_stream():
    print("--- Verifying PyTorch Data Loader Mapping ---")
    dataset_path = "data/synthetic_headset_dataset"
    
    # Instantiate your custom dataset class
    dataset = HeadsetSpatialMagnifierDataset(dataset_path)
    print(f"Total simulated scenes found: {len(dataset)}")
    
    # Create a PyTorch DataLoader
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab a single batch
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"\nSuccessfully loaded Batch #{batch_idx + 1}!")
        print(f"Input Tensor Shape  (Batch, Mics, Samples): {inputs.shape}")
        print(f"Target Tensor Shape (Batch, Mics, Samples): {targets.shape}")
        
        print("\nChannel Mapping Breakdown:")
        print(f"-> Model reads {inputs.shape[1]} Real Microphone Channels.")
        print(f"-> Model predicts {targets.shape[1]} Expanded Virtual Channels.")
        break # Just checking one batch for validation

if __name__ == "__main__":
    test_data_stream()