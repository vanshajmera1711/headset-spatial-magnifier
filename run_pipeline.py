import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from generate_dataset import simulate_single_scene, parse_dns_dataset

def main():
    # 1. Load configuration file
    config_path = os.path.join("config", "simulation_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    print("--- Crawling DNS Dataset Directories ---")
    speech_files, noise_files = parse_dns_dataset(cfg["dns_speech_dir"], cfg["dns_noise_dir"])
    
    if len(speech_files) < 2 or len(noise_files) < 1:
        print("\n[ERROR] Could not find enough .wav files.")
        print(f"Please check that your DNS data is placed inside:\n-> {cfg['dns_speech_dir']}\n-> {cfg['dns_noise_dir']}")
        return

    # 2. Prepare output directory
    os.makedirs(cfg["output_dataset_dir"], exist_ok=True)
    
    total_samples = cfg["total_training_samples"]
    print(f"\nDeploying parallel CPU workers to generate {total_samples} spatial scenes...")
    
    # 3. Launch parallel task runner across all available CPU threads
    Parallel(n_jobs=-1)(
        delayed(simulate_single_scene)(
            i, speech_files, noise_files, cfg["output_dataset_dir"]
        ) for i in tqdm(range(total_samples), desc="Generating Dataset")
    )
    
    print(f"\n[SUCCESS] Pipeline execution complete!")
    print(f"Your multi-channel dataset is stored at: {cfg['output_dataset_dir']}")

if __name__ == "__main__":
    main()