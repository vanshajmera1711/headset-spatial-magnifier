import numpy as np
import os

DATA_DIR = "C:/projects/headset-spatial-magnifier/data"
R_true = np.load(os.path.join(DATA_DIR, "true_R_noise.npy"))

# Look at the raw cross-channel phase covariance for a single bin
print("--- TRUE COVARIANCE CORNER SLICE ---")
print(R_true[0:2, 0:2, 32])