# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 22:38:04 2025
@author: c3055922

Convert the events .dat file in ./outputs to a NumPy .npy file
with columns [x, y, polarity, timestamp_seconds]. Similar to the video script,
this uses hardcoded filenames based on the .dat location.
"""

import os
import sys
import numpy as np

# Append src directory to import custom loader
sys.path.append("../../src")
from dat_files import load_dat_event

# Hardcoded input .dat file path (same as video output)
filename = './outputs/ev_100_10_100_40_0.4_0.01.dat'
# Derive output .npy path by replacing extension
output_path = filename[:-4] + '.npy'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load events: returns ts (µs), x, y, p arrays
ts, x, y, p = load_dat_event(filename)
print(f"Loaded {ts.size} events from '{filename}'")

# Convert timestamp from microseconds to seconds
timestamps_sec = ts.astype(np.float64) * 1e-6

# Ensure integer types for coordinates and polarity
x = x.astype(np.int32)
y = y.astype(np.int32)
p = p.astype(np.int32)

# Stack into (N,4) array: [x, y, polarity, timestamp_seconds]
data = np.stack([x, y, p, timestamps_sec], axis=1)

# Save to .npy
np.save(output_path, data)
print(f"Saved {data.shape[0]} events to '{output_path}', shape {data.shape}")
