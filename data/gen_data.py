import numpy as np
import os

# Output directory
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# Generate complex input signals for 2^4 to 2^18
for exp in range(4, 19):
    N = 2**exp
    real_part = np.random.uniform(-1, 1, N).astype(np.float32)
    imag_part = np.random.uniform(-1, 1, N).astype(np.float32)
    complex_data = np.stack([real_part, imag_part], axis=1)

    filename = os.path.join(output_dir, f"input_{N}.txt")
    np.savetxt(filename, complex_data, fmt="%.6f")  # Format: real imag

output_files = sorted(os.listdir(output_dir))
output_files