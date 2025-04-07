import subprocess
import os

def run_cpu_fft(executable_path, n):
    input_file = f"data/input_{n}.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found. Generate input first.")

    result = subprocess.run(
        [executable_path, input_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Execution failed:\n{result.stderr}")

    return float(result.stdout.strip())  # return time in seconds