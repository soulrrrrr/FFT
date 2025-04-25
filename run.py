# run.py

import argparse
from scripts.main import benchmark_cpu, benchmark_gpu  # CPU benchmark
from config import EXECUTABLE_FFTW, EXECUTABLE_FFT_CPU, EXECUTABLE_VKFFT

import sys

def tee_stdout_to_file(filename):
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(filename, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)  # optional

def main():
    parser = argparse.ArgumentParser(description="FFT Benchmark Tool")
    parser.add_argument("--fftw", action="store_true", help="Run FFTW benchmark")
    parser.add_argument("--cpu", action="store_true", help="Run CPU implementation")
    parser.add_argument("--vkfft", action="store_true", help="Run VkFFT CUDA benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--out", type=str, default=None, help="Output CSV base name")

    args = parser.parse_args()

    fftw_results = []
    fft_cpu_results = []
    vkfft_results = []

    if args.fftw or args.all:
        print("🏁 Running FFTW benchmark...")
        fftw_results = benchmark_cpu(EXECUTABLE_FFTW)

    if args.cpu or args.all:
        print("🏁 Running CPU implementation...")
        fft_cpu_results = benchmark_cpu(EXECUTABLE_FFT_CPU)
    
    if args.vkfft or args.all:
        print("🏁 Running VkFFT implementation...")
        vkfft_results = benchmark_gpu(EXECUTABLE_VKFFT)

    if args.out:
        log_path = args.out.replace(".csv", ".log")
        tee_stdout_to_file(log_path)
    
        import pandas as pd
        import os
        os.makedirs("results", exist_ok=True)

        base = args.out.replace(".csv", "")
        if fftw_results:
            pd.DataFrame(fftw_results, columns=["N", "time", "FLOPS"]).to_csv(f"{base}.csv", index=False)
        if fft_cpu_results:
            pd.DataFrame(fft_cpu_results, columns=["N", "time", "FLOPS"]).to_csv(f"{base}.csv", index=False)
        if vkfft_results:
            pd.DataFrame(vkfft_results, columns=["N", "time", "FLOPS"]).to_csv(f"{base}.csv", index=False)

        print(f"✅ Results saved to {base}.csv")



if __name__ == "__main__":
    main()