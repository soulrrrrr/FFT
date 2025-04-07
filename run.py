# run.py

import argparse
from scripts.main import benchmark_cpu  # CPU benchmark
# from scripts.main import benchmark_gpu  # GPU benchmark (future)
from config import EXECUTABLE_FFTW, EXECUTABLE_FFT_CPU

def main():
    parser = argparse.ArgumentParser(description="FFT Benchmark Tool")
    parser.add_argument("--fftw", action="store_true", help="Run FFTW benchmark")
    parser.add_argument("--cpu", action="store_true", help="Run CPU implementation")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmark (TODO)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--out", type=str, default=None, help="Output CSV base name")

    args = parser.parse_args()

    fftw_results = []
    fft_cpu_results = []

    if args.fftw or args.all:
        print("🏁 Running FFTW benchmark...")
        fftw_results = benchmark_cpu(EXECUTABLE_FFTW)

    if args.cpu or args.all:
        print("🏁 Running CPU implementation...")
        fft_cpu_results = benchmark_cpu(EXECUTABLE_FFT_CPU)

    if args.out:
        import pandas as pd
        import os
        os.makedirs("results", exist_ok=True)

        base = args.out.replace(".csv", "")
        if fftw_results:
            pd.DataFrame(fftw_results, columns=["N", "time", "FLOPS"]).to_csv(f"{base}_fftw.csv", index=False)
        if fft_cpu_results:
            pd.DataFrame(fft_cpu_results, columns=["N", "time", "FLOPS"]).to_csv(f"{base}_myfft.csv", index=False)

        print(f"✅ Results saved to {base}_fftw.csv and {base}_myfft.csv")

    if args.gpu or args.all:
        print("🚧 GPU benchmark not yet implemented.")
        # gpu_results = benchmark_gpu()
        # TODO: Save GPU results too

if __name__ == "__main__":
    main()