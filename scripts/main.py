# scripts/main.py

from runners.cpu_runner import run_cpu_fft
from runners.gpu_runner import run_gpu_fft
from analysis.metrics import geometric_mean, estimate_flops
from config import SIZES, NUM_TRIALS

def benchmark_cpu(path):
    results = []

    for size in SIZES:
        times = []
        for _ in range(NUM_TRIALS):
            try:
                t = run_cpu_fft(path, size)
            except Exception as e:
                print(f"❌ Error running FFT on N={size}: {e}")
                continue
            times.append(t)

        gmean = geometric_mean(times)
        flops = estimate_flops(size, gmean)

        print(f"N={size:<7} Time={gmean:.6f}s  FLOPS={flops/1e6:.2f} MFLOPS")
        results.append((size, gmean, flops))

    return results

def benchmark_gpu(path):
    results = []

    for size in SIZES:
        times = []
        for _ in range(NUM_TRIALS):
            try:
                t = run_gpu_fft(path, size)
            except Exception as e:
                print(f"❌ Error running VkFFT CUDA on N={size}: {e}")
                continue
            times.append(t)

        gmean = geometric_mean(times)
        flops = estimate_flops(size, gmean)

        print(f"N={size:<7} Time={gmean:.6f}s  FLOPS={flops/1e6:.2f} MFLOPS")
        results.append((size, gmean, flops))

    return results

def calculate_max_error_for_sizes():
    import numpy as np
    from config import SIZES

    max_errors = {}

    for size in SIZES:
        try:
            gpu_output = np.loadtxt(f"data/output_fft_gpu_{size}.txt")
            cpu_output = np.loadtxt(f"data/output_fft_cpu_{size}.txt")

            max_error = np.max(np.abs(gpu_output - cpu_output))
            max_errors[size] = max_error
            print(f"⚠️ Maximum error for size {size}: {max_error}")
        except Exception as e:
            print(f"❌ Error calculating maximum error for size {size}: {e}")

    return max_errors