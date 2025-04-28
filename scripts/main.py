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