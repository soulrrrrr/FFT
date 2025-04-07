import numpy as np

def geometric_mean(times):
    return np.exp(np.mean(np.log(times)))

def estimate_flops(n, time_sec):
    # 5 * N * log2(N) is a rough estimate
    ops = 5 * n * np.log2(n)
    return ops / time_sec