import numpy as np

def geometric_mean(times):
    return np.exp(np.mean(np.log(times)))

def estimate_flops(n, time_sec):
    # 5 * N * log2(N) is a rough estimate
    ops = 5 * n * np.log2(n)

    # 1.angle = 2π * k * n / N → 2 multiplies + 1 divide = 3 FLOPs
    # 2.std::polar() (i.e., cos/sin) → approx 25 FLOPs
    # std::polar(1.0f, theta) = std::cos(theta) + i * std::sin(theta)
    # sin() or cos() ≈ 12 FLOPs each, using using polynomial approximations (e.g. Taylor or minimax)
    # 3.Complex multiply (data[n] * w) → 4 FLOPs
    # 4.Complex add (sum += ...) → 2 FLOPs

# Total ≈ 34 FLOPs per (k, n) pair
    # ops = 34 * (n ** 2) # CPU naive
    return ops / time_sec