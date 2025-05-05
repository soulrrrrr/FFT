# config.py

# Executables
EXECUTABLE_FFTW = "./bin/fftw"
EXECUTABLE_FFT_CPU = "./bin/fft_cpu"
EXECUTABLE_VKFFT = "./bin/vkfft"  # for future
EXECUTABLE_FFT_GPU = "./bin/fft_gpu"

# Benchmark sizes and repetitions
SIZES = [2**i for i in range(4, 19)]   # 2^4 to 2^18
NUM_TRIALS = 10

# Output
RESULTS_DIR = "results"