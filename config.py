# config.py

# Executables
EXECUTABLE_FFTW = "./bin/fftw"
EXECUTABLE_FFT_CPU = "./bin/fft_cpu"
EXECUTABLE_GPU = "./bin/vkfft_runner"  # for future

# Benchmark sizes and repetitions
SIZES = [2**i for i in range(4, 15)]   # 2^4 to 2^18
NUM_TRIALS = 10

# Output
RESULTS_DIR = "results"