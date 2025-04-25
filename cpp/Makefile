CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -g

# Paths
BIN_DIR = ../bin
FFTW_INCLUDE = $(HOME)/local/fftw/include
FFTW_LIB     = $(HOME)/local/fftw/lib

# Targets
TARGETS = $(BIN_DIR)/fft_cpu $(BIN_DIR)/fftw $(BIN_DIR)/vkfft

all: $(TARGETS)

$(BIN_DIR)/fft_cpu: fft_cpu.cpp
	$(CXX) $(CXXFLAGS) fft_cpu.cpp -o $(BIN_DIR)/fft_cpu

$(BIN_DIR)/fftw: fftw.cpp
	$(CXX) $(CXXFLAGS) fftw.cpp -I$(FFTW_INCLUDE) -L$(FFTW_LIB) -lfftw3f -lm -o $(BIN_DIR)/fftw

$(BIN_DIR)/vkfft: vkfft.cpp utils_VkFFT.cpp
	$(CXX) $(CXXFLAGS) vkfft.cpp utils_VkFFT.cpp -I. -DVKFFT_BACKEND=1 -lcuda -lcudart -lnvrtc -o $(BIN_DIR)/vkfft

clean:
	rm -f $(BIN_DIR)/*