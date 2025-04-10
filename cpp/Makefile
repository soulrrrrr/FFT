CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -g

# Paths
BIN_DIR = ../bin
FFTW_INCLUDE = $(HOME)/local/fftw/include
FFTW_LIB     = $(HOME)/local/fftw/lib

# Targets
TARGETS = $(BIN_DIR)/fft_cpu $(BIN_DIR)/fftw

all: $(TARGETS)

$(BIN_DIR)/fft_cpu: fft_cpu.cpp
	$(CXX) $(CXXFLAGS) fft_cpu.cpp -o $(BIN_DIR)/fft_cpu

$(BIN_DIR)/fftw: fftw.cpp
	$(CXX) $(CXXFLAGS) fftw.cpp -I$(FFTW_INCLUDE) -L$(FFTW_LIB) -lfftw3f -lm -o $(BIN_DIR)/fftw

clean:
	rm -f $(BIN_DIR)/*