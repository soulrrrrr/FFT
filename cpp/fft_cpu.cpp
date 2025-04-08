#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>
#include "io_utils.hpp"

void fft_naive(size_t N, std::vector<std::complex<float>> &data, std::vector<std::complex<float>> &out)
{
    for (int k = 0; k < N; k++)
    {
        std::complex<float> sum;
        for (int n = 0; n < N; n++)
        {
            float angle = 2.0f * M_PI * k * n / N;
            std::complex<float> w = std::polar(1.0f, -angle);
            sum += data[n] * w;
        }
        out[k] = sum;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./fft_cpu <input_path>" << std::endl;
        return 1;
    }

    // init data and output
    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    size_t N = data.size();
    std::vector<std::complex<float>> out(N);

    auto start = std::chrono::high_resolution_clock::now();
    fft_naive(N, data, out);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;
}