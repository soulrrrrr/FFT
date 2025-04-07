#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include "io_utils.hpp"

void my_fft(std::vector<std::complex<float>> &data)
{
    return;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./fft_cpu <input_path>" << std::endl;
        return 1;
    }

    // init data

    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    size_t N = data.size();

    for (int i = 0; i < N; ++i)
    {
        data[i] = {float(i), 0.0f};
    }

    auto start = std::chrono::high_resolution_clock::now();
    my_fft(data);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;
}