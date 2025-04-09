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

size_t bit_reverse(size_t x, int log2n)
{
    size_t result = 0;
    for (int i = 0; i < log2n; i++)
    {
        result <<= 1;
        result |= (x & 1);
        x >>= 1;
    }
    return result;
}

void fft_cooley_tukey(size_t N, std::vector<std::complex<float>> &data, std::vector<std::complex<float>> &out)
{
    // permute input to bit reversal
    for (size_t i = 0; i < N; i++)
    {
        out[i] = data[bit_reverse(i, log2(N))];
    }
    // cooley-tukey algorithm
    for (int len = 2; len <= N; len <<= 1)
    {
        int half = len / 2;
        for (int start = 0; start < N; start += len)
        {
            // do [start, start+len)
            float angle = 2.0f * M_PI / len;
            std::complex<float> w_step = std::polar(1.0f, -angle);
            std::complex<float> w = std::polar(1.0f, 0.0f);
            for (int i = 0; i < len / 2; i++)
            {
                // do start+i & start+i+half
                std::complex<float> a = out[start + i];
                std::complex<float> b = out[start + i + half];
                out[start + i] = a + w * b;
                out[start + i + half] = a - w * b;
                w *= w_step;
            }
        }
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
    fft_cooley_tukey(N, data, out);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    std::string out_file = "data/output_fft_cpu_" + std::to_string(N) + ".txt";
    write_complex_data(out, out_file);

    return 0;
}