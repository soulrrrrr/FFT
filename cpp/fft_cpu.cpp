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

inline size_t bit_reverse(size_t x, int log2n)
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
        float angle = 2.0f * M_PI / len;
        std::complex<float> w_step = std::polar(1.0f, -angle);
        for (int start = 0; start < N; start += len)
        {
            // do [start, start+len)
            std::complex<float> w{1.0f, 0.0f};
            for (int i = 0; i < half; i++)
            {
                // do start+i & start+i+half
                std::complex<float> a = out[start + i];
                std::complex<float> b = out[start + i + half];
                out[start + i] = a + w * b;
                out[start + i + half] = a - w * b;
                // std::cout << start + i << " " << start + i + half << std::endl;
                w = w * w_step;
            }
            // std::cout << "---" << std::endl;
        }
    }
}

void fft_stockham(size_t N, std::vector<std::complex<float>> &data, std::vector<std::complex<float>> &out)
{
    auto *p1 = data.data();
    auto *p2 = out.data();
    // stockham algorithm
    // reference: https://github.com/scientificgo/fft/blob/master/stockham.go
    const int half = N / 2;
    // stride: distance between fft pair
    // eg. N=8 stride=4: (0,4), (1,5), (2,6), (3,7)
    for (int stride = half; stride >= 1; stride >>= 1)
    {
        // block size: size of can use same w
        // int block_size = 2 * stride;
        int block_size = stride << 1;
        // float angle = 2.0f * M_PI / (N / stride);
        float angle = block_size * M_PI / N;
        std::complex<float> w_step = std::polar(1.0f, -angle);

        // block count for same w
        int block_count = N / block_size;
        std::complex<float> w{1.0f, 0.0f};
        for (int b = 0; b < block_count; b++)
        {
            // for stride=n, can reuse w n times
            for (int w_count = 0; w_count < stride; w_count++)
            {
                // std::cout << "in: " << b * block_size + w_count << ", " << b * block_size + w_count + stride << std::endl;
                // std::cout << "out: " << b * stride + w_count << ", " << b * stride + w_count + half << std::endl;
                std::complex<float> a = p1[b * block_size + w_count];
                std::complex<float> wb = w * p1[b * block_size + w_count + stride];
                p2[b * stride + w_count] = a + wb;
                p2[b * stride + w_count + half] = a - wb;
            }
            // update w
            w *= w_step;
        }
        // swap p1 and p2
        std::swap(p1, p2);
    }

    if (p1 != out.data())
    {
        std::copy(p1, p1 + N, out.data());
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
    // fft_cooley_tukey(N, data, out);
    fft_stockham(N, data, out);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    std::string out_file = "data/output_fft_cpu_" + std::to_string(N) + ".txt";
    write_complex_data(out, out_file);

    return 0;
}