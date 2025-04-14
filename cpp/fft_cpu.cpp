#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>
#include "io_utils.hpp"
#include <immintrin.h>

struct complex_t
{
    float Re, Im;
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
};

inline complex_t operator+(const complex_t &x, const complex_t &y)
{
    return complex_t(x.Re + y.Re, x.Im + y.Im);
}

inline complex_t operator-(const complex_t &x, const complex_t &y)
{
    return complex_t(x.Re - y.Re, x.Im - y.Im);
}

inline complex_t operator*(const complex_t &x, const complex_t &y)
{
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
}

// __m128 mulpz2f(const __m128 ab, const __m128 xy)
// {
//     const __m128 aa = _mm_unpacklo_ps(ab, ab);      // duplicate real parts
//     const __m128 bb = _mm_unpackhi_ps(ab, ab);      // duplicate imag parts
//     const __m128 yx = _mm_shuffle_ps(xy, xy, 0xB1); // swap real/imag of xy
//     return _mm_addsub_ps(_mm_mul_ps(aa, xy), _mm_mul_ps(bb, yx));
// }

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

// reference: http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham3.html
// example is DIF, I use DIT
void fftr(int n, int s, bool eo, complex_t *x, complex_t *y)
// n  : sequence length
// s  : stride
// eo : x is output if eo == 0, y is output if eo == 1
// x  : input sequence(or output sequence if eo == 0)
// y  : work area(or output sequence if eo == 1)
{
    const int m = n / 2;
    const float angle = 2 * M_PI / n;
    if (n == 1)
    {
        if (eo)
            for (int q = 0; q < s; q++)
                x[q] = y[q];
    }
    else
    {
        fftr(n / 2, 2 * s, !eo, y, x);
        complex_t wp{1.0f, 0.0f};
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
        for (int p = 0; p < m; p++)
        {
            if (s == 1)
            {
                for (int q = 0; q < s; q++)
                {
                    const complex_t a = y[q + s * (2 * p + 0)];
                    const complex_t b = y[q + s * (2 * p + 1)] * wp;
                    x[q + s * (p + 0)] = a + b;
                    x[q + s * (p + m)] = a - b;
                }
            }
            else
            {
                // for (int q = 0; q < s; q += 2)
                // {
                //     const std::complex<float> a = y[q + s * (2 * p + 0)];
                //     const std::complex<float> b = y[q + s * (2 * p + 1)] * wp;
                //     const std::complex<float> c = y[(q + 1) + s * (2 * p + 0)];
                //     const std::complex<float> d = y[(q + 1) + s * (2 * p + 1)] * wp;
                //     x[q + s * (p + 0)] = a + b;
                //     x[q + s * (p + m)] = a - b;
                //     x[(q + 1) + s * (p + 0)] = c + d;
                //     x[(q + 1) + s * (p + m)] = c - d;
                // }
                for (int q = 0; q < s; q += 2)
                {
                    const complex_t a = y[q + s * (2 * p + 0)];
                    const complex_t b = y[q + s * (2 * p + 1)] * wp;
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
                    const complex_t d = y[(q + 1) + s * (2 * p + 1)] * wp;
                    x[q + s * (p + 0)] = a + b;
                    x[q + s * (p + m)] = a - b;
                    x[(q + 1) + s * (p + 0)] = c + d;
                    x[(q + 1) + s * (p + m)] = c - d;
                }
            }
            wp = wp * wp_step;
        }
    }
}

void fft_stockham_recursive(size_t N, std::vector<std::complex<float>> &data, std::vector<std::complex<float>> &out)
{

    fftr(N, 1, 0, (complex_t *)data.data(), (complex_t *)out.data());
    std::copy(data.begin(), data.end(), out.begin()); // for fft stockham
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

    // fft_cooley_tukey(N, data, out);
    auto start = std::chrono::high_resolution_clock::now();
    fft_stockham_recursive(N, data, out);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    std::string out_file = "data/output_fft_cpu_" + std::to_string(N) + ".txt";
    write_complex_data(out, out_file);

    return 0;
}