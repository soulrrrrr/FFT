#include <iostream>
#include <fftw3.h>
#include <chrono>
#include "io_utils.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./fftw <input_file>" << std::endl;
        return 1;
    }

    // 讀資料 (std::complex<float>)
    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    size_t N = data.size();

    // 準備輸出
    std::vector<std::complex<float>> out(N);

    // FFTW 使用 float precision API (fftwf_)
    fftwf_plan plan = fftwf_plan_dft_1d(
        static_cast<int>(N),
        reinterpret_cast<fftwf_complex *>(data.data()),
        reinterpret_cast<fftwf_complex *>(out.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);

    auto start = std::chrono::high_resolution_clock::now();
    fftwf_execute(plan);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    fftwf_destroy_plan(plan);
    fftwf_cleanup();

    // 輸出檔案名稱根據長度命名
    std::string out_file = "data/output_fftw_" + std::to_string(N) + ".txt";
    write_complex_data(out, out_file);

    return 0;
}