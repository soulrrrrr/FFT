#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include "vkFFT.h"
#include "utils_VkFFT.h"
#include <chrono>
#include "io_utils.hpp"
#include <cuda.h>
#include <nvrtc.h>
#include <cuComplex.h>

void print_cuda_error(cudaError_t err, const char *context)
{
    std::cerr << "[CUDA ERROR] " << context << ": "
              << cudaGetErrorName(err) << " - "
              << cudaGetErrorString(err) << std::endl;
}

void print_cu_error(CUresult res, const char *context)
{
    const char *err_str = nullptr;
    cuGetErrorName(res, &err_str);
    std::cerr << "[CU ERROR] " << context << ": "
              << (err_str ? err_str : "Unknown") << " (" << res << ")\n";
}

void print_vkfft_error(VkFFTResult res, const char *context)
{
    std::cerr << "[VkFFT ERROR] " << context << ": VkFFTResult = " << res << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_data.txt>" << std::endl;
        return 1;
    }

    VkGPU *vkGPU = new VkGPU{};
    VkFFTResult resFFT = VKFFT_SUCCESS;
    CUresult res = CUDA_SUCCESS;
    cudaError_t res2 = cudaSuccess;
    vkGPU->device_id = 0;

    res = cuInit(0);
    if (res != CUDA_SUCCESS)
    {
        print_cu_error(res, "cuInit");
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    }

    res2 = cudaSetDevice((int)vkGPU->device_id);
    if (res2 != cudaSuccess)
    {
        print_cuda_error(res2, "cudaSetDevice");
        return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
    }

    res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
    if (res != CUDA_SUCCESS)
    {
        print_cu_error(res, "cuDeviceGet");
        return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
    }

    res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
    if (res != CUDA_SUCCESS)
    {
        print_cu_error(res, "cuCtxCreate");
        return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
    }

    // 讀入資料
    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    size_t N = data.size();

    // zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 1;
    configuration.size[0] = (int)N;
    uint64_t bufferSize = sizeof(float) * 2 * configuration.size[0];

    cuFloatComplex *buffer = nullptr;
    res2 = cudaMalloc((void **)&buffer, bufferSize);
    if (res2 != cudaSuccess)
    {
        print_cuda_error(res2, "cudaMalloc");
        return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    }
    configuration.buffer = (void **)&buffer;
    configuration.bufferSize = &bufferSize;
    configuration.device = &vkGPU->device;

    resFFT = transferDataFromCPU(vkGPU, data.data(), &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS)
    {
        print_vkfft_error(resFFT, "transferDataFromCPU");
        return resFFT;
    }

    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS)
    {
        print_vkfft_error(resFFT, "initializeVkFFT");
        return resFFT;
    }

    // 執行 FFT
    VkFFTLaunchParams launchParams = {};
    auto start = std::chrono::high_resolution_clock::now();

    resFFT = VkFFTAppend(&app, -1, &launchParams); // forward FFT
    if (resFFT != VKFFT_SUCCESS)
    {
        print_vkfft_error(resFFT, "VkFFTAppend (forward)");
        return resFFT;
    }
    res2 = cudaDeviceSynchronize();
    if (res2 != cudaSuccess)
        return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    // Optional: 拿回結果寫入檔案...
    std::vector<std::complex<float>> result(configuration.size[0]);

    resFFT = transferDataToCPU(vkGPU, result.data(), &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS)
    {
        print_vkfft_error(resFFT, "transferDataToCPU");
        return resFFT;
    }

    std::string out_file = "../data/output_fft_vkfft_" + std::to_string(configuration.size[0]) + ".txt";
    write_complex_data(result, out_file);
    // std::cout << "FFT result written to " << out_file << std::endl;

    cudaFree(buffer);
    deleteVkFFT(&app);
    cuCtxDestroy(vkGPU->context);
    delete vkGPU;

    return 0;
}