#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>
#include "io_utils.hpp"
#include <cuda/std/complex>

using my_t = cuda::std::complex<float>;

__host__ void get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Shared memory per SM: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
        std::cout << "Clock rate (kHz): " << deviceProp.clockRate << std::endl;
        std::cout << "Memory clock rate (kHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "Memory bus width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "L2 cache size: " << deviceProp.l2CacheSize << std::endl;
        std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Registers per SM: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "Supports cooperative launch: " << (deviceProp.cooperativeLaunch ? "Yes" : "No") << std::endl;
    }
}

// Device 0 name: NVIDIA TITAN V
// Computational capabilities: 7.0
// Max Global memory size: 12642746368
// Max Constant memory size: 65536
// Max Shared memory size per block: 49152
// Max threads per block: 1024
// Max block dimensions: 1024 x, 1024 y, 64 z
// Max grid dimensions: 2147483647 x, 65535 y, 65535 z
// Warp Size: 32
// Number of SMs: 80
// Max threads per SM: 2048
// Shared memory per SM: 98304
// Clock rate (kHz): 1455000
// Memory clock rate (kHz): 850000
// Memory bus width (bits): 3072
// L2 cache size: 4718592
// Registers per block: 65536
// Registers per SM: 65536

__global__ void FFT_4096(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    constexpr int THREADS = 256;
    __shared__ float input[(32 + 1) * 256]; // padding to avoid bank conflict

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx;
    int local_idx = tx;

    int ELEMENTS = N;
    int PADDED_ELEMENTS = ELEMENTS;

    ELEMENTS = 256;
    int padded_idx = local_idx + local_idx / 32;
    // Load from global memory to shared memory
    for (int i = 0; i < ELEMENTS; i += THREADS)
    {
        // load from global, coalesced
        float2 c = data[idx];
        // store to shared memory, avoid bank conflict
        input[padded_idx] = c.x;
        input[padded_idx + 1] = c.y;
    }
    __syncthreads();

    // Stockham in shared memory
    // for (int len = 2; len <= ELEMENTS; len <<= 1)
    // {
    //     int half = len >> 1;
    //     for (int i = 0; i < ELEMENTS; i += THREADS)
    //     {
    //         int real_idx = local_idx + i;
    //         if (real_idx < ELEMENTS)
    //         {
    //             int group = real_idx / half;
    //             int first = group * len + (real_idx % half);
    //             int second = first + half;

    //             int padded_first = first + first / 32;
    //             int padded_second = second + second / 32;

    //             if (second < ELEMENTS)
    //             {
    //                 float2 u = input[padded_first];
    //                 float2 v = input[padded_second];

    //                 float angle = -2.0f * M_PI * (real_idx % half) / len;
    //                 float2 w = {cosf(angle), sinf(angle)};

    //                 // Complex multiply: w * v
    //                 float2 wv;
    //                 wv.x = w.x * v.x - w.y * v.y;
    //                 wv.y = w.x * v.y + w.y * v.x;

    //                 input[padded_first].x = u.x + wv.x;
    //                 input[padded_first].y = u.y + wv.y;
    //                 input[padded_second].x = u.x - wv.x;
    //                 input[padded_second].y = u.y - wv.y;
    //             }
    //         }
    //     }
    //     __syncthreads();
    // }

    // Write back from shared memory to global memory
    for (int i = 0; i < ELEMENTS; i += THREADS)
    {
        float2 c;
        c.x = input[padded_idx];
        c.y = input[padded_idx + 1];
        out[idx] = c;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./fft_gpu <input_path>" << std::endl;
        return 1;
    }

    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    size_t N = data.size();

    std::vector<float2> host_data(N);
    for (size_t i = 0; i < N; ++i)
    {
        host_data[i] = {data[i].real(), data[i].imag()};
    }

    // Allocate device memory
    float2 *device_data;
    cudaMalloc(&device_data, N * sizeof(float2));
    cudaMemcpy(device_data, host_data.data(), N * sizeof(float2), cudaMemcpyHostToDevice);

    float2 *device_out;
    cudaMalloc(&device_out, N * sizeof(float2));

    // Launch kernel
    dim3 dimBlock(256);
    dim3 dimGrid(N / 4096);
    auto start = std::chrono::high_resolution_clock::now();
    FFT_4096<<<dimGrid, dimBlock>>>(device_data, device_out, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back
    cudaMemcpy(host_data.data(), device_out, N * sizeof(float2), cudaMemcpyDeviceToHost);

    // Timing & write output
    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    std::string out_file = "../data/output_fft_gpu_" + std::to_string(N) + ".txt";
    std::vector<std::complex<float>> result;
    result.reserve(N);
    for (const auto &f2 : host_data)
        result.emplace_back(f2.x, f2.y);
    write_complex_data(result, out_file);

    cudaFree(device_data);
    return 0;
}