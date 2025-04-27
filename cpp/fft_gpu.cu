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

__global__ void FFT_16(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    __shared__ float2 input[16];

    // each thread stores 4 elements
    float2 a, b, c, d;
    int tx = threadIdx.x;
    a = data[tx];
    b = data[tx + 4];
    c = data[tx + 8];
    d = data[tx + 12];

    // 1st stage: (a, c) -> (a, c), (b, d) -> (b, d)
    float2 t1 = make_float2(a.x + c.x, a.y + c.y);
    float2 t2 = make_float2(b.x + d.x, b.y + d.y);
    float2 t3 = make_float2(a.x - c.x, a.y - c.y);
    float2 t4 = make_float2(b.x - d.x, b.y - d.y);
    a = t1;
    b = t2;
    c = t3;
    d = t4;

    // 2nd stage: (a, b) -> (a, c), (c, d) -> (b, d)
    float angle = -2.0f * M_PI / 4.0f;
    float2 w_1 = make_float2(cos(angle), sin(angle));
    float2 t5 = make_float2(a.x + b.x, a.y + b.y);
    float2 t6 = make_float2(a.x - b.x, a.y - b.y);
    float2 t7 = make_float2(c.x + (w_1.x * d.x - w_1.y * d.y),
                            c.y + (w_1.x * d.y + w_1.y * d.x));
    float2 t8 = make_float2(c.x - (w_1.x * d.x - w_1.y * d.y),
                            c.y - (w_1.x * d.y + w_1.y * d.x));
    input[tx] = t5;
    input[tx + 8] = t6;
    input[tx + 4] = t7;
    input[tx + 12] = t8;
    __syncthreads();

    // 3rd stage, need shared memory
    int stride = 2;
    // (a, b) do, (c, d) do
    int padding = tx >= 2 ? 1 : 0;
    a = input[(stride * (tx * 4)) % 16 + padding];
    b = input[(stride * (tx * 4 + 1)) % 16 + padding];
    c = input[(stride * (tx * 4 + 2)) % 16 + padding];
    d = input[(stride * (tx * 4 + 3)) % 16 + padding];

    float angle_1 = 2.0f * M_PI / 8.0f * ((tx * 2) % 4);
    float angle_2 = 2.0f * M_PI / 8.0f * ((tx * 2 + 1) % 4);
    w_1 = make_float2(cos(angle_1), sin(-angle_1));
    float2 w_2 = make_float2(cos(angle_2), sin(-angle_2));
    t1 = make_float2(a.x + (w_1.x * b.x - w_1.y * b.y),
                     a.y + (w_1.x * b.y + w_1.y * b.x));
    t2 = make_float2(a.x - (w_1.x * b.x - w_1.y * b.y),
                     a.y - (w_1.x * b.y + w_1.y * b.x));
    t3 = make_float2(c.x + (w_2.x * d.x - w_2.y * d.y),
                     c.y + (w_2.x * d.y + w_2.y * d.x));
    t4 = make_float2(c.x - (w_2.x * d.x - w_2.y * d.y),
                     c.y - (w_2.x * d.y + w_2.y * d.x));

    int return_base = padding + 4 * (tx % 2);
    input[return_base] = t1;
    input[return_base + 8] = t2;
    input[return_base + 2] = t3;
    input[return_base + 10] = t4;
    __syncthreads();

    // last stage
    // stride = 1
    a = input[tx * 4];
    b = input[tx * 4 + 1];
    c = input[tx * 4 + 2];
    d = input[tx * 4 + 3];
    angle_1 = 2.0f * M_PI / 16.0f * tx * 2;
    angle_2 = 2.0f * M_PI / 16.0f * (tx * 2 + 1);
    w_1 = make_float2(cos(angle_1), sin(-angle_1));
    w_2 = make_float2(cos(angle_2), sin(-angle_2));
    t1 = make_float2(a.x + (w_1.x * b.x - w_1.y * b.y),
                     a.y + (w_1.x * b.y + w_1.y * b.x));
    t2 = make_float2(a.x - (w_1.x * b.x - w_1.y * b.y),
                     a.y - (w_1.x * b.y + w_1.y * b.x));
    t3 = make_float2(c.x + (w_2.x * d.x - w_2.y * d.y),
                     c.y + (w_2.x * d.y + w_2.y * d.x));
    t4 = make_float2(c.x - (w_2.x * d.x - w_2.y * d.y),
                     c.y - (w_2.x * d.y + w_2.y * d.x));
    return_base = tx * 2;
    input[return_base] = t1;
    input[return_base + 8] = t2;
    input[return_base + 1] = t3;
    input[return_base + 9] = t4;
    __syncthreads();
    // Write back from shared memory to global memory
    out[tx] = input[tx];
    out[tx + 4] = input[tx + 4];
    out[tx + 8] = input[tx + 8];
    out[tx + 12] = input[tx + 12];
}

__global__ void FFT_4096(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    constexpr int THREADS = 256;
    __shared__ float input[8192 + 8192 / 32]; // padding to avoid bank conflict
    // Each thread stores
    //     its own part of the input sequence in its registers and uses
    //         shared memory as a communication buffer.
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    // int idx = tx;
    // int local_idx = tx;

    // why 2048 no bank conflict but 4096 has bank conflict?
    int ELEMENTS = N;

    // int padded_idx = local_idx + local_idx / 32;
    // Load from global memory to shared memory
    for (int i = 0; i < ELEMENTS; i += THREADS)
    {
        int idx = tx + i;
        int padded_idx = idx * 2 + idx * 2 / 32;
        // load from global, coalesced
        float2 c = data[idx + i];
        // store to shared memory, avoid bank conflict
        input[padded_idx] = c.x;
        input[padded_idx + 1] = c.y;
    }
    __syncthreads();

    // Write back from shared memory to global memory
    for (int i = 0; i < ELEMENTS; i += THREADS)
    {
        int idx = tx + i;
        int padded_idx = idx * 2 + idx / 16 * 2;
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

    auto start = std::chrono::high_resolution_clock::now();
    if (N == 16)
    {
        // vkfft setting: 1 block, 4 threads
        FFT_16<<<1, 4>>>(device_data, device_out, N);
    }
    else
    {
        dim3 dimBlock(256);
        dim3 dimGrid(N / 4096);
        FFT_4096<<<dimGrid, dimBlock>>>(device_data, device_out, N);
    }
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