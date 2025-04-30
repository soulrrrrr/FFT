#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>
#include "io_utils.hpp"
#include "fft_gpu.hpp"
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

__device__ __forceinline__ void radix2(float2 &a, float2 &b)
{
    float2 t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;

    b.x = a.x - b.x;
    b.y = a.y - b.y;

    a = t;
}

__device__ __forceinline__ void radix2_w(float2 &a, float2 &b, float2 &w)
{
    float2 new_a, new_b;
    new_a.x = a.x + w.x * b.x - w.y * b.y;
    new_a.y = a.y + w.x * b.y + w.y * b.x;
    new_b.x = a.x - w.x * b.x + w.y * b.y;
    new_b.y = a.y - w.x * b.y - w.y * b.x;
    a = new_a;
    b = new_b;
}

__device__ __forceinline__ void radix4(
    float2 *input, float2 *output)
{
    // 4‐point DFT
    // X0 = a + b + c + d
    // X1 = a − b + c − d
    // X2 = a + b − c − d
    // X3 = a − b − c + d

    float2 a = input[0];
    float2 b = input[1];
    float2 c = input[2];
    float2 d = input[3];

    // stage 1: (a,c), (b,d) 各自做 2‐point butterfly
    float2 s0 = {a.x + c.x, a.y + c.y};
    float2 s1 = {b.x + d.x, b.y + d.y};
    float2 d0 = {a.x - c.x, a.y - c.y};
    float2 d1 = {b.x - d.x, b.y - d.y};

    // stage 2: 合併成真正的 4‐point DFT
    // X0 = s0 + s1
    output[0] = {s0.x + s1.x, s0.y + s1.y};
    // X1 = d0 − j·d1
    output[1] = {d0.x + d1.y, d0.y - d1.x};
    // X2 = s0 − s1
    output[2] = {s0.x - s1.x, s0.y - s1.y};
    // X3 = d0 + j·d1
    output[3] = {d0.x - d1.y, d0.y + d1.x};
}

__global__ void FFT_16(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    __shared__ float2 s[16];

    // each thread stores 4 elements
    float2 a, b, c, d;
    float2 t[4];
    int tx = threadIdx.x;
    for (int i = 0; i < 4; i++)
    {
        t[i] = data[tx + i * 4];
    }
    // a = data[tx];
    // b = data[tx + 4];
    // c = data[tx + 8];
    // d = data[tx + 12];
    radix4(t, t);
    // s[tx * 4 + 0] = a;
    // s[tx * 4 + 1] = b;
    // s[tx * 4 + 2] = c;
    // s[tx * 4 + 3] = d;

    // twiddle and transpose
    const float two_pi = -2.0f * M_PI / 16.0f;

    // 4.2 multiply twiddle W16^(row*col)
    auto twiddle = [&](float2 &v, int row)
    {
        float ang = two_pi * (row * tx);
        float co = cosf(ang), si = sinf(ang);
        v = make_float2(v.x * co - v.y * si,
                        v.x * si + v.y * co);
    };
    for (int i = 0; i < 4; i++)
    {
        twiddle(t[i], i);
    }
    // twiddle(a, 0);
    // twiddle(b, 1);
    // twiddle(c, 2);
    // twiddle(d, 3);

    // write twiddled back, do transpose
    for (int i = 0; i < 4; i++)
    {
        s[tx * 4 + i] = t[i];
    }
    __syncthreads();

    // load column
    for (int i = 0; i < 4; i++)
    {
        t[i] = s[tx + i * 4];
    }

    radix4(t, t);

    // 最後把結果散回一維 out[k]：k = p*4 + cidx
    for (int i = 0; i < 4; i++)
    {
        out[i * 4 + tx] = t[i];
    }
    // out[0 * 4 + tx] = a;
    // out[1 * 4 + tx] = b;
    // out[2 * 4 + tx] = c;
    // out[3 * 4 + tx] = d;
}

// each thread handles 2 elements
__global__ void FFT_N(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = N / 2;
    if (tx >= threads)
        return;
    extern __shared__ float2 s[];

    // init load
    s[tx] = data[tx];
    s[tx + threads] = data[tx + threads];
    __syncthreads();

    const int half = N >> 1;

    for (int stride = half; stride >= 1; stride >>= 1)
    {
        int block_size = stride << 1;
        int block_count = N / block_size;
        int block_idx = tx / stride;
        int block_offset = tx % stride;
        // load from shared memory
        float2 a = s[block_idx * block_size + block_offset];
        float2 b = s[block_idx * block_size + block_offset + stride];

        // calculate
        const float two_pi = -2.0f * M_PI / (N / stride);
        float ang = two_pi * block_idx;
        // w = (co, si), do a = a+wb, b = a-wb
        float2 w = make_float2(cosf(ang), sinf(ang));
        radix2_w(a, b, w);

        // store back to shared memory
        s[block_idx * stride + block_offset] = a;
        s[block_idx * stride + block_offset + half] = b;
        __syncthreads();
    }

    // final store
    out[tx] = s[tx];
    out[tx + threads] = s[tx + threads];
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

    // planning
    std::vector<int> plan;
    plan_fft(plan, N);
    int *device_plan;
    cudaMalloc(&device_plan, plan.size() * sizeof(int));
    cudaMemcpy(device_plan, plan.data(), plan.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = N / 2;
    // now max 2048 elements
    auto start = std::chrono::high_resolution_clock::now();
    FFT_N<<<1, threads, N * sizeof(float2)>>>(device_data, device_out, N);
    // FFT_16<<<1, 4>>>(device_data, device_out, N);
    // // kernel
    // if (N == 16)
    // {
    //     // vkfft setting: 1 block, 4 threads
    //     FFT_16<<<1, 4>>>(device_data, device_out, N);
    // }
    // else
    // {
    //     dim3 dimBlock(256);
    //     dim3 dimGrid(N / 4096);
    //     FFT_4096<<<dimGrid, dimBlock>>>(device_data, device_out, N);
    // }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back
    cudaMemcpy(host_data.data(), device_out, N * sizeof(float2), cudaMemcpyDeviceToHost);

    // Timing & write output
    std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << std::endl;

    std::string out_file = "data/output_fft_gpu_" + std::to_string(N) + ".txt";
    std::vector<std::complex<float>> result;
    result.reserve(N);
    for (const auto &f2 : host_data)
        result.emplace_back(f2.x, f2.y);
    write_complex_data(result, out_file);

    cudaFree(device_data);
    return 0;
}