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

// complex ops
__device__ __forceinline__ float2 complex_add(const float2 &a, const float2 &b)
{
    return {a.x + b.x, a.y + b.y};
}

__device__ __forceinline__ float2 complex_sub(const float2 &a, const float2 &b)
{
    return {a.x - b.x, a.y - b.y};
}

__device__ __forceinline__ float2 complex_mul(const float2 &a, const float2 &b)
{
    // (a.x + i a.y) * (b.x + i b.y)
    return {
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x};
}

__device__ __forceinline__ void radix2(float2 &a, float2 &b)
{
    float2 t;
    t.x = a.x + b.x;
    t.y = a.y + b.y;

    b.x = a.x - b.x;
    b.y = a.y - b.y;

    a = t;
}

__device__ __forceinline__ void radix2_w(float2 &a, float2 &b, const float2 &w)
{
    float2 new_a, new_b;
    new_a.x = a.x + w.x * b.x - w.y * b.y;
    new_a.y = a.y + w.x * b.y + w.y * b.x;
    new_b.x = a.x - w.x * b.x + w.y * b.y;
    new_b.y = a.y - w.x * b.y - w.y * b.x;
    a = new_a;
    b = new_b;
}

__device__ __forceinline__ void radix4_w(
    float2 &a, float2 &b, float2 &c, float2 &d, const float2 &w)
{
    // 4‐point DFT
    // w0 = 1
    // w1 = w
    const float2 w2 = complex_mul(w, w);  // w2 = w^2
    const float2 w3 = complex_mul(w, w2); // w3 = w^3

    // reference: https://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html

    // stage 1: 2nd matrix * input
    // float2 t0 = a + c;
    // float2 t1 = a - c;
    // float2 t2 = b + d;
    // float2 t3 = b - d;
    b = complex_mul(w, b);
    c = complex_mul(w2, c);
    d = complex_mul(w3, d);

    float2 t0 = complex_add(a, c);
    float2 t1 = complex_sub(a, c);
    float2 t2 = complex_add(b, d);
    float2 t3 = complex_sub(b, d);

    // stage 2: 1st matrix * stage 1
    // a = t0 + t2
    // b = t1 - j * t3
    // c = t0 - t2
    // d = t1 + j * t3
    // j = (0, 1)
    float2 jt3 = {-t3.y, t3.x}; // j * t3
    a = complex_add(t0, t2);
    b = complex_sub(t1, jt3);
    c = complex_sub(t0, t2);
    d = complex_add(t1, jt3);
}

// each thread handles 2 elements
__global__ void FFT_N_2(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
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
        // int block_count = N / block_size;
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

// each thread handles 4 elements
__global__ void FFT_N_4(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    const int quarter = N >> 2;
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = N >> 2;
    if (tx >= threads)
        return;
    extern __shared__ float2 s[];

// init load
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int idx = tx + i * threads;
        s[idx] = data[idx];
    }

    for (int stride = quarter; stride >= 1; stride >>= 2)
    {
        __syncthreads();
        const int block_size = stride << 2;
        // int block_count = N / block_size;
        const int block_idx = tx / stride;
        const int block_offset = tx % stride;
        // load from shared memory
        const int base_idx = block_idx * block_size + block_offset;
        float2 a = s[base_idx];
        float2 b = s[base_idx + stride];
        float2 c = s[base_idx + stride * 2];
        float2 d = s[base_idx + stride * 3];
        __syncthreads();

        // calculate
        const float two_pi = -2.0f * M_PI / (N / stride);
        const float ang = two_pi * block_idx;
        const float2 w = make_float2(cosf(ang), sinf(ang));
        radix4_w(a, b, c, d, w);

        // store back to shared memory
        const int base_stride_idx = block_idx * stride + block_offset;
        s[base_stride_idx] = a;
        s[base_stride_idx + quarter] = b;
        s[base_stride_idx + quarter * 2] = c;
        s[base_stride_idx + quarter * 3] = d;
    }
    __syncthreads();

// final store
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int idx = tx + i * threads;
        out[idx] = s[idx];
    }
}

// all 4 but last is 2
__global__ void FFT_N_4_last_2(float2 *__restrict__ data, float2 *__restrict__ out, size_t N)
{
    const int half = N >> 1;
    const int quarter = N >> 2;
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = N >> 2;
    if (tx >= threads)
        return;
    extern __shared__ float2 s[];

// init load
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int idx = tx + i * threads;
        s[idx] = data[idx];
    }

    for (int stride = quarter; stride >= 1; stride >>= 2)
    {
        __syncthreads();
        int block_size = stride << 2;
        // int block_count = N / block_size;
        int block_idx = tx / stride;
        int block_offset = tx % stride;
        // load from shared memory
        int base_idx = block_idx * block_size + block_offset;
        float2 a = s[base_idx];
        float2 b = s[base_idx + stride];
        float2 c = s[base_idx + stride * 2];
        float2 d = s[base_idx + stride * 3];
        __syncthreads();

        // calculate
        const float two_pi = -2.0f * M_PI / (N / stride);
        float ang = two_pi * block_idx;
        // float ang2 = two_pi * (block_idx * 2);
        // float ang3 = two_pi * (block_idx * 3);
        // w = (co, si), do a = a+wb, b = a-wb
        float2 w = make_float2(cosf(ang), sinf(ang));
        // float2 w2 = make_float2(cosf(ang2), sinf(ang2));
        // float2 w3 = make_float2(cosf(ang3), sinf(ang3));
        // radix4_w(a, w1 * b, w2 * c, w3 * d);
        // b = {b.x * w1.x - b.y * w1.y, b.x * w1.y + b.y * w1.x};
        // c = {c.x * w2.x - c.y * w2.y, c.x * w2.y + c.y * w2.x};
        // d = {d.x * w3.x - d.y * w3.y, d.x * w3.y + d.y * w3.x};
        radix4_w(a, b, c, d, w);

        // store back to shared memory
        int base_stride_idx = block_idx * stride + block_offset;
        s[base_stride_idx] = a;
        s[base_stride_idx + quarter] = b;
        s[base_stride_idx + quarter * 2] = c;
        s[base_stride_idx + quarter * 3] = d;
    }
    __syncthreads();

    // do 2 radix2
    int stride = 1;
    int block_size = 2;
    // int block_count = N / block_size;
    int block_idx_0 = 2 * tx;
    int block_idx_1 = block_idx_0 + 1;
    int block_offset = tx % stride;
    // load from shared memory
    float2 a = s[block_idx_0 * block_size + block_offset];
    float2 b = s[block_idx_0 * block_size + block_offset + stride];
    float2 c = s[block_idx_1 * block_size + block_offset];
    float2 d = s[block_idx_1 * block_size + block_offset + stride];

    // calculate
    const float two_pi = -2.0f * M_PI / N;
    float ang_b = two_pi * block_idx_0;
    float ang_d = two_pi * block_idx_1;
    // w = (co, si), do a = a+wb, b = a-wb
    float2 w_b = make_float2(cosf(ang_b), sinf(ang_b));
    float2 w_d = make_float2(cosf(ang_d), sinf(ang_d));
    radix2_w(a, b, w_b);
    radix2_w(c, d, w_d);

    // store back to global memory
    out[block_idx_0 * stride + block_offset] = a;
    out[block_idx_0 * stride + block_offset + half] = b;
    out[block_idx_1 * stride + block_offset] = c;
    out[block_idx_1 * stride + block_offset + half] = d;

    // __syncthreads();
    // final store
    // #pragma unroll
    // for (int i = 0; i < 4; i++)
    // {
    //     int idx = tx + i * threads;
    //     out[idx] = s[idx];
    // }
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
    // std::vector<int> plan;
    // plan_fft(plan, N);
    // int *device_plan;
    // cudaMalloc(&device_plan, plan.size() * sizeof(int));
    // cudaMemcpy(device_plan, plan.data(), plan.size() * sizeof(int), cudaMemcpyHostToDevice);

    auto isPowerOf4 = [](int n)
    {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555);
    };
    int threads;
    int shared_mem_size = (N) * sizeof(float2);
    // now max 2048 elements
    std::chrono::duration<float> diff;
    // radix
    constexpr int RADIX = 4;
    if (RADIX == 2)
    {
        threads = N / 2;
        auto start = std::chrono::high_resolution_clock::now();
        FFT_N_2<<<1, threads, shared_mem_size>>>(device_data, device_out, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        diff = end - start;
    }
    else if (RADIX == 4)
    {
        threads = N / 4;
        if (isPowerOf4(N))
        {
            auto start = std::chrono::high_resolution_clock::now();
            FFT_N_4<<<1, threads, shared_mem_size>>>(device_data, device_out, N);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            diff = end - start;
        }
        else
        {
            auto start = std::chrono::high_resolution_clock::now();
            FFT_N_4_last_2<<<1, threads, shared_mem_size>>>(device_data, device_out, N);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            diff = end - start;
        }
    }
    else
    {
        std::cerr << "Unsupported radix" << std::endl;
        return 1;
    }

    // Copy result back
    cudaMemcpy(host_data.data(), device_out, N * sizeof(float2), cudaMemcpyDeviceToHost);

    // Timing & write output
    // std::chrono::duration<float> diff = end - start;
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