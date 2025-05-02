# FFT project

## worktime
4/7 1600-1700 1900-2030
found naive code cannot vectorize
- std::complex is black box
- std::polar() uses sin/cos
- out[k] += is cumulative
- memory aliasing (data[n] and out[k])
Using 34 * N^2 FLOPS (reason in analysis/metrics.py)
N=16      Time=0.000023s  FLOPS=376.71 MFLOPS
N=32      Time=0.000044s  FLOPS=799.42 MFLOPS
N=64      Time=0.000140s  FLOPS=995.32 MFLOPS
N=128     Time=0.000509s  FLOPS=1093.41 MFLOPS
N=256     Time=0.002307s  FLOPS=965.84 MFLOPS
N=512     Time=0.008209s  FLOPS=1085.79 MFLOPS
N=1024    Time=0.033122s  FLOPS=1076.36 MFLOPS
N=2048    Time=0.132390s  FLOPS=1077.17 MFLOPS
N=4096    Time=0.464395s  FLOPS=1228.32 MFLOPS
N=8192    Time=1.762852s  FLOPS=1294.32 MFLOPS
N=16384   Time=6.842768s  FLOPS=1333.79 MFLOPS
 
4/8 2300-2330
- naive cooley-tukey (use 5N*log2(N))
N=16      Time=0.000021s  FLOPS=15.35 MFLOPS
N=32      Time=0.000022s  FLOPS=36.40 MFLOPS
N=64      Time=0.000026s  FLOPS=74.97 MFLOPS
N=128     Time=0.000039s  FLOPS=113.97 MFLOPS
N=256     Time=0.000044s  FLOPS=230.69 MFLOPS
N=512     Time=0.000051s  FLOPS=454.23 MFLOPS
N=1024    Time=0.000082s  FLOPS=627.76 MFLOPS
N=2048    Time=0.000145s  FLOPS=779.21 MFLOPS
N=4096    Time=0.000285s  FLOPS=862.12 MFLOPS
N=8192    Time=0.000570s  FLOPS=933.88 MFLOPS
N=16384   Time=0.001123s  FLOPS=1021.47 MFLOPS
N=32768   Time=0.002429s  FLOPS=1011.86 MFLOPS
N=65536   Time=0.005221s  FLOPS=1004.25 MFLOPS
N=131072  Time=0.010114s  FLOPS=1101.56 MFLOPS
N=262144  Time=0.020347s  FLOPS=1159.51 MFLOPS

4/9 1000-1100
- observations
    - -O3 inlined `bit_reverse` function
        - ```
            2f66:	4c 89 f9             	mov    %r15,%rcx      ; rcx = x
            2f69:	31 d2                	xor    %edx,%edx       ; edx = i = 0
            2f6b:	31 c0                	xor    %eax,%eax       ; eax = result = 0

            ...loop...

            2f80:	48 89 ce             	mov    %rcx,%rsi       ; s = x
            2f83:	48 01 c0             	add    %rax,%rax       ; result <<= 1
            2f86:	ff c2                	inc    %edx            ; i++
            2f88:	48 d1 e9             	shr    $1,%rcx         ; x >>= 1
            2f8b:	83 e6 01             	and    $0x1,%esi       ; x & 1
            2f8e:	48 09 f0             	or     %rsi,%rax       ; result |= (x & 1)
            2f91:	39 d7                	cmp    %edx,%edi        ; i < log2n?
            2f93:	75 eb                	jne    2f80             ; loop
            ```
- Work
    - use `perf record ./bin/fftw ../data/input_131072.txt` found FFTW uses `n1fv_128`
    - look into fftw source code
    - improvements:
        - add  -march=native
        - asm shows avx2 commands eg. `vfmadd231ss %xmm3,%xmm10,%xmm0`
        - ```
            N=16      Time=0.000020s  FLOPS=15.61 MFLOPS
            N=32      Time=0.000018s  FLOPS=43.27 MFLOPS
            N=64      Time=0.000025s  FLOPS=75.77 MFLOPS
            N=128     Time=0.000022s  FLOPS=204.06 MFLOPS
            N=256     Time=0.000027s  FLOPS=376.38 MFLOPS
            N=512     Time=0.000039s  FLOPS=595.63 MFLOPS
            N=1024    Time=0.000063s  FLOPS=807.46 MFLOPS
            N=2048    Time=0.000108s  FLOPS=1039.02 MFLOPS
            N=4096    Time=0.000224s  FLOPS=1098.13 MFLOPS
            N=8192    Time=0.000463s  FLOPS=1149.42 MFLOPS
            N=16384   Time=0.000896s  FLOPS=1280.69 MFLOPS
            N=32768   Time=0.001935s  FLOPS=1269.78 MFLOPS
            N=65536   Time=0.003860s  FLOPS=1358.32 MFLOPS
            N=131072  Time=0.007472s  FLOPS=1491.06 MFLOPS
            N=262144  Time=0.016484s  FLOPS=1431.27 MFLOPS
            ```
    - need to work on
        - use pre calculated cos/sin value
        - use avx commands

### 4/9 1930-2130
- identified bottleneck
    - not using avx2 (%ymm)
    - complex multiplication (w * b)
    - bit_reverse (about 15% of execution time)
- maybe read FFTW paper to see how to improve or using other algorithm?
    - explore Stockham Auto-Sort FFT
        - no need bit_reverse

### 4/10 2230-0100
- Read [FFTW Paper](https://www.fftw.org/fftw-paper-ieee.pdf)
- Learn [Stockham Algorithm](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)
- Still doesn't know how to write stockham algorithm

## 4/11 1030-1100, 1400-1600
- Stockham OK, but need to know how it calculates
- [Reference](https://github.com/scientificgo/fft/blob/master/stockham.go)
- Self-implemented, use result to find the pattern, but I think I didn't understand the crux
- ```
    N=16      Time=0.000014s  FLOPS=23.58 MFLOPS
    N=32      Time=0.000011s  FLOPS=72.04 MFLOPS
    N=64      Time=0.000014s  FLOPS=137.26 MFLOPS
    N=128     Time=0.000013s  FLOPS=337.40 MFLOPS
    N=256     Time=0.000017s  FLOPS=612.30 MFLOPS
    N=512     Time=0.000020s  FLOPS=1171.24 MFLOPS
    N=1024    Time=0.000029s  FLOPS=1767.54 MFLOPS
    N=2048    Time=0.000051s  FLOPS=2217.33 MFLOPS
    N=4096    Time=0.000105s  FLOPS=2343.78 MFLOPS
    N=8192    Time=0.000204s  FLOPS=2608.96 MFLOPS
    N=16384   Time=0.000391s  FLOPS=2932.28 MFLOPS
    N=32768   Time=0.000842s  FLOPS=2919.95 MFLOPS
    N=65536   Time=0.001565s  FLOPS=3350.91 MFLOPS
    N=131072  Time=0.003119s  FLOPS=3571.77 MFLOPS
    N=262144  Time=0.006395s  FLOPS=3689.41 MFLOPS
    ```
- How FFTW plan?
    - 2, . . . , 16, 32, 64 : direct plan (hard-coded)
    - n = r*m: radix-r Cooley-Tukey
- improvements
    - perf found complex resolve spent the most time (~ 50%)
    - self complex type and mm256?
    - explore radix 4 algorithm?

## 4/14 1400-1600
- Stockham recursive ver. OK, performance similar
- Try manual complex, complex ops uses ymm
- ```
    N=16      Time=0.000014s  FLOPS=22.10 MFLOPS
    N=32      Time=0.000014s  FLOPS=57.45 MFLOPS
    N=64      Time=0.000015s  FLOPS=128.09 MFLOPS
    N=128     Time=0.000022s  FLOPS=200.73 MFLOPS
    N=256     Time=0.000026s  FLOPS=393.11 MFLOPS
    N=512     Time=0.000029s  FLOPS=797.76 MFLOPS
    N=1024    Time=0.000033s  FLOPS=1544.98 MFLOPS
    N=2048    Time=0.000047s  FLOPS=2406.33 MFLOPS
    N=4096    Time=0.000086s  FLOPS=2868.44 MFLOPS
    N=8192    Time=0.000164s  FLOPS=3254.14 MFLOPS
    N=16384   Time=0.000318s  FLOPS=3609.30 MFLOPS
    N=32768   Time=0.000678s  FLOPS=3626.12 MFLOPS
    N=65536   Time=0.001347s  FLOPS=3890.84 MFLOPS
    N=131072  Time=0.002581s  FLOPS=4316.80 MFLOPS
    N=262144  Time=0.005040s  FLOPS=4681.23 MFLOPS
    ```

## 4/25 1200-2000
- Install vkfft, init fft_gpu
- VkFFT time:
- ```
    N=16      Time=0.000053s  FLOPS=6.06 MFLOPS
    N=32      Time=0.000054s  FLOPS=14.85 MFLOPS
    N=64      Time=0.000052s  FLOPS=36.98 MFLOPS
    N=128     Time=0.000053s  FLOPS=84.94 MFLOPS
    N=256     Time=0.000055s  FLOPS=187.72 MFLOPS
    N=512     Time=0.000052s  FLOPS=445.91 MFLOPS
    N=1024    Time=0.000051s  FLOPS=1008.53 MFLOPS
    N=2048    Time=0.000054s  FLOPS=2103.39 MFLOPS
    N=4096    Time=0.000063s  FLOPS=3897.68 MFLOPS
    ```
- My Time:
- ```
    N=16      Time=0.000572s  FLOPS=0.56 MFLOPS
    N=32      Time=0.000338s  FLOPS=2.37 MFLOPS
    N=64      Time=0.000368s  FLOPS=5.22 MFLOPS
    N=128     Time=0.000337s  FLOPS=13.29 MFLOPS
    N=256     Time=0.000333s  FLOPS=30.74 MFLOPS
    N=512     Time=0.000350s  FLOPS=65.83 MFLOPS
    N=1024    Time=0.000348s  FLOPS=147.17 MFLOPS
    N=2048    Time=0.000356s  FLOPS=316.75 MFLOPS
    N=4096    Time=0.000468s  FLOPS=525.36 MFLOPS
    ```
- Improvements:
    - Change to Stockham

## 4/27 1000-1500, 2100-2200
- Hand written FFT_16
- Try use more register
- my implementation ncu profile 8.64 us (but cpp profiling still 1/10 speed N=16      Time=0.000514s  FLOPS=0.62 MFLOPS)
    - vkfft 3.33us
- improvement
    - it's time to implement radix-n fft
    - maybe start with squared? I don't know
    - move focus to mixed-radix FFT
- Radix-4
    - same memory access pattern
    - but time 6.72us

## 4/29 1300-1700
- still doesn't figure out how to use 3d fft to write 1d
- feel stuck
    - implementing more radix kernel?
    - use radix 4 (radix 2) to implement all kernel using cooley-tukey
    - or deal with always 2d?
    - implementing stockham on gpu first

# 4/30 1300-1500
- Finished stockham radix-2
- on 2048: 20.32 us / comparing with VkFFt 14.43us

# 5/2 1000-1300
- on 2048 no bank: 17.44us
- the calculation is significant more than VkFFt
    - my:    4912 fused and 2848 non-fused FP32
    - vkfft: 776 fused and 2128 non-fused FP32



# To set up Python environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt