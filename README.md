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

# To set up Python environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt