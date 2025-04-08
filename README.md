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

# To set up Python environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt