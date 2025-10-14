# gPIE Profiling & Benchmarking

This directory contains benchmarking scripts and profiling analyses for evaluating the performance of **gPIE** on various computational imaging models.

---

## File Structure
```
profile/
├─ benchmark_utils.py # Timing and cProfile utilities
├─ benchmark_holography.py # Holography benchmark
├─ benchmark_random_cdi.py # Random CDI benchmark
├─ benchmark_coded_diffraction_pattern.py # CDP benchmark
├─ benchmark_ptychography.py # Ptychography benchmark (new)
└─ README.md # This file                               
```


---

## Target Models
We benchmarked four representative computational imaging models implemented in gPIE:

1. **Holography**  
   - The simplest model, involving two FFTs per iteration.
2. **Random Structured Matrix CDI**  
   - A CDI model using multiple phase masks and FFT layers.
3. **Coded Diffraction Pattern (CDP)**  
   - A multi-measurement model involving repeated FFTs with coded phase masks.
4. **Ptychography** *(new)*  
   - A patch-based inverse problem using overlapping probe positions, modeled by the `SlicePropagator` and `AccumulativeUncertainArray`.

---

## How to Run

### CPU (NumPy, default FFT):
```bash
python gpie/profile/benchmark_ptychography.py --backend numpy
```

---

##  How to Run
### CPU (NumPy, default FFT):
```bash
python gpie/profile/benchmark_holography.py --backend numpy
python gpie/profile/benchmark_random_cdi.py --backend numpy
python gpie/profile/benchmark_coded_diffraction_pattern.py --backend numpy
python gpie/profile/benchmark_ptychography.py --backend numpy
```

### CPU (NumPy + FFTW)
```bash
python gpie/profile/benchmark_holography.py --backend numpy --fftw
python gpie/profile/benchmark_holography.py --backend numpy --fftw --threads 4 --planner-effort FFTW_MEASURE
python gpie/profile/benchmark_holography.py --backend numpy --fftw --threads 8 --planner-effort FFTW_PATIENT
```
⚠️ Note: FFTW backend is only valid with --backend numpy. Using --backend cupy --fftw will raise an error.

### GPU (CuPy) Benchmark:
```bash
python gpie/profile/benchmark_holography.py --backend cupy
python gpie/profile/benchmark_random_cdi.py --backend cupy
python gpie/profile/benchmark_coded_diffraction_pattern.py --backend cupy
python gpie/profile/benchmark_ptychography.py --backend cupy
```

### Profiling(cProfile):
```bash
python gpie/profile/benchmark_holography.py --backend numpy --profile
python gpie/profile/benchmark_random_cdi.py --backend cupy --profile
```

## Benchmark Environment of the developer

- **OS:** Windows 11 Home, ver. 24H2, OS build 26100 4652
- **CPU:** Intel Core i7-14650K (16 cores, 24 threads)  
- **RAM:** 32 GB 
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)  
- **NVIDIA Driver:** 576.02
- **CUDA Toolkit:** 12.9
- **Python:** 3.10.5 (venv)  
- **Libraries:**
  - NumPy: 2.2.6
  - CuPy: 13.5.1

- **Note**: Results are device-dependent and may vary on different hardware or driver configurations.

##  Benchmark Results (512 x 512 pixels, 100 iterations)

| Model                  | NumPy (default FFT) | NumPy + FFTW (1 thread) | NumPy + FFTW (4 threads) | CuPy (GPU, 2nd run)|
|------------------------ |------------------- |------------------------ |--------------------------|--------------------|
| **Holography**          | 2.6 s              | 2.6 s                   | 2.4 s                    | 0.5 s             |
| **Random CDI**          | 6.2 s              | 6.1 s                   | 5.7 s                    | 0.8 s            |
| **CDP (4 measurements)** | 15.2 s            | 15.3 s                   | 14.2 s                    | 0.7 s              |
| **Ptychography (40 scans)** | 11.2 s            | 10.6 s                   | 10.6 s                    | 0.8 s              |

## Profiling Insights

In the Holography, Random CDI, and CDP benchmarks, the share of FFT
operations under the NumPy backend accounts for roughly **25--45%** of
total runtime. FFT is therefore the first natural target for
acceleration, but other costs quickly emerge as new bottlenecks.
In the ptychography benchmark, the portion of FFT in the runtime is below 15%.

The main competing sources of cost are the `UncertainArray.__truediv__`
operation and the Laplace-approximation--based message updates in
`AmplitudeMeasurement` nodes. Optimizing these components is a key
direction for future development.

On the GPU, the FFT share drops to about **10%**, effectively removing it
as a bottleneck. Excluding initialization overheads such as kernel
compilation and RNG setup, the remaining hotspots are the
Laplace-approximation updates in `AmplitudeMeasurement` and `UncertainArray.as_scalar_precision()`.

For example, in the Holography benchmark, `UA.__truediv__` accounts for
about **25%** of runtime with the NumPy backend but falls below **10%**
on CuPy (after the first run). Similar to FFT, elementwise operations
like `__mul__` and `__truediv__` benefit greatly from GPU acceleration,
while reductions such as `as_scalar_precision` (involving array
summations) tend to become the dominant cost.
