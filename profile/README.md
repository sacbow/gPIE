# gPIE Profiling & Benchmarking

This directory contains benchmarking scripts and profiling analyses for evaluating the performance of **gPIE** on various computational imaging models.

---

## File Structure
```
profile/
 ├─ benchmark_utils.py                         # Timing and cProfile utilities
 ├─ benchmark_holography.py                    # Holography benchmark
 ├─ benchmark_random_cdi.py                    # Random CDI benchmark
 ├─ benchmark_coded_diffraction_pattern.py     # CDP benchmark
 └─ README.md                                  # This file
```

## Target Models
We benchmarked three representative computational imaging models implemented in gPIE:

1. **Holography**  
   - The simplest model, involving a single FFT per iteration.
2. **Random Structured Matrix CDI**  
   - A CDI model using multiple phase masks and FFT layers.
3. **Coded Diffraction Pattern (CDP)**  
   - A multi-measurement model involving repeated FFTs with coded phase masks.

---

##  How to Run
### CPU (NumPy) Benchmark:
```bash
python gpie/profile/benchmark_holography.py --backend numpy
python gpie/profile/benchmark_random_cdi.py --backend numpy
python gpie/profile/benchmark_coded_diffraction_pattern.py --backend numpy
```

### GPU (CuPy) Benchmark:
```bash
python gpie/profile/benchmark_holography.py --backend cupy
python gpie/profile/benchmark_random_cdi.py --backend cupy
python gpie/profile/benchmark_coded_diffraction_pattern.py --backend cupy
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

##  Benchmark Results (512×512 images, 100 iterations)

| Model                  | NumPy (CPU) | CuPy 1st Run (GPU) | CuPy 2nd Run (GPU) | Speedup (Stable) |
|------------------------ |------------ |------------------- |------------------- |----------------- |
| **Holography**         | 2.5 s       | 2.2 s             | 0.5 s             | ~5×             |
| **Random CDI**         | 5.8 s       | 2.5 s             | 0.8 s             | ~7×             |
| **CDP (4 measurements)** | 15.0 s     | 3.3 s             | 1.3 s             | ~10×             |

## Profiling insights

- **Numpy (CPU):**
    - FFT dominates the computational time.
    - EP message updates in Measurement objects and UncertainArray operations (e.g. \__truediv\__ method) form the next major cost.

- **Cupy (GPU):**
    - FFT bottleneck is effectively removed via GPU-acceleration.
    - The remaining costs shift to EP message updates and UncertainArray operations, along with Cupy array initialization overhead.