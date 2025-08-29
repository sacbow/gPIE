# Examples

This directory contains example experiments using the **gPIE** framework.

Each subdirectory corresponds to a different inverse problem or imaging model, implemented with expectation propagation (EP) on a factor graph.

---

## 1. Holography

This experiment demonstrates inline holography using a factor graph with the following components:

- **Object**: Complex-valued image (either synthetic or loaded from `skimage.data`)
- **Reference wave**: Predefined or loaded complex field
- **Forward model**: Inline interference → Fourier transform
- **Measurement**: Amplitude-only detector with additive Gaussian noise

The corresponding experiment script is:

```
examples/holography/holography.py
```

### 🔧 Command-line options

```bash
python examples/holography/holography.py --obj-img cameraman --ref-img moon --obj-radius 0.15 --ref-radius 0.2 --n-iter 100 --save-graph
```

| Option         | Description                                         |
|----------------|-----------------------------------------------------|
| `--obj-img`    | Object image name from `skimage.data`              |
| `--ref-img`    | Reference wave image (optional)                    |
| `--obj-radius` | Support mask radius for object (if not using image)|
| `--ref-radius` | Support mask radius for reference wave             |
| `--n-iter`     | Number of EP iterations                            |
| `--save-graph` | Save factor graph visualization as HTML            |

### 💾 Outputs (saved to `examples/holography/results/`)

- `true_amp.png`, `true_phase.png`: Ground truth object
- `reconstructed_amp.png`, `reconstructed_phase.png`: Reconstructed object
- `ref_amp.png`, `ref_phase.png`: Reference wave
- `convergence.png`: PSE convergence plot
- `graph.html`: Interactive factor graph rendered with Bokeh

---

## 2. Coded Diffraction Pattern

**[To be completed]**  
Will demonstrate the EP reconstruction of CDP datasets using random phase masks.

---

## 3. Random Structured CDI

**[To be completed]**  
Demonstrates inference in CDI models with randomized illumination and structured sparsity priors.

---

## 4. Compressed Sensing

**[To be completed]**  
Linear measurement with sparsity prior (e.g., wavelet), supporting both real and complex signal recovery.

---

## 📁 Data

Sample images are automatically downloaded and cached into:

```
examples/sample_data/
```

via `skimage.data`.  
No external datasets are required.

---

## 🔧 Notes

- All scripts assume NumPy backend.
- GPU acceleration with CuPy can be enabled in future benchmarks.
- Figures are saved using `matplotlib.pyplot.imsave()`.

---