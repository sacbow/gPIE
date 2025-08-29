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

### 📌 Script Location

- **Path**: `examples/holography/holography.py`
- **Results**: `examples/holography/results/`

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

## 2. Coded Diffraction Pattern (CDP)

Coded Diffraction Pattern (CDP) is a phase retrieval model where a complex-valued image is recovered from multiple intensity-only observations obtained through different random phase masks. In gPIE, this model is constructed using multiple `PhaseMaskPropagator` and `AmplitudeMeasurement` factors connected through a shared latent `Wave` node.

### 📌 Script Location

- **Path**: `examples/coded_diffraction_pattern/coded_diffraction_pattern.py`
- **Results**: `examples/coded_diffraction_pattern/results/`

### 🖼️ Used Images

- `camera` (from `skimage.data`) → used as **amplitude**
- `moon` (from `skimage.data`) → used as **phase**

These are combined into a complex-valued image used as the ground truth sample for reconstruction.

### 🧪 Example Command

```bash
python examples/coded_diffraction_pattern/coded_diffraction_pattern.py \
  --n-iter 200 \
  --size 256 \
  --measurements 3 \
  --save-graph
```

### 💡 Command Line Options

| Option             | Description                                            |
|--------------------|--------------------------------------------------------|
| `--n-iter`         | Number of EP iterations                                |
| `--size`           | Image size (H = W)                                     |
| `--measurements`   | Number of phase masks used in CDP                      |
| `--save-graph`     | Save interactive factor graph as `graph.html`          |

### 💾 Output Files

| File Name                  | Description                             |
|----------------------------|-----------------------------------------|
| `true_amp.png`             | Ground truth amplitude                  |
| `true_phase.png`           | Ground truth phase                      |
| `reconstructed_amp.png`    | Reconstructed amplitude                 |
| `reconstructed_phase.png`  | Reconstructed phase                     |
| `convergence.png`          | PMSE over iterations                    |
| `graph.html`               | Graph structure visualization (optional) |


> 📖 **Reference**:  
> Candès, E. J., Li, X., & Soltanolkotabi, M. (2015).  
> *Phase retrieval from coded diffraction patterns*.  
> Applied and Computational Harmonic Analysis, 39(2), 277–299.  
> https://doi.org/10.1016/j.acha.2014.09.004

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