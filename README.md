# gPIE: Graph-based Probabilistic Inference Engine

**gPIE** is a modular, extensible Python framework for structured probabilistic inference via **Expectation Propagation (EP)** on factor graphs.
It provides built-in support for complex-valued variables, NumPy/CuPy backend switching, and specialized components for computational imaging models.

## Project Structure
```
gpie/
├── gpie/ # Core package (importable as gpie)
│ ├── __init__.py
│ ├── core/ # Core data structures and utilities
│ │ ├── uncertain_array.py
│ │ ├── types.py
│ │ ├── linalg_utils.py
│ │ ├── rng_utils.py
│ │ └── metrics.py
│ │
│ ├── graph/ # Factor graph and EP components
│ │ ├── wave.py
│ │ ├── factor.py
│ │ ├── shortcuts.py
│ │ ├── prior/ # Priors: Gaussian, sparse, etc.
│ │ ├── propagator/ # Binary/unary propagators: FFT, masks, etc.
│ │ ├── measurement/ # Likelihood models: amplitude, Gaussian, etc.
│ │ └── structure/ # Graph connectivity
│ │         └── graph.py
│ └── Others

├── examples/ # Example scripts and notebooks
│ ├── io_utils.py
│ ├── sample_data/ # Output directory for example images
│ ├── notebooks/ # Jupyter notebooks for tutorials
│ └── scripts/ # Example Python scripts

├── profile/ # Profiling & benchmarking scripts

├── test/ # Unit tests (pytest-based)

├── pyproject.toml # Build configuration (PEP 621)
├── requirements.txt # Dependency pinning for development
├── README.md
└── LICENSE


```

## Features
- **Expectation Propagation (EP)** on factor graphs.
- **UncertainArray abstraction** for representing complex Gaussian distributions
- NumPy/CuPy backend support: switch seamlessly between CPU and GPU with:
```bash
  import numpy as np, cupy as cp, gpie
  gpie.set_backend(cp)  # or np
```
- Flexible **factor graph construction** with directional message scheduling
- Modular components:
  - Priors (e.g., Gaussian, sparse, support-based)
  - Unary and binary propagators (e.g., FFT2D, phase modulation, multiplication)
  - Measurement models (e.g., Gaussian, amplitude-based)

- Built-in **sampling** and **expectation propagation** based on topological sort
- Visual graph inspection via `graph.visualize()`

## What's New (v0.1)

- **Drastically Simplified Model Definition**
Define complex probabilistic models with minimal code using the new @model decorator.
For example, a full random structured CDI model is just:
```bash
  @model
  def random_structured_cdi(support, phase_masks, noise_var):
      x = ~SupportPrior(support=support, label="sample", dtype=np.complex64)
      for phase_mask in phase_masks:
          x = fft2(phase_mask * x)
      AmplitudeMeasurement(var=noise_var, damping=0.3) << x
```

- **Flexible and High-Quality Graph Visualization**
gPIE now supports full visual inspection of factor graphs with:

> Layout engines: Choose between networkx (fast, layoutable) and graphviz (high-quality, canonical).

> Rendering backends: Visualize with either matplotlib (static) or bokeh (interactive HTML).

Usage:
```bash
  graph.visualize(layout="graphviz", backend="bokeh")
```


## Tutorials & Notebooks
A set of demonstration notebooks is available under:
``
examples/notebooks/
``

Each notebook corresponds to a different inverse problem or imaging model:

- `holography_demo.ipynb`
- `coded_diffraction_pattern_demo.ipynb`
- `random_structured_cdi_demo.ipynb`
- `compressed_sensing_demo.ipynb`

These illustrate the use of gPIE for EP-based inference on realistic synthetic data.


## Benchmarks & profiling
- GPU acceleration via CuPy
- Profiling utilities (profile/) include:
```bash
  python profile/benchmark_holography.py --backend cupy --profile
  python profile/benchmark_coded_diffraction_pattern.py --backend numpy
```
See [profile/README.md](./profile/README.md) for detailed results and profiling insights.

##  Installation

This project has been tested on **Python 3.10.5**.

---

##  Dependencies

### Core Dependencies
| Package      | Version   | Purpose                        |
|--------------|-----------|--------------------------------|
| `numpy`      | ≥2.2.6    | Core tensor computation (CPU backend) |

###  Optional (for GPU and visualization)
| Package        | Version     | Used for                          |
|----------------|-------------|-----------------------------------|
| `cupy`         | ≥13.5.0     | GPU backend acceleration          |
| `matplotlib`   | ≥3.10.5     | Static visualization    |
| `bokeh`        | ≥3.7.3      | Interactive visualization    |
| `networkx`     | ≥3.3        | Graph structure layouting          |
| `pygraphviz`   | ≥1.10       |  Graph structure layouting        |
| `graphviz`     | system pkg  | Required by `pygraphviz` (native) |

> **Notes:**
> - To use **CuPy**, ensure that your environment has a supported CUDA toolkit version installed.
> - `pygraphviz` requires [Graphviz](https://graphviz.org/) to be **installed separately**.

---

### 📦 Install with pip

**Minimum setup (core functionality only):**

```bash
pip install -e .
```

###  Development Setup

Clone and install the repository in editable mode:

```bash
git clone https://github.com/sacbow/gpie.git
cd gpie
pip install -e .
```

This will allow you to make changes to the source code without reinstalling the package.

## Running Tests

This project uses `pytest` for unit testing. To run the full test suite:

```bash
pytest test/ --cov=gpie --cov-report=term-missing
```

As of the latest release, the test coverage is approximately 87%, covering both CPU and GPU (CuPy) backends.

##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.


## Contact
For questions, please open an issue or contact:
- Hajime Ueda (ueda@mns.k.u-tokyo.ac.jp)

