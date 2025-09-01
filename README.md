# gPIE: Graph-based Probabilistic Inference Engine

**gPIE** is a modular, extensible Python framework for structured probabilistic inference via **Expectation Propagation (EP)** on factor graphs.
It provides built-in support for complex-valued variables, NumPy/CuPy backend switching, and specialized components for computational imaging models.

## Project Structure
```
 gpie/
 │
 ├── core/               # Core data structures and utilities
 │ ├── uncertain_array.py
 │ ├── uncertain_array_tensor.py
 │ ├── types.py
 │ ├── linalg_utils.py
 │ ├── rng_utils.py       # Random number utilities
 │ └── metrics.py
 │
 ├── graph/              # Factor graph and EP message passing
 │ ├── wave.py
 │ ├── factor.py
 │ ├── shortcuts.py
 │ ├── prior/
 │ ├── propagator/
 │ ├── measurement/
 │ ├── structure/
 │ │ └── graph.py
 │
+├── examples/            # Example notebooks and demos (e.g., holography/CDI/CDP)
+├── profile/             # Benchmarking and profiling scripts
 ├── test/                # Unit tests (pytest-based)
-└── setup.py             # Legacy setup file (consider pyproject.toml)

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

## Tutorials & Notebooks
A set of demonstration notebooks is available under:
``
examples/notebooks/
``

Each notebook corresponds to a different inverse problem or imaging model:

- `holography_demo.ipynb`
- `cdp_demo.ipynb`
- `random_structured_cdi_demo.ipynb`
- `compressed_sensing_demo.ipynb`

These illustrate the use of gPIE for EP-based inference on realistic synthetic data.


## Benchmarks & profiling
- GPU acceleration via CuPy
- Profiling utilities (profile/) include:
```bash
  python gpie/profile/benchmark_holography.py --backend cupy --profile
  python gpie/profile/benchmark_coded_diffraction_pattern.py --backend numpy
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
| `bokeh`        | ≥3.7.3      | Interactive graph visualization    |
| `networkx`     | ≥3.3        | Graph structure modeling          |
| `pygraphviz`   | ≥1.10       | Graphviz-based visualization       |
| `graphviz`     | system pkg  | Required by `pygraphviz` (native) |

> **Notes:**
> - To use **CuPy**, ensure that your environment has a supported CUDA toolkit version installed.
> - `pygraphviz` requires [Graphviz](https://graphviz.org/) to be **installed separately**.

---

### 📦 Install with pip

**Minimum setup (core functionality only):**

```bash
pip install numpy>=2.2.6
```

###  Development Setup

Clone and install the repository in editable mode:

```bash
git clone https://github.com/sacbow/gpie.git
cd gpie
pip install -e .
```

This will allow you to make changes to the source code without reinstalling the package.

##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.


## Contact
For questions, please open an issue or contact:
- Hajime Ueda (ueda@mns.k.u-tokyo.ac.jp)

