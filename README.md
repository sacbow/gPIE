# gPIE: Graph-based Probabilistic Inference Engine

[![Tests](https://github.com/sacbow/gPIE/actions/workflows/tests.yml/badge.svg)](https://github.com/sacbow/gPIE/actions/workflows/tests.yml)

[![codecov](https://codecov.io/gh/sacbow/gPIE/graph/badge.svg?token=OVKYM0YQZ4)](https://codecov.io/gh/sacbow/gPIE)

**gPIE** is a modular, extensible Python framework for structured probabilistic inference via **Expectation Propagation (EP)** on factor graphs.
It provides built-in support for complex-valued variables, NumPy/CuPy backend switching, and specialized components for computational imaging models.

## Project Structure
```
gpie/
├── gpie/  # Core package (importable as gpie)
│  ├── __init__.py
│  ├── core/                         # Core data structures and utils
│  │  ├── uncertain_array/           # UA base + ops
│  │  ├── accumulative_uncertain_array.py
│  │  ├── backend.py
│  │  ├── linalg_utils.py
│  │  ├── rng_utils.py
│  │  ├── fft.py
│  │  └── metrics.py
│  │
│  ├── graph/                        # Factor graph + EP components
│  │  ├── wave.py
│  │  ├── factor.py
│  │  ├── shortcuts.py
│  │  ├── prior/                     # Gaussian, sparse, support priors
│  │  ├── propagator/                # Unary/Binary propagators
│  │  │  ├── fft_2d_propagator.py
│  │  │  ├── ifft_2d_propagator.py
│  │  │  ├── phase_mask_fft_propagator.py
│  │  │  ├── fork_propagator.py
│  │  │  ├── slice_propagator.py
│  │  │  └── zero_pad_propagator.py
│  │  ├── measurement/               # Amplitude, Gaussian, ...
│  │  └── structure/                 # Graph + model DSL + visualization
│  │     ├── graph.py
│  │     ├── model.py
│  │     └── visualization.py
│  │
│  └── imaging/
│     └── ptychography/
│        ├── data/
│        │  ├── dataset.py           # PtychographyDataset
│        │  └── diffraction_data.py
│        ├── simulator/
│        │  ├── forward.py           # ptychography_graph (sim)
│        │  ├── probe.py             # generate_probe
│        │  └── scan.py              # generate_fermat_spiral_positions
│        └── utils/
│           ├── geometry.py
│           └── visualization.py     # show/plot helpers
│
├── examples/
│  ├── io_utils.py
│  ├── sample_data/
│  ├── notebooks/
│  │  ├── holography_demo.ipynb
│  │  ├── coded_diffraction_pattern_demo.ipynb
│  │  ├── random_structured_cdi_demo.ipynb
│  │  ├── compressed_sensing_demo.ipynb
│  │  └── ptychography_demo.ipynb
│  └── scripts/
│     ├── holography/
│     ├── coded_diffraction_pattern/
│     ├── random_structured_cdi/
│     ├── compressed_sensing/
│     └── ptychography/
│        └── ptychography_demo.py
│
├── profile/
├── test/
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE

```

## Features
- **Expectation Propagation (EP)** on factor graphs.
- **UncertainArray abstraction** for representing complex/real Gaussian distributions
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

## What's New
See [CHANGELOG.md](./CHANGELOG.md) for full release notes.

**v1.2.0**: Added **ptychography** support — dataset container, scan simulation, patch-based forward model (SlicePropagator + AUA), and a complete reconstruction demo script/notebook.



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


## Related libraries

gPIE shares common ground with several existing frameworks for message passing inference:

#### [ForneyLab (Julia)](https://biaslab.github.io/project/forneylab/)
A declarative probabilistic programming framework based on factor graphs.

- **Strength**: strong abstraction of inference algorithms via `MessageUpdateRule`.
- **Difference**: gPIE instead provides a declarative Python API using only operator overloading, without macros or custom syntax.


#### [Tree-AMP (Python)](https://sphinxteam.github.io/tramp.docs/0.1/html/index.html)
A framework for Expectation Propagation algorithms, built on top of networkx.

- **Strength**: well-suited for graph manipulations.  
  It also provides theoretical tools such as **state evolution** and **free entropy formalisms**.
- **Difference**: gPIE is not tied to networkx and emphasizes a lightweight declarative DSL with GPU acceleration (CuPy backend).

#### [Dimple (Java/Matlab)](https://github.com/analog-garage/dimple)
A factor-graph based system with support for hardware acceleration via GP5.

- **Strength**: early pioneer in message passing DSLs with HPC focus.
- **Difference**: gPIE targets GPU acceleration in Python, making it more accessible to computational imaging researchers.

#### [Infer.NET (C#)](https://dotnet.github.io/infer/)
A probabilistic programming framework developed at Microsoft, supporting Expectation Propagation, Variational Message Passing, and Gibbs Sampling.

- **Strength**: industrial-grade implementation with a rich set of inference algorithms. Widely used in academia and industry.  
- **Difference**: unlike gPIE, Infer.NET is implemented in C# and not designed for GPU acceleration or scientific Python workflows.



##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.


## Contact
For questions, please open an issue or contact:
- Hajime Ueda (ueda@mns.k.u-tokyo.ac.jp)

