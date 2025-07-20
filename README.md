# gPIE: Graph-based Probabilistic Inference Engine

**gPIE** is a modular, extensible Python framework for structured probabilistic inference via message passing. It supports expectation propagation (EP) over factor graphs, including support for complex-valued variables and various types of priors, propagators, and measurements.

## Project Structure
```
gpie/
│
├── core/ # Core data structures: UncertainArray, types, etc.
│ ├── uncertain_array.py
│ ├── uncertain_array_tensor.py
│ ├── types.py
│ ├── linalg_utils.py
│ └── metrics.py
│
├── graph/ # Computational factor graph logic
│ ├── wave.py # Latent variables
│ ├── factor.py # Abstract factor base class
│ ├── shortcuts.py # High-level operator shortcuts (e.g., fft2, ifft2)
│ │
│ ├── prior/ # Prior factor nodes
│ ├── propagator/ # Propagation factors (linear, FFT, nonlinear, etc.)
│ └── measurement/ # Observation models (Gaussian, amplitude, etc.)
│
├── graph/structure/
│ └── graph.py # Graph scheduling, message passing, and sampling
│
├── test/ # Example notebooks and validation scripts
│
└── setup.py # Optional setup file for packaging
```

## Features

- **UncertainArray abstraction** for representing complex Gaussian distributions
- Batch support via `UncertainArrayTensor`
- Flexible **factor graph construction** with directional message scheduling
- Modular components:
  - Priors (e.g., Gaussian, sparse, support-based)
  - Unary and binary propagators (e.g., FFT2D, phase modulation, multiplication)
  - Measurement models (e.g., Gaussian, amplitude-based)
- Built-in **sampling**, **message damping**, and **precision control** (scalar/array)
- Visual graph inspection via `graph.visualize()`


## 🛠️ Installation

This project has been tested on **Python 3.10.5**.

### ✅ Required Dependencies

| Package   | Version | Purpose                                   |
|-----------|---------|-------------------------------------------|
| `numpy`   | ≥2.2.6  | Core tensor computation and math support  |

### ⚙️ Optional Dependencies

| Package   | Version | Used In             | Description                             |
|-----------|---------|---------------------|-----------------------------------------|
| `bokeh`   | ≥3.7.3  | `graph.visualize()` | Visualization of computational graphs   |

Note: `bokeh` may install additional transitive dependencies such as `Jinja2`, `tornado`, etc., automatically via pip.

---

### 📦 Install with pip

Minimum setup (core functionality only):

```bash
pip install numpy>=2.2.6
```

Optional: for graph visualization via `Graph.visualize()`:

```bash
pip install bokeh>=3.7.3
```

---

###  Development Setup

Clone and install the repository in editable mode:

```bash
git clone https://github.com/your-org/gpie.git
cd gpie
pip install -e .
```

This will allow you to make changes to the source code without reinstalling the package.

##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

