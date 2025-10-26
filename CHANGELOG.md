# Changelog

## [v0.2.1] — 2025-10-26

### Added
- New demo: `examples/scripts/blind_ptychography_with_phase.py`  
  Demonstrates bilinear inference (object × probe) under phase observation.

### Improved
- `MultiplyPropagator`: introduced inner-loop variational updates for more stable VMP convergence.

### Notes
This release improves numerical stability for bilinear models and adds a reproducible example for hybrid EP/VMP inference.



## [0.2.0] - 2025-10-10
### Added
- Ptychography support via new factor-graph modules:

  - PtychographyDataset: unified container for object, probe, and diffraction data.

  - SlicePropagator + AccumulativeUncertainArray: A syntax to describe the physical model of ptychography.

- Example script:
  - examples/notebook/ptychography_demo.ipynb — An introduction to ptychographic phase retrieval via gPIE

  - examples/scripts/ptychography/ptychography.py — complete forward and reconstruction workflow.

### Note
- Ptychographic reconstruction via gPIE is seen as an implementation of the Ptycho-EP algorithm proposed in our paper **Ueda, H., Katakami, S., & Okada, M. (2025). A Message-Passing Perspective on Ptychographic Phase Retrieval** on [Arxiv](https://arxiv.org/abs/2504.05668).


## [0.1.2] - 2025-10-01
### Added
- New propagators:
  - **ForkPropagator**: replicate input waves across batch dimension.
  - **SlicePropagator**: extract fixed-size patches (for ptychography, etc.).
  - **ZeroPadPropagator**: apply zero-padding.
- `Wave` class convenience methods:
  - `Wave.extract_patches()` and `Wave.__getitem__` for intuitive slicing.
  - `Wave.zero_pad()` for easy zero-padding in models.


### Changed
- Improved coverage of `Wave` error handling, message passing, and sampling logic.
- Refactored tests for slice and zero-padding propagators to ensure >90% coverage.

### Notes
- Reported test coverage may differ between environments:
  - On local machines with **CuPy, pygraphviz, and bokeh** installed, both all backends are tested, yielding >91% coverage.
  - On CI environments without CuPy, coverage may appear slightly lower.



## [0.1.1] - 2025-09-25
### Added
- CuPy backend now uses `CuPyFFTBackend` with plan caching (faster FFT on GPU).
- Benchmark scripts now support `--fftw`, `--threads`, and `--planner-effort` options.
- Profiling insights added for Holography, Random CDI, and CDP (1024×1024 scale).

### Changed
- `set_backend()` now automatically chooses the right FFT backend (NumPy → DefaultFFT, CuPy → CuPyFFT).
- Default FFTW planner effort changed from `FFTW_MEASURE` to `FFTW_ESTIMATE` for faster startup.

### Fixed
- Minor docstring improvements.
- Profiling README clarified.


---

## [0.1.0] - 2025-09-21
### Added
- `@model` syntax: drastically simplified model description.

### Changed
- `UncertainArray` now retains a batch of arrays.
- `UncertainArrayTensor` class removed (merged into `UncertainArray`).
- Refactored dtype management in Measurement classes.

---

## [0.0.0] - 2025-09-01
### Added
- Initial public release.
