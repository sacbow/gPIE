# Changelog

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
