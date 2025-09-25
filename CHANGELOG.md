# Changelog

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
