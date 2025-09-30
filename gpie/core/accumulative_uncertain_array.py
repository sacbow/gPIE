from __future__ import annotations
from typing import Optional
from .backend import np
from .types import get_real_dtype
from .linalg_utils import scatter_add


class AccumulativeUncertainArray:
    """
    AccumulativeUncertainArray (AUA)

    A lightweight data structure for accumulating Gaussian messages efficiently.
    Instead of storing means directly, it stores:
        - weighted_data = data * precision
        - precision

    This makes Gaussian fusion (multiplication of PDFs) reduce to
    elementwise addition of `weighted_data` and `precision`.

    Typical usage:
        - Constructed once for a fixed list of patch indices.
        - scatter_mul(): accumulates a batched UA into the AUA
          at the pre-defined patch positions.
        - extract_patches(): retrieves the accumulated content
          at each patch position as a batched UncertainArray.
        - as_uncertain_array(): converts the whole AUA into a single UA
          (batched=False) with data = weighted_data / precision.

    Notes:
        - Designed for use inside Propagators (e.g., SlicePropagator).
        - Always batched=False internally (shape = (1, *event_shape)).
        - Supports overlapping patches (values are accumulated).
    """

    def __init__(self, event_shape, indices, dtype=np().complex64):
        self.event_shape = event_shape
        self.dtype = dtype
        real_dtype = get_real_dtype(dtype)

        # initialize arrays
        self.weighted_data = np().zeros((1,) + event_shape, dtype=dtype)
        self.precision = np().zeros((1,) + event_shape, dtype=real_dtype)

        # precompute coords for all patches
        self._coords_all, self._sizes, self._indices = self._precompute_coords(indices)

    def _precompute_coords(self, indices):
        coords_all = []
        sizes = []
        for idx in indices:
            grids = np().meshgrid(*[np().arange(s.start, s.stop) for s in idx], indexing="ij")
            coords = np().stack(grids, axis=-1).reshape(-1, len(idx))
            coords_all.append(coords)
            sizes.append(coords.shape[0])
        return np().concatenate(coords_all, axis=0), sizes, indices

    def scatter_mul(self, ua):
        """
        Multiply (fuse) a batched UncertainArray into this AUA at pre-defined patch positions.

        Args:
            ua (UncertainArray): Batched UA with shape (batch_size, *patch_shape).
                                 Must have batch_size = len(self._indices).
        """
        local_prec = ua.precision(raw=False).reshape(ua.batch_size, -1)
        local_weighted = (ua.data * ua.precision(raw=False)).reshape(ua.batch_size, -1)

        flat_prec = np().concatenate(
            [local_prec[b, :n] for b, n in enumerate(self._sizes)], axis=0
        )
        flat_weighted = np().concatenate(
            [local_weighted[b, :n] for b, n in enumerate(self._sizes)], axis=0
        )

        coords = tuple(self._coords_all.T)

        scatter_add(self.precision[0], coords, flat_prec)
        scatter_add(self.weighted_data[0].real, coords, flat_weighted.real)
        scatter_add(self.weighted_data[0].imag, coords, flat_weighted.imag)

    def as_uncertain_array(self):
        """
        Convert the full AUA into a standard UncertainArray (batched=False).

        Returns:
            UncertainArray: with data = weighted_data / precision.
        """
        from .uncertain_array import UncertainArray

        eps = get_real_dtype(self.dtype)(1e-8)
        precision_safe = np().maximum(self.precision, eps)
        data = self.weighted_data / precision_safe

        return UncertainArray(data, dtype=self.dtype, precision=self.precision, batched=False)


    def extract_patches(self):
        """
        Extract all patches (given by the original indices) as a batched UncertainArray.

        Returns:
            UncertainArray: with shape (num_patches, *patch_shape).
                            Each batch entry corresponds to one patch.
        """
        from .uncertain_array import UncertainArray

        data_slices = [self.weighted_data[(0,) + idx] for idx in self._indices]
        prec_slices = [self.precision[(0,) + idx] for idx in self._indices]

        stacked_weighted = np().stack(data_slices, axis=0)
        stacked_prec = np().stack(prec_slices, axis=0)

        eps = get_real_dtype(self.dtype)(1e-8)
        precision_safe = np().maximum(stacked_prec, eps)
        data = stacked_weighted / precision_safe

        return UncertainArray(data, dtype=self.dtype, precision=stacked_prec, batched=True)



    def initialize_from_ua(self, ua) -> None:
        """
        Initialize weighted_data and precision from a given UncertainArray.

        This reuses the precomputed indices cache (_coords_all, _sizes, _indices),
        while resetting the content of the AUA.

        Args:
            ua (UncertainArray): Must have batch_size=1 and event_shape matching self.event_shape.
        """
        if ua.batch_size != 1:
            raise ValueError("initialize_from_ua expects a UA with batch_size=1.")
        if ua.event_shape != self.event_shape:
            raise ValueError(
                f"event_shape mismatch: expected {self.event_shape}, got {ua.event_shape}"
            )
        if ua.dtype != self.dtype:
            raise TypeError(
                f"dtype mismatch: expected {self.dtype}, got {ua.dtype}"
            )

        # overwrite with new arrays
        self.weighted_data = ua.data[0] * ua.precision(raw=True)[0]
        self.precision = ua.precision(raw=False)[0]


    def __repr__(self):
        return f"AUA(event_shape={self.event_shape}, patches={len(self._indices)}, dtype={self.dtype})"
