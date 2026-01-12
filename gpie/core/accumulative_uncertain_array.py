from __future__ import annotations
from typing import Optional
from .backend import np, move_array_to_current_backend
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
        eps = real_dtype(1e-8)

        # initialize arrays
        self.weighted_data = np().zeros(event_shape, dtype=dtype)
        self.precision = np().full(event_shape, eps, dtype=real_dtype)

        # precompute coords for all patches
        self._coords_all, self._sizes, self._indices = self._precompute_coords(indices)
    
    def to_backend(self) -> None:
        """
        Move internal arrays to the current backend (NumPy or CuPy).

        This ensures that AUA remains consistent when switching between
        NumPy and CuPy backends via gpie.set_backend().
        """
        from .backend import np, move_array_to_current_backend
        real_dtype = get_real_dtype(self.dtype)

        # Move main arrays
        self.weighted_data = move_array_to_current_backend(self.weighted_data, dtype=self.dtype)
        self.precision = move_array_to_current_backend(self.precision, dtype=real_dtype)

        # Move cached coordinate arrays
        self._coords_all = move_array_to_current_backend(self._coords_all, dtype=int)


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

        scatter_add(self.precision, coords, flat_prec)
        scatter_add(self.weighted_data.real, coords, flat_weighted.real)
        scatter_add(self.weighted_data.imag, coords, flat_weighted.imag)


    # ------------------------------------------------------------
    # Internal helper: compute flat coord/value range for a block
    # ------------------------------------------------------------
    def _get_block_ranges(self, block: slice):
        """
        Compute flat coordinate ranges corresponding to a patch block.

        Args:
            block (slice): Patch index slice [start:stop)

        Returns:
            start_flat (int): Start index into self._coords_all
            stop_flat (int): Stop index into self._coords_all
        """
        start = block.start or 0
        stop = block.stop

        if start < 0 or stop > len(self._sizes) or start >= stop:
            raise ValueError(
                f"Invalid block slice {block} for {len(self._sizes)} patches."
            )

        # Prefix sum over patch sizes
        start_flat = sum(self._sizes[:start])
        stop_flat = sum(self._sizes[:stop])

        return start_flat, stop_flat


    # ------------------------------------------------------------
    # Block-aware scatter ADD
    # ------------------------------------------------------------
    def scatter_add_ua(self, ua, block: Optional[slice] = None):
        """
        Add (fuse) a batched UncertainArray into this AUA.

        Args:
            ua (UncertainArray):
                Batched UA with shape (batch_size, *patch_shape).
            block (slice or None):
                Patch index block to scatter.
                If None, full-batch scatter is performed.
        """
        if block is None:
            # Full-batch behavior (existing semantics)
            self.scatter_mul(ua)
            return

        # --------------------------------------------------------
        # Block-wise scatter
        # --------------------------------------------------------
        # Extract local block from UA
        ua_blk = ua.extract_block(block)

        # Flatten local precision and weighted data
        local_prec = ua_blk.precision(raw=False).reshape(ua_blk.batch_size, -1)
        local_weighted = (ua_blk.data * ua_blk.precision(raw=False)).reshape(
            ua_blk.batch_size, -1
        )

        # Compute flat ranges for this block
        start_flat, stop_flat = self._get_block_ranges(block)

        flat_prec = np().concatenate(
            [local_prec[b, :n]
             for b, n in enumerate(self._sizes[block.start:block.stop])],
            axis=0,
        )
        flat_weighted = np().concatenate(
            [local_weighted[b, :n]
             for b, n in enumerate(self._sizes[block.start:block.stop])],
            axis=0,
        )

        # Corresponding coordinates
        coords = tuple(self._coords_all[start_flat:stop_flat].T)

        # Scatter-add
        scatter_add(self.precision, coords, flat_prec)
        scatter_add(self.weighted_data.real, coords, flat_weighted.real)
        scatter_add(self.weighted_data.imag, coords, flat_weighted.imag)


    # ------------------------------------------------------------
    # Block-aware scatter SUB
    # ------------------------------------------------------------
    def scatter_sub_ua(
        self,
        ua,
        block: Optional[slice] = None,
    ):
        """
        Subtract a batched UncertainArray contribution from this AUA.

        Args:
            ua (UncertainArray):
                Batched UA with shape (batch_size, *patch_shape).
            block (slice or None):
                Patch index block to subtract.
            eps (float or None):
                Lower bound for precision after subtraction.
                If None, use dtype-dependent default.
        """

        if block is None:
            raise RuntimeError(
                "scatter_sub_ua(block=None) is not supported. "
                "For full-batch updates, clear the AUA and rebuild it instead."
            )
        else:
            # ----------------------------------------------------
            # Block-wise subtraction = add with negative values
            # ----------------------------------------------------
            ua_blk = ua.extract_block(block)

            local_prec = ua_blk.precision(raw=False).reshape(ua_blk.batch_size, -1)
            local_weighted = (ua_blk.data * ua_blk.precision(raw=False)).reshape(
                ua_blk.batch_size, -1
            )

            start_flat, stop_flat = self._get_block_ranges(block)

            flat_prec = np().concatenate(
                [-local_prec[b, :n]
                 for b, n in enumerate(self._sizes[block.start:block.stop])],
                axis=0,
            )
            flat_weighted = np().concatenate(
                [-local_weighted[b, :n]
                 for b, n in enumerate(self._sizes[block.start:block.stop])],
                axis=0,
            )

            coords = tuple(self._coords_all[start_flat:stop_flat].T)

            scatter_add(self.precision, coords, flat_prec)
            scatter_add(self.weighted_data.real, coords, flat_weighted.real)
            scatter_add(self.weighted_data.imag, coords, flat_weighted.imag)


    def as_uncertain_array(self):
        """
        Convert the full AUA into a standard UncertainArray (batched=False).

        Returns:
            UncertainArray: with data = weighted_data / precision.
        """
        from .uncertain_array import UncertainArray
        data = self.weighted_data / self.precision
        return UncertainArray(data, dtype=self.dtype, precision=self.precision, batched=False)


    def extract_patches(self, block: Optional[slice] = None):
        """
        Extract patches as a batched UncertainArray.

        Args:
            block (slice or None):
                Patch index slice [start:stop). If None, extract all patches.

        Returns:
            UncertainArray:
                Batched UA with shape (num_patches_in_block, *patch_shape).
        """
        from .uncertain_array import UncertainArray

        # Select patch indices
        if block is None:
            indices = self._indices
        else:
            start = block.start or 0
            stop = block.stop
            indices = self._indices[start:stop]

        if len(indices) == 0:
            raise ValueError("Empty block passed to extract_patches().")

        # Gather weighted data and precision for selected patches
        data_slices = [self.weighted_data[idx] for idx in indices]
        prec_slices = [self.precision[idx] for idx in indices]

        stacked_weighted = np().stack(data_slices, axis=0)
        stacked_prec = np().stack(prec_slices, axis=0)

        data = stacked_weighted / stacked_prec

        return UncertainArray(
            data,
            dtype=self.dtype,
            precision=stacked_prec,
            batched=True,
        )



    def clear(self):
        """
        Reset weighted_data and precision to zeros, keeping cached indices.

        This is useful when reusing the same AUA structure (event_shape + indices)
        for multiple forward/backward passes.
        """
        real_dtype = get_real_dtype(self.dtype)
        eps = real_dtype(1e-12)
        self.weighted_data[...] = 0
        self.precision[...] = eps
    
    def mul_ua(self, ua) -> None:
        """
        Multiply (fuse) a full-size UncertainArray into this AUA.

        Args:
            ua (UncertainArray): UA with batch_size=1 and event_shape == self.event_shape.
        """
        if ua.batch_size != 1:
            raise ValueError("mul_ua expects a UA with batch_size=1.")
        if ua.event_shape != self.event_shape:
            raise ValueError(
                f"event_shape mismatch: expected {self.event_shape}, got {ua.event_shape}"
            )
        if ua.dtype != self.dtype:
            raise TypeError(
                f"dtype mismatch: expected {self.dtype}, got {ua.dtype}"
            )

        prec = ua.precision(raw=False)[0]   # shape = event_shape
        data = ua.data[0]

        self.precision += prec
        self.weighted_data += data * prec


    def __repr__(self):
        return f"AUA(event_shape={self.event_shape}, patches={len(self._indices)}, dtype={self.dtype})"
