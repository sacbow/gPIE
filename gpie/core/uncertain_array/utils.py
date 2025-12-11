from .base import UncertainArray
from ..linalg_utils import reduce_precision_to_scalar
from ..types import get_real_dtype
from ..backend import np
from ..fft import get_fft_backend

def get_slice(self : UncertainArray, idx) -> UncertainArray:
    """
    Return a sliced view of the UncertainArray along event dimensions.

    This method behaves similarly to NumPy slicing, but only applies
    to the event dimensions (batch dimension is fixed at 1).

    Note:
        For extracting *many* patches in one call, use
        `UncertainArray.extract_patches` for better performance.

    Args:
        idx (slice or tuple of slices):
            Slice object(s) specifying which portion of the event_shape
            to extract. Do not include the batch dimension.

    Returns:
        UncertainArray:
            A new UncertainArray containing the sliced region.
    """

    if self.batch_size != 1:
        raise ValueError("__getitem__ expects batch_size=1 UncertainArray as input.")

    if not isinstance(idx, tuple):
        idx = (idx,)

    sliced_data = self.data[(0,) + idx]        # pick batch 0 explicitly
    sliced_prec = self.precision(raw=False)[(0,) + idx]

    # reshape back to (1, *sliced_event_shape)
    sliced_data = sliced_data.reshape((1,) + sliced_data.shape)
    sliced_prec = sliced_prec.reshape((1,) + sliced_prec.shape)

    return UncertainArray(sliced_data, dtype=self.dtype, precision=sliced_prec)


def zero_pad(self : UncertainArray, pad_width: tuple[tuple[int, int], ...]) -> UncertainArray:
    """
    Apply zero-padding to the UncertainArray along event dimensions.

    Data in the padded regions are set to 0, and precision in those regions
    is set to a very large number (≈1e8) with the same real dtype as the UA,
    representing deterministic zeros.
    """
    pad_full = ((0, 0),) + pad_width
    padded_data = np().pad(self.data, pad_full, mode="constant", constant_values=0)

    # select real dtype (float32 for complex64, etc.)
    real_dtype = get_real_dtype(self.dtype)
    large_prec = real_dtype(1e8)  # precision scalar in correct dtype

    padded_prec = np().pad(
            self.precision(raw=False),
            pad_full,
            mode="constant",
            constant_values=large_prec,
        )

    return UncertainArray(padded_data, dtype=self.dtype, precision=padded_prec)


def extract_patches(self : UncertainArray, indices: list[tuple]) -> UncertainArray:
    """
    Extract multiple patches (slices) from the UncertainArray and
    aggregate them into a new batched UncertainArray.

    Each index in `indices` corresponds to a slice on the event_shape.
    The resulting UncertainArray has batch_size = len(indices).

    Args:
        indices (list[tuple]):
            List of slice tuples specifying patches to extract.
            Each tuple should be compatible with the event_shape.

    Returns:
        UncertainArray:
            A new UncertainArray with batch_size = len(indices), where
            each batch entry corresponds to one extracted patch.
    """
    if self.batch_size != 1:
        raise ValueError("extract_patches() expects batch_size=1 UncertainArray as input.")

    data_slices = [self.data[(0,) + idx] for idx in indices]
    stacked_data = np().stack(data_slices, axis=0)

    prec_slices = [self.precision(raw=False)[(0,) + idx] for idx in indices]
    stacked_prec = np().stack(prec_slices, axis=0)

    return UncertainArray(stacked_data, dtype=self.dtype, precision=stacked_prec)


def as_scalar_precision(self: UncertainArray) -> UncertainArray:
    """
    Convert to scalar precision mode (batch-wise harmonic reduction).
    """
    if self._scalar_precision:
        return self

    scalar_prec = reduce_precision_to_scalar(self.precision(raw=True))
    return UncertainArray(self.data, dtype=self.dtype, precision=scalar_prec)

    
def as_array_precision(self: UncertainArray) -> UncertainArray:
    """
    Convert this UncertainArray to array precision mode.

    Promotes scalar precision to full per-element array.
    """
    if not self._scalar_precision:
        return self

    # Extract scalar value (guaranteed to be broadcastable constant)
    raw_prec = self.precision(raw=True)

    scalar_value = raw_prec.reshape((self.batch_size,) + (1,) * len(self.event_shape))

    array_precision = np().full(self.data.shape, scalar_value, dtype=get_real_dtype(self.dtype))

    return UncertainArray(self.data, dtype=self.dtype, precision=array_precision)

    
    
def fft2_centered(self: UncertainArray) -> UncertainArray:
    """
    Apply centered 2D FFT to the UncertainArray, assuming EP-style Gaussian message.

    This operation applies `ifftshift → fft2 → fftshift` to the data.
    The resulting UncertainArray always uses **scalar precision**:
    - If the input is already scalar precision: precision remains unchanged
    - If the input is array precision: precision is reduced using harmonic mean

    Returns:
        UncertainArray: FFT-transformed UA with scalar precision.
    """
    fft_backend = get_fft_backend()
    transformed_data = fft_backend.fft2_centered(self.data)

    if self._scalar_precision:
        new_precision = self.precision(raw=True)
    else:
        new_precision = reduce_precision_to_scalar(self.precision(raw=True))

    return UncertainArray(transformed_data, dtype=self.dtype, precision=new_precision)

def ifft2_centered(self: UncertainArray) -> UncertainArray:
    """
    Apply centered 2D inverse FFT to the UncertainArray.

    This operation applies `ifftshift → ifft2 → fftshift` to the data.
    The resulting UncertainArray always uses **scalar precision**:
    - If the input is already scalar precision: precision remains unchanged
    - If the input is array precision: precision is reduced using harmonic mean

    Returns:
        UncertainArray: Inverse FFT-transformed UA with scalar precision.
    """
    fft_backend = get_fft_backend()
    transformed_data = fft_backend.ifft2_centered(self.data)

    if self._scalar_precision:
        new_precision = self.precision(raw=True)
    else:
        new_precision = reduce_precision_to_scalar(self.precision(raw=True))

    return UncertainArray(transformed_data, dtype=self.dtype, precision=new_precision)


def extract_block(self: UncertainArray, block: slice) -> UncertainArray:
    """
    Extract a contiguous slice along the batch dimension and return it
    as a new UncertainArray.

    Args:
        block (slice):
            Slice along the batch dimension, e.g., slice(i, j).

    Returns:
        UncertainArray:
            A new UA with batch_size = (block.stop - block.start)
            and the same event_shape, dtype, and precision_mode.
    """
    if block is None:
        return self
    if not isinstance(block, slice):
        raise TypeError("block must be a slice object.")

    start, stop = block.start, block.stop
    if start < 0 or stop > self.batch_size or start >= stop:
        raise ValueError(
            f"Invalid block slice {block} for batch_size={self.batch_size}."
        )

    # Extract mean data
    data_sub = self.data[start:stop].copy()

    # Extract precision (broadcasted form)
    prec_sub = self.precision(raw=True)[start:stop].copy()

    return UncertainArray(
        array=data_sub,
        dtype=self.dtype,
        precision=prec_sub,
        batched=True,
    )



def insert_block(self: UncertainArray, block: slice, sub: UncertainArray) -> None:
    """
    Insert (overwrite) a batch slice with another UncertainArray.

    If block is None:
        → full-batch overwrite (sub must have identical batch_size and metadata)

    If block is a slice:
        → overwrite self.data[start:stop] with sub.data

    Common constraints (checked first):
        - event_shape must match
        - dtype must match
        - precision_mode must match
    """

    # ------------------------------------------------------------
    # 1. Metadata compatibility checks (common to full/slice updates)
    # ------------------------------------------------------------
    if self.event_shape != sub.event_shape:
        raise ValueError("Event shape mismatch in insert_block().")

    if self.dtype != sub.dtype:
        raise TypeError("dtype mismatch in insert_block().")

    if self.precision_mode != sub.precision_mode:
        raise ValueError("precision_mode mismatch in insert_block().")

    # ------------------------------------------------------------
    # 2. Full-batch overwrite (block=None)
    # ------------------------------------------------------------
    if block is None:
        if self.batch_size != sub.batch_size:
            raise ValueError(
                f"Full overwrite requested but batch_size mismatch: "
                f"{self.batch_size} vs {sub.batch_size}"
            )

        # overwrite mean
        self.data[...] = sub.data

        # overwrite precision (raw)
        raw_self = self.precision(raw=True)
        raw_sub = sub.precision(raw=True)
        raw_self[...] = raw_sub
        return

    # ------------------------------------------------------------
    # 3. Slice overwrite
    # ------------------------------------------------------------
    if not isinstance(block, slice):
        raise TypeError("block must be a slice or None.")

    start, stop = block.start, block.stop
    if start < 0 or stop > self.batch_size or start >= stop:
        raise ValueError(
            f"Invalid block slice {block} for batch_size={self.batch_size}."
        )

    block_size = stop - start
    if block_size != sub.batch_size:
        raise ValueError(
            f"sub.batch_size={sub.batch_size} does not match block size {block_size}."
        )

    # overwrite mean
    self.data[start:stop] = sub.data

    # overwrite precision (raw)
    raw_self = self.precision(raw=True)
    raw_sub = sub.precision(raw=True)
    raw_self[start:stop] = raw_sub



# --- monkey patch ---
UncertainArray.extract_patches = extract_patches
UncertainArray.zero_pad = zero_pad
UncertainArray.__getitem__ = get_slice
UncertainArray.fft2_centered = fft2_centered
UncertainArray.ifft2_centered = ifft2_centered
UncertainArray.as_scalar_precision = as_scalar_precision
UncertainArray.as_array_precision = as_array_precision
UncertainArray.extract_block = extract_block
UncertainArray.insert_block = insert_block
