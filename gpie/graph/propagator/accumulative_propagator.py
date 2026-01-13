from abc import ABC, abstractmethod
from typing import Optional
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA


class AccumulativePropagator(Propagator, ABC):
    """
    Base class for propagators that maintain an internal accumulated state
    and support speculative block-wise updates.

    This class provides:
        - Speculative cache lifecycle
        - Block normalization
        - Next-block speculation (contiguous scan)
        - A unified control flow for block-wise backward with miss fallback
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._spec_block: Optional[slice] = None
        self._spec_old_block: Optional[UA] = None

    # -------------------------
    # Speculative cache helpers
    # -------------------------
    def _clear_spec_cache(self) -> None:
        self._spec_block = None
        self._spec_old_block = None

    def _normalize_block(self, block: slice) -> slice:
        if not isinstance(block, slice):
            raise TypeError(f"block must be a slice, got {type(block)}")

        start = 0 if block.start is None else block.start
        stop = block.stop
        if stop is None:
            raise ValueError("block.stop must not be None for block-wise mode.")

        if start < 0 or start >= stop or stop > self.batch_size:
            raise ValueError(f"Invalid block slice {block} for batch_size={self.batch_size}.")

        return slice(start, stop)

    def _speculate_next_block(self, current_block: slice) -> slice:
        """
        Contiguous next-block speculation under BlockGenerator semantics.
        """
        start = 0 if current_block.start is None else current_block.start
        stop = current_block.stop
        if stop is None:
            raise ValueError("current_block.stop must not be None.")

        bsz = stop - start
        if bsz <= 0:
            raise ValueError(f"Invalid block size inferred from {current_block}.")

        next_start = stop if stop < self.batch_size else 0
        next_stop = min(next_start + bsz, self.batch_size)
        return slice(next_start, next_stop)

    # -------------------------
    # Hooks for subclasses
    # -------------------------
    @abstractmethod
    def _rebuild_accumulator_from_output(self, out_msg: UA) -> None:
        """
        Rebuild internal accumulated state from the full output message.
        """
        ...

    @abstractmethod
    def _apply_incremental_update(self, new_blk: UA, old_blk: UA, blk: slice) -> None:
        """
        Apply incremental update to internal accumulated state for the given global block.
        """
        ...

    # -------------------------
    # Shared block-wise backward core
    # -------------------------
    def _backward_with_speculation(self, out_msg: UA, blk: slice) -> None:
        """
        Shared control flow:
            - try incremental if cache hit
            - else rebuild
            - then snapshot next block
        """
        did_incremental = (
            self._spec_block is not None
            and self._spec_old_block is not None
            and (self._spec_block.start, self._spec_block.stop) == (blk.start, blk.stop)
        )

        if did_incremental:
            new_blk = out_msg.extract_block(blk)
            old_blk = self._spec_old_block
            self._apply_incremental_update(new_blk, old_blk, blk)
        else:
            self._rebuild_accumulator_from_output(out_msg)

        next_blk = self._speculate_next_block(blk)
        self._spec_block = next_blk
        self._spec_old_block = out_msg.extract_block(next_blk).copy()
