from typing import Optional, Dict
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.backend import np
from .base import Propagator


class ForkPropagator(Propagator):
    """
    Replicates a single input Wave (batch_size=1) into a batched output
    with `batch_size > 1`.

    Precision-mode policy:
        - Supported: SCALAR, ARRAY
        - Not supported: SCALAR_TO_ARRAY, ARRAY_TO_SCALAR


    Note on caching the previous input:
        We cache the reduced product of the message from output previous input message by self.child_product.
    """

    def __init__(self, batch_size: int, dtype: np().dtype = np().complex64):
        """
        This propagator replicates a single input wave (batch_size = 1)
        into a batched output wave with batch_size > 1.

        For block-wise (sequential) scheduling, this class maintains
        incremental caches of products of output-side messages in order
        to update the backward message efficiently without recomputing
        full batch products at every step.

        Args:
            batch_size:
                Target batch size of the output wave (must be >= 1).
            dtype:
                Complex dtype used for internal UncertainArray operations.
        """
        super().__init__(input_names=("input",), dtype=dtype)

        if batch_size < 1:
            raise ValueError("ForkPropagator requires batch_size >= 1.")

        self.batch_size = batch_size

        # ------------------------------------------------------------------
        # Incremental backward-message cache (for sequential scheduling)
        # ------------------------------------------------------------------

        # Global product of all child (output-side) messages.
        self.child_product: Optional[UA] = None

        # Cache of per-block products.
        self.product_cache: dict[tuple[int, int], UA] = {}

        # Cached block size used in the previous sequential run.
        self._cached_block_size: Optional[int] = None


    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect the input wave and create an output wave with replicated batch size.
        """
        if wave.batch_size != 1:
            raise ValueError("ForkPropagator only accepts input waves with batch_size=1.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = wave.dtype
        self.event_shape = wave.event_shape

        out_wave = Wave(
            event_shape=self.event_shape,
            batch_size=self.batch_size,
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    # -------- Precision mode handling --------
    def _set_precision_mode(self, mode: UnaryPropagatorPrecisionMode):
        """
        Restrict precision mode to SCALAR or ARRAY for fork.
        """
        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("ForkPropagator requires UnaryPropagatorPrecisionMode.")
        if mode not in (UnaryPropagatorPrecisionMode.SCALAR, UnaryPropagatorPrecisionMode.ARRAY):
            raise ValueError(f"Unsupported precision mode for ForkPropagator: {mode}")
        # Delegate to base consistency checks if needed
        super()._set_precision_mode(mode)

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif x_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif y_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    
    # -------- Utility for making block hashable --------
    def _normalize_block(self, block) -> tuple[int, int]:
        """
        Normalize a block specification to a hashable (start, stop) tuple.

        Args:
            block: slice or None

        Returns:
            (start, stop)
        """
        if block is None:
            return (0, self.batch_size)

        if not isinstance(block, slice):
            raise TypeError(f"block must be a slice or None, got {type(block)}")

        start = 0 if block.start is None else block.start
        stop = self.batch_size if block.stop is None else block.stop

        if start < 0 or stop > self.batch_size or start >= stop:
            raise ValueError(f"Invalid block slice: {block}")

        return (start, stop)


    # -------- Message passing --------
    def _compute_forward(self, inputs: Dict[str, UA], block=None) -> UA:
        # Normalize block
        block_key = self._normalize_block(block)
        start, stop = block_key
        block_size = stop - start

        out_msg = self.output_message

        # ------------------------------------------------------------
        # Handle block-size change
        # ------------------------------------------------------------
        if self._cached_block_size != block_size:
            self.product_cache = {}
            self.child_product = None
            self._cached_block_size = block_size

            if out_msg is not None:
                B = self.batch_size
                for s in range(0, B, block_size):
                    blk_key = (s, min(s + block_size, B))
                    blk_slice = slice(*blk_key)

                    out_blk = out_msg.extract_block(blk_slice)
                    blk_product = out_blk.product_reduce_over_batch()

                    self.product_cache[blk_key] = blk_product
                    self.child_product = (
                        blk_product if self.child_product is None
                        else self.child_product * blk_product
                    )

        # ------------------------------------------------------------
        # Input message (batch_size=1)
        # ------------------------------------------------------------
        m_in = inputs["input"]
        if m_in.batch_size != 1:
            raise ValueError("ForkPropagator expects input UA with batch_size=1.")

        belief = m_in if self.child_product is None else (m_in * self.child_product)

        belief_forked = belief.fork(batch_size=block_size)

        if out_msg is None:
            return belief_forked

        out_blk = out_msg.extract_block(slice(start, stop))
        return belief_forked / out_blk


    def backward(self, block=None) -> None:
        out_msg = self.output_message
        if out_msg is None:
            raise RuntimeError("Missing output message for backward.")

        block_key = self._normalize_block(block)
        start, stop = block_key
        block_size = stop - start

        if self._cached_block_size != block_size:
            raise RuntimeError(
                "ForkPropagator.backward() called with block_size "
                f"{block_size}, but cached_block_size is {self._cached_block_size}. "
                "Call forward() first to rebuild caches."
            )

        out_blk = out_msg.extract_block(slice(start, stop))
        blk_product = out_blk.product_reduce_over_batch()

        old = self.product_cache.get(block_key)

        if old is None:
            self.child_product = (
                blk_product if self.child_product is None
                else self.child_product * blk_product
            )
        else:
            self.child_product = (self.child_product * blk_product) / old

        self.product_cache[block_key] = blk_product

        # Send backward message (batch_size=1)
        x_wave = self.inputs["input"]
        x_wave.receive_message(self, self.child_product)

    
    def get_sample_for_output(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        forked_sample = np().broadcast_to(x, (self.batch_size,) + x_wave.event_shape).copy()
        return forked_sample