from __future__ import annotations

from typing import Optional, Dict

from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ..wave import Wave
from .accumulative_propagator import AccumulativePropagator


class ForkPropagator(AccumulativePropagator):
    """
    Replicate a single input Wave (batch_size=1) into a batched output Wave.

    This propagator is the "batch fork" counterpart of SlicePropagator:
        - Forward: replicate belief to output batch and send EP residuals.
        - Backward: accumulate (multiply) output-side messages across batch.

    Block-wise scheduling:
        - Uses speculative caching implemented by AccumulativePropagator.
        - Full rebuild on cache miss via product_reduce_over_batch().

    Precision-mode policy:
        - Supported: SCALAR, ARRAY
        - Not supported: SCALAR_TO_ARRAY, ARRAY_TO_SCALAR
    """

    def __init__(self, batch_size: int, dtype: np().dtype = np().complex64):
        """
        Args:
            batch_size: Target batch size of the output wave (must be >= 1).
            dtype: Complex dtype used for internal UncertainArray operations.
        """
        super().__init__(input_names=("input",), precision_mode=None)

        if batch_size < 1:
            raise ValueError("ForkPropagator requires batch_size >= 1.")

        self.batch_size = int(batch_size)
        self.dtype = dtype

        # Shape metadata (filled in __matmul__)
        self.event_shape: Optional[tuple[int, ...]] = None

        # Accumulator over output-side messages (batch-reduced product)
        self.output_product: Optional[UA] = None

        # Current internal precision-mode selection (derived from connected waves)
        self._precision_mode: Optional[UnaryPropagatorPrecisionMode] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
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
            dtype=self.dtype,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    # ------------------------------------------------------------------
    # Precision mode handling
    # ------------------------------------------------------------------
    def _set_precision_mode(self, mode: UnaryPropagatorPrecisionMode) -> None:
        """
        Restrict precision mode to SCALAR or ARRAY for fork.
        """
        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("ForkPropagator requires UnaryPropagatorPrecisionMode.")
        if mode not in (UnaryPropagatorPrecisionMode.SCALAR, UnaryPropagatorPrecisionMode.ARRAY):
            raise ValueError(f"Unsupported precision mode for ForkPropagator: {mode}")

        # Enforce consistency (no conflicts)
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing={self._precision_mode}, requested={mode}"
            )
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self) -> None:
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif x_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def set_precision_mode_backward(self) -> None:
        y_wave = self.output
        if y_wave is None:
            return
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif y_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    # ------------------------------------------------------------------
    # AccumulativePropagator hooks
    # ------------------------------------------------------------------
    def _rebuild_accumulator_from_output(self, out_msg: UA) -> None:
        """
        Full rebuild: product over the entire output batch -> batch_size=1 UA.
        """
        if out_msg.batch_size != self.batch_size:
            raise ValueError(
                f"Output message batch size mismatch: expected {self.batch_size}, got {out_msg.batch_size}"
            )
        self.output_product = out_msg.product_reduce_over_batch()

    def _apply_incremental_update(self, new_blk: UA, old_blk: UA, blk: slice) -> None:
        """
        Incremental update on cache hit:
            output_product <- output_product * prod(new_blk) / prod(old_blk)
        """
        if self.output_product is None:
            # If accumulator is missing, fall back to full rebuild semantics upstream.
            raise RuntimeError("output_product is not initialized for incremental update.")

        new_prod = new_blk.product_reduce_over_batch()
        old_prod = old_blk.product_reduce_over_batch()

        # Update accumulated product
        self.output_product = (self.output_product * new_prod) / old_prod

    # ------------------------------------------------------------------
    # Forward message computation
    # ------------------------------------------------------------------
    def _compute_forward(self, inputs: Dict[str, UA], block=None) -> UA:
        """
        Compute forward message: input (batch=1) -> output (batch=B or block_size).

        Semantics:
            - Warm-start (output_message is None): replicate input message.
            - Otherwise: EP residual = (replicated belief) / output_message (block-wise supported).

        Note:
            output_product is a batch-reduced product of output-side messages and is
            treated as read-only here. It is updated in backward().
        """
        x_msg = inputs["input"]
        if x_msg.batch_size != 1:
            raise ValueError("ForkPropagator expects input UA with batch_size=1.")

        # Warm-start: deterministic replication only
        if self.output_message is None:
            if block is not None:
                raise RuntimeError(
                    "Block-wise forward called before warm-start. "
                    "Run a full-batch forward() once before sequential updates."
                )
            return x_msg.fork(batch_size=self.batch_size)

        # Need accumulator to build belief
        if self.output_product is None:
            raise RuntimeError(
                "output_product is not initialized. "
                "Run backward() (full or block-wise) to build the accumulator."
            )

        # Full-batch forward
        if block is None:
            belief = x_msg * self.output_product
            belief_forked = belief.fork(batch_size=self.batch_size)
            return belief_forked / self.output_message

        # Block-wise forward
        blk = self._normalize_block(block)
        blk_size = blk.stop - blk.start

        belief = x_msg * self.output_product
        belief_forked = belief.fork(batch_size=blk_size)

        out_blk = self.output_message.extract_block(blk)
        return belief_forked / out_blk

    # ------------------------------------------------------------------
    # Backward pass (block-aware)
    # ------------------------------------------------------------------
    def backward(self, block=None) -> None:
        """
        Send backward message to the single input wave (batch_size=1).

        Uses AccumulativePropagator's speculative caching for block-wise scheduling.
        """
        out_msg = self.output_message
        if out_msg is None:
            raise RuntimeError("Missing output message for backward.")

        x_wave = self.inputs.get("input")
        if x_wave is None:
            raise RuntimeError("Input wave not connected.")

        # Full-batch backward: rebuild from scratch and reset speculative cache
        if block is None:
            self._rebuild_accumulator_from_output(out_msg)
            msg_in = self.output_product
            x_wave.receive_message(self, msg_in)
            self._store_backward_message(x_wave, msg_in)
            self._clear_spec_cache()
            return

        # Block-wise backward: speculative incremental or miss->rebuild is handled by base
        blk = self._normalize_block(block)
        self._backward_with_speculation(out_msg, blk)

        msg_in = self.output_product
        x_wave.receive_message(self, msg_in)
        self._store_backward_message(x_wave, msg_in)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def get_sample_for_output(self, rng=None):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        forked_sample = np().broadcast_to(x, (self.batch_size,) + x_wave.event_shape).copy()
        return forked_sample
