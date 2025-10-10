from typing import Union, Optional
from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.backend import np, move_array_to_current_backend
from ...core.types import (
    PrecisionMode,
    UnaryPropagatorPrecisionMode,
    get_lower_precision_dtype,
    get_real_dtype,
)


class MultiplyConstPropagator(Propagator):
    """
    MultiplyConstPropagator
    -----------------------
    Deterministic unary propagator that multiplies the incoming wave
    by a fixed constant or field (e.g., a probe illumination function in ptychography).

    This propagator performs elementwise multiplication:
        Forward :  μ_out = μ_in × const
        Backward:  μ_in  = μ_out × conj(const) / (|const|² + ε)

    Precision (inverse variance) propagation:
        Forward :  prec_out = prec_in / (|const|² + ε)
        Backward:  prec_in  = prec_out × |const|²

    where ε > 0 is a small stabilizer added to the denominator to avoid
    precision explosion when |const| → 0.

    Parameters
    ----------
    const : float | complex | ndarray
        Constant multiplier (illumination field). Can be scalar or any array
        broadcastable to the input wave shape.
    eps : float, optional (default: 1e-8)
        Positive stabilizer added to |const|² in divisions for numerical stability.

    Attributes
    ----------
    const : ndarray
        The complex multiplicative field.
    const_conj : ndarray
        Complex conjugate of `const`.
    const_abs_sq : ndarray
        Squared amplitude |const|² used for precision scaling.
    _eps : ndarray
        Stabilization constant stored as a scalar array on the current backend.
    const_dtype : dtype
        Data type of the constant field.
    _init_rng : Generator | None
        Optional RNG for initializing messages when needed.
    """

    def __init__(self, const: Union[float, complex, np().ndarray], *, eps: float = 1e-12, dtype = np().complex128):
        """
        Initialize a propagator that multiplies the incoming message by a fixed complex field.

        Parameters
        ----------
        const : float | complex | ndarray
            The illumination field (probe) or any constant multiplicative field.
            It can be scalar or an array broadcastable to the wave shape later in `__matmul__`.
        eps : float, optional (default: 1e-8)
            Non-negative stabilizer added to |const|^2 in divisions to avoid precision blow-ups:
                forward  : precision_out = precision_in / (|const|^2 + eps)
                backward : mean_in = mean_out * conj(const) / (|const|^2 + eps)
            Note: this does NOT clamp const itself; it regularizes only the denominators.
        """
        super().__init__(input_names=("input",))
        # Store the raw constant as provided (no clamping). Keep dtype for later synchronization.
        self.const = np().asarray(const, dtype = dtype)
        self.const_dtype = self.const.dtype
        self._init_rng = None

        # Validate and store stabilizer epsilon in a dtype-consistent 0-D array.
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        self._eps = np().array(eps, dtype=get_real_dtype(self.const_dtype))

        # Precompute caches used by forward/backward:
        #   - const_conj  : complex conjugate of const
        #   - const_abs_sq: |const|^2 (real)
        self._rebuild_cached_fields()

    def _rebuild_cached_fields(self) -> None:
        """(Re)build cached arrays derived from `self.const`. Call after dtype/backend changes."""
        # Conjugate shares the complex dtype with `const`.
        self.const_conj = np().conj(self.const)
        # |const|^2 is real-valued with the corresponding real dtype.
        self.const_abs_sq = np().abs(self.const) ** 2

    def to_backend(self):
        """
        Move internal arrays to the current backend and refresh cached fields.

        Notes
        -----
        - `const` is moved first and becomes the source of truth.
        - `const_conj` and `const_abs_sq` are rebuilt to guarantee consistency.
        - `_eps` is cast to the current real dtype to keep arithmetic stable.
        """
        # Move `const` to the active backend with its original complex dtype.
        self.const = move_array_to_current_backend(self.const, dtype=self.const_dtype)
        self.const_dtype = self.const.dtype  # sync dtype after move

        # Rebuild caches on the active backend to ensure exact consistency.
        self._rebuild_cached_fields()

        # Cast epsilon to the current backend and real dtype corresponding to `const`.
        self._eps = move_array_to_current_backend(self._eps, dtype=get_real_dtype(self.const_dtype))


    def set_init_rng(self, rng):
        self._init_rng = rng
    

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)

        if mode not in {
            UnaryPropagatorPrecisionMode.ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        }:
            raise ValueError(f"Unsupported precision mode for MultiplyConstPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        mode = self.inputs["input"].precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)


    def set_precision_mode_backward(self):
        input_mode = self.inputs["input"].precision_mode_enum
        output_mode = self.output.precision_mode_enum
        if output_mode is None or output_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
        elif input_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
        else:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)


    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY


    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY

    def _compute_forward(self, input_msg: UA) -> UA:
        """
        Compute the forward message through the multiplicative constant factor.

        Forward mapping:
            μ_out = μ_in × const
            prec_out = prec_in / (|const|² + ε)

        Notes
        -----
        - This operation propagates the uncertainty through elementwise multiplication.
        - The denominator regularization (|const|² + ε) prevents precision blow-up
        near regions where |const| ≈ 0.
        """
        dtype = np().result_type(input_msg.dtype, self.const_dtype)
        input_msg = input_msg.astype(dtype)

        const = self.const.astype(dtype)
        abs_sq = self.const_abs_sq.astype(get_real_dtype(dtype))
        eps = self._eps.astype(get_real_dtype(dtype))

        mu = input_msg.data * const
        prec = input_msg.precision(raw=True) / (abs_sq + eps)
        return UA(mu, dtype=dtype, precision=prec)


    def _compute_backward(self, output_msg: UA) -> UA:
        """
        Compute the backward message through the multiplicative constant factor.

        Backward mapping (adjoint of the forward transform):
            μ_in = μ_out × conj(const) / (|const|² + ε)
            prec_in = prec_out × |const|²

        Notes
        -----
        - Uses complex conjugation to ensure correct adjoint mapping.
        - The same regularization ε used in forward ensures numerical stability
        while maintaining conjugate symmetry.
        """
        dtype = np().result_type(output_msg.dtype, self.const_dtype)
        const_conj = self.const_conj.astype(dtype)
        abs_sq = self.const_abs_sq.astype(get_real_dtype(dtype))
        eps = self._eps.astype(get_real_dtype(dtype))

        mu = output_msg.data * const_conj / (abs_sq + eps)
        prec = output_msg.precision(raw=True) * abs_sq
        return UA(mu, dtype=dtype, precision=prec)


    def forward(self):
        input_msg = self.input_messages[self.inputs["input"]]
        if input_msg is None:
            raise RuntimeError("No message to forward")
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            else:
                scalar = self.output._precision_mode == PrecisionMode.SCALAR
                msg = UA.random(
                event_shape=self.output.event_shape,
                batch_size=self.output.batch_size,
                dtype=self.dtype,
                scalar_precision=scalar,
                rng=self._init_rng,
                )
                self.output.receive_message(self, msg)
                return

        msg = self._compute_forward(input_msg)
        if self.output.precision_mode_enum == PrecisionMode.ARRAY:
            self.output.receive_message(self, msg)
        else:
            qy = (msg * self.output_message.as_array_precision()).as_scalar_precision()
            msg_to_send = qy / self.output_message
            self.output.receive_message(self, msg_to_send)
        return

    def backward(self):
        msg = self._compute_backward(self.output_message)
        if self.inputs["input"].precision_mode_enum == PrecisionMode.ARRAY:
            self.inputs["input"].receive_message(self, msg)
        else:
            input_msg = self.input_messages[self.inputs["input"]]
            qx = (msg * input_msg.as_array_precision()).as_scalar_precision()
            msg_to_send = qx / input_msg
            self.inputs["input"].receive_message(self, msg_to_send)

    def get_sample_for_output(self, rng=None):
        x = self.inputs["input"].get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        const = self.const.astype(x.dtype) if self.const_dtype != x.dtype else self.const
        return x * const

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to a Wave via the `@` operator.

        This operation determines the common dtype, broadcasts the constant
        to match the Wave shape, and constructs the output Wave.

        Parameters
        ----------
        wave : Wave
            Input Wave node. Its (batch_size, event_shape, dtype) determine
            how `const` will be broadcast and cast.

        Returns
        -------
        Wave
            Output Wave produced by this propagator.
        """
        # --- dtype negotiation ---
        self.dtype = get_lower_precision_dtype(wave.dtype, self.const_dtype)
        self.const = np().asarray(self.const, dtype=self.dtype)
        self.const_dtype = self.const.dtype

        # --- broadcast constant to expected shape ---
        expected_shape = (wave.batch_size, *wave.event_shape)
        try:
            self.const = np().broadcast_to(self.const, expected_shape)
        except ValueError:
            raise ValueError(
                f"MultiplyConstPropagator: constant shape {self.const.shape} "
                f"is not broadcastable to wave shape {expected_shape}"
            )

        self._rebuild_cached_fields()

        # Ensure epsilon lives on the same backend and dtype
        self._eps = move_array_to_current_backend(self._eps, dtype=get_real_dtype(self.const_dtype))

        # --- connect graph structure ---
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(
            event_shape=wave.event_shape,
            batch_size=wave.batch_size,
            dtype=self.dtype,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output



    def __repr__(self) -> str:
        mode = self.precision_mode or "unset"
        shape_str = (
            f"scalar" if np().isscalar(self.const) or self.const.shape == ()
            else f"shape={self.const.shape}"
        )
        return f"MultiplyConst(mode={mode}, {shape_str})"