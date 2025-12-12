from typing import Optional, Union, Any
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode, get_real_dtype
from .base import Measurement


class GaussianMeasurement(Measurement):
    """
    Gaussian measurement model: y ~ N(x, var) or CN(x, var)

    This measurement assumes the observation y is the latent x plus Gaussian noise:
        y = x + ε,  ε ~ N(0, var) or CN(0, var)

    Supports real and complex types, scalar or array precision, and optional masking.
    """

    expected_input_dtype = None
    expected_observed_dtype = None

    def __init__(
        self,
        var: float = 1.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        with_mask: bool = False,
        label: str = None
    ) -> None:
        self._var = var
        self._precision_value = 1.0 / var
        self.belief = None

        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        super().__init__(with_mask = with_mask, label = label)
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        # Use the same dtype for observed data as for latent variable
        return input_dtype

    def _compute_message(self, incoming: UA, block=None) -> UA:
        """
        Compute backward EP message for GaussianMeasurement.

        Block semantics:
            - block=None:
                return full-batch observed message
            - block=slice:
                return only the corresponding batch slice
        """
        self._check_observed()

        # Extract block from observed
        obs_blk = self.observed.extract_block(block)

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return obs_blk.as_scalar_precision()

        return obs_blk


    def _generate_sample(self, rng: Any) -> None:
        """
        Draw synthetic sample: x + ε, where ε ~ Normal(0, var)
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")
        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        self._sample = (x + np().sqrt(self._var) * noise).astype(self.input_dtype)
    
    def compute_belief(self) -> UA:
        """
        Construct and return the current belief q(z) ∝ m(z) * l(z).

        Since both the incoming message and the likelihood are Gaussian,
        the posterior belief is obtained by simple multiplication of the
        corresponding UncertainArrays.

        Returns
        -------
        UncertainArray
            The posterior belief (mean and precision fused).

        Raises
        ------
        RuntimeError
            If the observed data or incoming message is missing.
        """
        self._check_observed()
        if self.input is None or self.input not in self.input_messages:
            raise RuntimeError("Cannot compute belief: missing input message.")

        msg = self.input_messages[self.input]
        obs = self.observed

        # Combine message and likelihood: q(z) ∝ m(z) * l(z)
        belief = obs * msg
        self.belief = belief
        return belief

    def compute_fitness(self) -> float:
        """
        Compute precision-weighted squared error between the belief mean
        and the observed data.

        The fitness is defined as:
            fitness = Σ_i γ_i * |μ_belief_i - y_i|² / Σ_i γ_i
        where γ_i are observation precisions.

        Masked elements (if any) are excluded from the sum.

        Returns
        -------
        float
            Scalar fitness value (smaller means better data-fit).
        """
        self._check_observed()
        belief = self.compute_belief()
        obs = self.observed
        xp = np()

        # Extract mean and precision arrays
        mu_belief = belief.data
        y = obs.data
        gamma = obs.precision(raw=True)

        # Compute squared magnitude difference
        diff2 = xp.abs(mu_belief - y) ** 2

        # Precision-weighted average
        fitness = xp.mean(gamma * diff2)

        return float(fitness) 


    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen}, mode={self.precision_mode})"
