from typing import Optional
import numpy as np
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA


class BinaryPropagator(Propagator):
    """
    Abstract base class for propagators that combine two input Waves
    into a single output Wave using element-wise operations (e.g., addition or multiplication).

    This class handles precision_mode propagation logic and provides 
    abstract methods for forward and backward message computations.

    Subclasses must implement `_compute_forward()` and `_compute_backward()`.
    """

    def __init__(self, dtype=np.complex128, precision_mode=None):
        """
        Initialize the BinaryPropagator.

        Args:
            dtype (np.dtype): Data type of the Wave signals.
            precision_mode (str or None): Optional initial precision mode.
        """
        super().__init__(input_names=("a", "b"), dtype=dtype, precision_mode=precision_mode)

    def __matmul__(self, inputs: tuple[Wave, Wave]) -> Wave:
        """
        Support syntax like: Z = AddPropagator() @ (X, Y)

        Args:
            inputs (tuple): Tuple of two Wave objects (a, b)

        Returns:
            Wave: The output Wave resulting from the propagation.
        """
        if not (isinstance(inputs, tuple) and len(inputs) == 2):
            raise ValueError("BinaryPropagator requires a tuple of two Wave objects.")

        a, b = inputs
        self.add_input("a", a)
        self.add_input("b", b)
        self.dtype = a.dtype

        output = Wave(a.shape, dtype=self.dtype)
        self.connect_output(output)
        return output

    def _set_precision_mode(self, mode: str):
        """
        Set the internal precision_mode for this propagator.

        Raises:
            ValueError: If the requested mode is unsupported or conflicts with an already set mode.
        """
        allowed = (
            "scalar",
            "array",
            "scalar/array to array",
            "array/scalar to array"
        )
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for BinaryPropagator: '{mode}'")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict in BinaryPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def set_precision_mode_forward(self):
        """
        Infer precision_mode from input Waves during forward propagation.
        This function sets only the propagator's internal mode and does not modify input/output Waves.
        """
        a_mode = self.inputs["a"].precision_mode
        b_mode = self.inputs["b"].precision_mode

        if a_mode is None and b_mode is None:
            return

        elif a_mode is not None and b_mode is not None:
            if a_mode == "array" and b_mode == "array":
                self._set_precision_mode("array")
                return
            elif a_mode == "array" and b_mode == "scalar":
                self._set_precision_mode("array/scalar to array")
                return
            elif a_mode == "scalar" and b_mode == "array":
                self._set_precision_mode("scalar/array to array")
                return
            elif a_mode == "scalar" and b_mode == "scalar":
                self._set_precision_mode("scalar")
                return
            else:
                raise ValueError(f"Invalid precision_mode for Wave: '{a_mode}', '{b_mode}'")

        else:
            if a_mode == "array" or b_mode == "array":
                self._set_precision_mode("array")
            elif a_mode == "scalar" or b_mode == "scalar":
                return
            else:
                raise ValueError(f"Invalid precision_mode for Wave: '{a_mode}', '{b_mode}'")

    def set_precision_mode_backward(self):
        """
        Infer precision_mode from the output Wave during backward propagation.
        Based on the current state of input precision_modes, determine the appropriate propagator mode.
        """
        z_mode = self.output.precision_mode
        a_wave = self.inputs["a"]
        b_wave = self.inputs["b"]
        a_mode = a_wave.precision_mode
        b_mode = b_wave.precision_mode


        if z_mode is None:
            return

        if z_mode == "scalar":
            self._set_precision_mode("scalar")
            return

        if z_mode == "array":
            # Case 1: both inputs undefined
            if a_mode is None and b_mode is None:
                self._set_precision_mode("array")
                return
            # Case 2: one undefined, other scalar
            elif a_mode is None and b_mode == "scalar":
                self._set_precision_mode("array/scalar to array")
                return
            elif a_mode == "scalar" and b_mode is None:
                self._set_precision_mode("scalar/array to array")
                return
            elif a_mode == "scalar" and b_mode == "scalar":
                raise ValueError(
                    "Inconsistent state: output is array but both inputs are scalar."
                )
            # Case 3 : one of inputs array
            elif a_mode == "array" or b_mode == "array":
                return
            
        raise ValueError(
            f"Unhandled combination in set_precision_mode_backward(): "
            f"a={a_mode}, b={b_mode}, output={z_mode}"
        )

    def get_output_precision_mode(self) -> Optional[str]:
        """
        Suggest the appropriate precision_mode for the output Wave based on the internal mode.

        Returns:
            str or None: "scalar" or "array", or None if undetermined.
        """
        mode = self.precision_mode

        if mode is None:
            return None

        if mode == "scalar":
            return "scalar"

        # All other modes yield array output
        return "array"

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        """
        Suggest the precision_mode required by a specific input Wave.

        Args:
            wave (Wave): One of the input Waves ("a" or "b")

        Returns:
            str or None: "scalar" or "array", or None if undecided.

        Raises:
            ValueError: If the wave is not recognized as a valid input.
        """
        mode = self.precision_mode
        if mode is None:
            return None

        if mode == "scalar":
            return "scalar"

        if mode == "array":
            return "array"

        if mode == "scalar/array to array":
            if wave == self.inputs["a"]:
                return "scalar"
            elif wave == self.inputs["b"]:
                return "array"

        if mode == "array/scalar to array":
            if wave == self.inputs["a"]:
                return "array"
            elif wave == self.inputs["b"]:
                return "scalar"

        raise ValueError(f"Wave {wave} not recognized as input of this factor.")

    def forward(self):
        """
        Compute and send the message to the output Wave using the inputs' messages.
        """
        a_msg = self.input_messages[self.inputs["a"]]
        b_msg = self.input_messages[self.inputs["b"]]

        if a_msg is None or b_msg is None:
            raise RuntimeError("Missing input messages for BinaryPropagator.")

        output_msg = self._compute_forward({"a": a_msg, "b": b_msg})
        self.output.receive_message(self, output_msg)

    def backward(self):
        """
        Compute and send messages to each input Wave based on the output message.
        Each backward message is computed by excluding the respective input.
        """
        if self.output_message is None:
            raise RuntimeError("Missing output message for backward pass.")

        for exclude in ("a", "b"):
            wave = self.inputs[exclude]
            msg = self._compute_backward(self.output_message, exclude=exclude)
            wave.receive_message(self, msg)

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Abstract method to compute the forward message from the input messages.

        Args:
            inputs (dict): Dictionary containing keys "a" and "b" mapped to UncertainArrays.

        Returns:
            UncertainArray: The forward-propagated message to the output Wave.
        """
        raise NotImplementedError()

    def _compute_backward(self, output: UA, exclude: str) -> UA:
        """
        Abstract method to compute the backward message to one input Wave.

        Args:
            output (UncertainArray): The message from the output Wave.
            exclude (str): Either "a" or "b" indicating which input to exclude.

        Returns:
            UncertainArray: The back-propagated message to the specified input.
        """
        raise NotImplementedError()