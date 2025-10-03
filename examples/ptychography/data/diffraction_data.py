from dataclasses import dataclass, field
from typing import Tuple, Optional
from gpie.core.backend import np


@dataclass
class DiffractionData:
    """
    Container class for a single diffraction pattern and its metadata.

    Attributes:
        position: (y, x) position of the scan.
        diffraction: The complex-valued 2D diffraction field (amplitude or phase).
        noise: Optional noise variance (σ²), used in likelihood modeling.
        indices: Optional slice for cropping into the object region.
        meta: Arbitrary metadata dictionary.
    """
    position: Tuple[int, int]
    diffraction: np().ndarray
    noise: Optional[float] = None  # Variance
    indices: Optional[Tuple[slice, slice]] = None
    meta: dict = field(default_factory=dict)

    def intensity(self) -> np().ndarray:
        """Return the squared amplitude (intensity) image."""
        return np().abs(self.diffraction) ** 2

    def get_noise(self) -> float:
        """Return noise variance if set, else raise."""
        if self.noise is None:
            raise ValueError("Noise variance is not set for this diffraction pattern.")
        return self.noise

    def summary(self) -> str:
        """Return a compact string summarizing this object."""
        return f"Pos={self.position}, shape={self.diffraction.shape}, noise={self.noise}"

    def show(self, ax=None, log_scale=True, cmap="viridis"):
        """Visualize the diffraction pattern."""
        import matplotlib.pyplot as plt
        data = np().abs(self.diffraction)
        if log_scale:
            data = np().log10(data + 1e-8)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(data, cmap=cmap)
        ax.set_title(f"Diffraction @ {self.position}")
        ax.axis("off")
        return ax
