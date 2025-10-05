# examples/ptychography/data/dataset.py

from typing import List, Tuple, Union, Optional
from .diffraction_data import DiffractionData
from ..simulator.forward import generate_diffraction_data
from ..utils.geometry import realspace_to_pixel_coords, filter_positions_within_object, slices_from_positions
from gpie.core.backend import np


class PtychographyDataset:
    """
    Container class for managing object, probe, and diffraction data in a ptychographic experiment.
    """

    def __init__(self):
        self.obj: Optional[np().ndarray] = None
        self.prb: Optional[np().ndarray] = None
        self.pixel_size_um: Optional[float] = 1.0
        self._diff_data: List[DiffractionData] = []

    # --- Object ,Probe, and pixel scale---
    def set_object(self, obj: np().ndarray):
        self.obj = obj

    def set_probe(self, prb: np().ndarray):
        self.prb = prb
    
    def set_pixel_size(self, pixel_size_um: float):
        if pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive.")
        self.pixel_size_um = pixel_size_um


    @property
    def obj_shape(self) -> Optional[Tuple[int, int]]:
        return None if self.obj is None else self.obj.shape

    @property
    def prb_shape(self) -> Optional[Tuple[int, int]]:
        return None if self.prb is None else self.prb.shape

    # --- Diffraction data management ---
    def add_data(self, data: Union[DiffractionData, List[DiffractionData]]):
        """Add one or more DiffractionData objects."""
        if isinstance(data, DiffractionData):
            self._diff_data.append(data)
        elif isinstance(data, list):
            if not all(isinstance(d, DiffractionData) for d in data):
                raise TypeError("List must contain only DiffractionData instances.")
            self._diff_data.extend(data)
        else:
            raise TypeError("Input must be DiffractionData or List[DiffractionData].")

    def clear_data(self):
        self._diff_data.clear()

    def sort_data(
        self,
        key: Union[str, callable] = "center_distance",
        center: Optional[Tuple[int, int]] = None,
        reverse: bool = False,
    ):
        """Sort diffraction data by position or metadata."""

        if callable(key):
            score = key

        elif key == "center_distance":
            if center is None:
                if self.obj is None:
                    raise ValueError("Center not specified and object not set.")
                h, w = self.obj.shape
                center = (h // 2, w // 2)
            cy, cx = center
            def score(d: DiffractionData):
                y, x = d.position
                return (y - cy) ** 2 + (x - cx) ** 2

        elif key.startswith("meta:"):
            meta_key = key.split("meta:", 1)[-1]
            def score(d: DiffractionData):
                return d.meta.get(meta_key, 0.0)

        else:
            raise ValueError(f"Unknown sort key: {key}")

        self._diff_data.sort(key=score, reverse=reverse)

    # --- Accessors ---
    @property
    def scan_positions(self) -> List[Tuple[int, int]]:
        return [d.position for d in self._diff_data]

    @property
    def diffraction_patterns(self) -> List[np().ndarray]:
        return [d.diffraction for d in self._diff_data]

    @property
    def size(self) -> int:
        return len(self._diff_data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> DiffractionData:
        return self._diff_data[idx]
    
    # ---- sumulation ----
    def simulate_diffraction(
        self,
        scan_generator: "Generator[Tuple[float, float], None, None]",
        max_num_points: int,
        noise: float = 1e-4,
        rng: Optional["np.random.Generator"] = None,
    ):
        """
        Generate synthetic diffraction data using internal object/probe and a scan generator.

        Parameters
        ----------
        scan_generator : generator
            Generator yielding (y_um, x_um) real-space scan coordinates.
        max_num_points : int
            Maximum number of scan points to generate.
        noise : float
            Noise variance (σ²) for amplitude measurements.
        rng : np.random.Generator, optional
            Random number generator.
        """
        if self.obj is None or self.prb is None or self.pixel_size_um is None:
            raise ValueError("Object, probe, and pixel size must be set before simulation.")

        # Step 1: Generate real-space scan positions
        positions_real = [next(scan_generator) for _ in range(max_num_points)]

        # Step 2: Convert to pixel coordinates
        positions_pixel = realspace_to_pixel_coords(
            positions_real, pixel_size_um=self.pixel_size_um, obj_shape=self.obj.shape
        )

        # Step 3: Filter out invalid positions
        valid_pixel_positions = filter_positions_within_object(
            positions_pixel, obj_shape=self.obj.shape, probe_shape=self.prb.shape
        )

        # Step 4: Get slice indices
        indices = slices_from_positions(
            valid_pixel_positions, probe_shape=self.prb.shape, obj_shape=self.obj.shape
        )

        # Step 5: Generate diffraction data
        from ..simulator.forward import ptychography_graph
        from ..data.diffraction_data import DiffractionData

        graph = ptychography_graph(
            obj_shape=self.obj.shape,
            prb_shape=self.prb.shape,
            indices=indices,
            noise=noise,
        )
        graph.get_wave("object").set_sample(self.obj)
        graph.get_wave("probe").set_sample(self.prb)
        graph.generate_sample(rng=rng, update_observed=True)
        diffs = graph.get_factor("meas").get_sample()

        # Step 6: Store results
        results = []
        for i, (pix_pos, idx) in enumerate(zip(valid_pixel_positions, indices)):
            real_pos = (
                (pix_pos[0] - self.obj.shape[0] // 2) * self.pixel_size_um,
                (pix_pos[1] - self.obj.shape[1] // 2) * self.pixel_size_um,
            )
            results.append(DiffractionData(
                position=real_pos,
                diffraction=diffs[i],
                noise=noise,
                indices=idx,
            ))
        self.add_data(results)
