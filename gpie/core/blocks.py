from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator, List


@dataclass
class BlockGenerator:
    """
    Minimal contiguous block generator for batch-wise message passing.

    Semantics:
        - If block_size is None or block_size >= B:
            → treat as a single full block [0:B]
        - If block_size == 1:
            → blocks are [0:1], [1:2], ..., [B-1:B]
        - If 1 < block_size < B:
            → contiguous blocks of requested size.
    """
    B: int
    block_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Compute contiguous slices based solely on block_size."""
        if self.B <= 0:
            raise ValueError(f"B must be positive, got {self.B}.")

        # Determine effective block size
        if self.block_size is None:
            bsz = self.B
        else:
            if self.block_size <= 0:
                raise ValueError(f"block_size must be positive, got {self.block_size}.")
            # Cap at B
            bsz = min(self.block_size, self.B)

        self.block_size = bsz

        # Precompute contiguous slice blocks
        # Example: B=8, bsz=2 → [0:2], [2:4], [4:6], [6:8]
        edges = list(range(0, self.B, self.block_size))
        if edges[-1] != self.B:
            edges.append(self.B)

        blocks: List[slice] = []
        for i in range(len(edges) - 1):
            start, stop = edges[i], edges[i+1]
            if start < stop:
                blocks.append(slice(start, stop))

        self._blocks = blocks

    def iter_blocks(self) -> Iterator[slice]:
        """
        Yield blocks in sequential order for each epoch.
        """
        for s in self._blocks:
            yield s

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return f"BlockGenerator(B={self.B}, block_size={self.block_size}, n_blocks={len(self)})"
