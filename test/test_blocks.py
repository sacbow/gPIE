# test_blocks.py
# All comments in English.

import pytest
from gpie.core.blocks import BlockGenerator


# -----------------------------
# Basic construction tests
# -----------------------------
def test_full_batch_default():
    """If block_size=None, the generator should return a single full block."""
    gen = BlockGenerator(B=8, block_size=None)
    blocks = list(gen.iter_blocks())
    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].stop == 8
    assert len(gen) == 1


def test_full_batch_blocksize_geq_B():
    """If block_size >= B, it must behave as full-batch."""
    gen = BlockGenerator(B=8, block_size=10)
    blocks = list(gen.iter_blocks())
    assert len(blocks) == 1
    assert blocks[0] == slice(0, 8)
    assert len(gen) == 1


def test_blocksize_one():
    """block_size=1 should generate B blocks of size 1."""
    B = 5
    gen = BlockGenerator(B=B, block_size=1)
    blocks = list(gen.iter_blocks())
    assert len(blocks) == B
    for i, s in enumerate(blocks):
        assert s == slice(i, i+1)


def test_blocksize_intermediate_even():
    """Intermediate block_size (even division)."""
    gen = BlockGenerator(B=8, block_size=2)
    blocks = list(gen.iter_blocks())
    assert len(blocks) == 4
    assert blocks == [slice(0,2), slice(2,4), slice(4,6), slice(6,8)]


def test_blocksize_intermediate_uneven():
    """Intermediate block_size (uneven division)."""
    gen = BlockGenerator(B=7, block_size=3)
    blocks = list(gen.iter_blocks())
    # Expected: [0:3], [3:6], [6:7]
    assert len(blocks) == 3
    assert blocks[0] == slice(0,3)
    assert blocks[1] == slice(3,6)
    assert blocks[2] == slice(6,7)


# -----------------------------
# Boundary and error conditions
# -----------------------------
def test_B_must_be_positive():
    """B must be > 0."""
    with pytest.raises(ValueError):
        BlockGenerator(B=0, block_size=1)


def test_blocksize_must_be_positive():
    """block_size must be positive if provided."""
    with pytest.raises(ValueError):
        BlockGenerator(B=5, block_size=0)
    with pytest.raises(ValueError):
        BlockGenerator(B=5, block_size=-3)


def test_blocksize_equals_B():
    """block_size=B → full-batch block."""
    gen = BlockGenerator(B=6, block_size=6)
    blocks = list(gen.iter_blocks())
    assert len(blocks) == 1
    assert blocks[0] == slice(0,6)


# -----------------------------
# Regression: reproducible structure
# -----------------------------
def test_len_matches_generated_blocks():
    """__len__() should match the number of blocks in iter_blocks."""
    gen = BlockGenerator(B=9, block_size=4)
    blocks = list(gen.iter_blocks())
    assert len(gen) == len(blocks)


def test_repr_has_key_fields():
    """__repr__ should contain B and block_size."""
    gen = BlockGenerator(B=8, block_size=2)
    rep = repr(gen)
    assert "B=8" in rep
    assert "block_size=2" in rep
    assert "n_blocks=4" in rep
