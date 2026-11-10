"""CPU tests: per-shape DMA ring pieces (ported tunamitom/flashinfer@2676f2b pcie_dma hunk + @0aa09dd)."""

from types import SimpleNamespace

import pytest

from b12x.comm.pcie.pcie_dma import SHAPE_PIECES, PCIeDmaAllReduce


def pick(world, rows, cols, override=0, shape=True):
    ring = SimpleNamespace(world_size=world, _pieces_override=override)
    shard = rows * cols // world
    return PCIeDmaAllReduce._pick_pieces(ring, shard, shard * 2, shape=(rows, cols) if shape else None)


@pytest.mark.parametrize("rows", [128, 256, 512, 1024, 1536, 2048, 2560, 3072])
@pytest.mark.parametrize("cols", [3584, 7168])
def test_world8_table_selects_measured_pieces(rows, cols):
    expected = SHAPE_PIECES[(rows, cols)]
    shard_bytes = rows * cols // 8 * 2
    if shard_bytes // expected < 512 << 10:  # below the 512 KiB minimum piece the ring falls back
        expected = 2 if shard_bytes // 2 >= 512 << 10 else 1
    assert pick(8, rows, cols) == expected


def test_table_only_two_pieces_for_the_largest_hidden_reductions():
    assert {k for k, v in SHAPE_PIECES.items() if v != 1} == {(2560, 7168), (3072, 7168)}


@pytest.mark.parametrize("world", [2, 3, 4])
def test_other_worlds_keep_the_constant_two_rule(world):
    assert pick(world, 3072, 7168) == 2
    assert pick(world, 3072, 7168, shape=False) == 2


def test_override_wins_when_usable_and_unknown_shapes_keep_two():
    assert pick(8, 3072, 3584, override=4) == 4
    assert pick(8, 4096, 7168) == 2
    assert pick(8, 64, 7168) == 1  # 112 KiB shard: one piece
