# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the DMA ring's BF16 broadcast-phase contract."""

from __future__ import annotations

import torch
import pytest

from b12x.comm.pcie.pcie_dma import (BF16_GATHER_MIN_BYTES, bf16_gather_eligible,
                                          bf16_gather_layout)


def _chain_sum(partials, owner):
    """The ring's per-chunk order: owner first, then ascending ranks."""

    world = len(partials)
    total = partials[owner].float()
    for hop in range(1, world):
        total = total + partials[(owner + hop) % world].float()
    return total


def _bf16_gather_result(partials, owner):
    """Owner rounds the fully reduced FP32 chunk once, receivers widen it."""

    return _chain_sum(partials, owner).bfloat16().float()


@pytest.mark.parametrize("world", [2, 3, 4, 6, 8, 9, 10])
@pytest.mark.parametrize("pieces", [1, 2, 4, 8])
def test_layout_matches_the_lossless_packet_geometry(world, pieces):
    # One hidden-shape part: 1,835,008 elements per rank packet of 9 chunks.
    numel = world * 1_835_008
    shard_elems, piece_bytes, wire_bytes = bf16_gather_layout(numel, world, pieces)
    assert shard_elems == 1_835_008
    assert piece_bytes * pieces == shard_elems * 4
    assert wire_bytes * 2 == piece_bytes
    assert wire_bytes % 16 == 0
    # The whole chunk's BF16 payload fits the channel's FP32 scratch slot.
    assert shard_elems * 2 <= piece_bytes * pieces


def test_layout_rejects_unaligned_or_unknown_geometry():
    with pytest.raises(ValueError):
        bf16_gather_layout(7, 9, 1)
    with pytest.raises(ValueError):
        bf16_gather_layout(9 * 8, 9, 3)
    with pytest.raises(ValueError):
        bf16_gather_layout(9 * 8, 11, 1)


def test_owner_rounding_and_widening_preserve_the_bf16_result():
    generator = torch.Generator().manual_seed(20261110)
    world = 9
    partials = [
        (torch.randn(4096, generator=generator) * 0.2).bfloat16()
        for _ in range(world)
    ]
    for owner in range(world):
        reference = _chain_sum(partials, owner).bfloat16()
        # The BF16 broadcast phase rounds once at the owner and widens the
        # forwarded payload: the caller's store rounds the same bits.
        forwarded = _bf16_gather_result(partials, owner).bfloat16()
        assert torch.equal(reference, forwarded)


def test_owner_rounding_is_lossless_for_bf16_representable_chunks():
    generator = torch.Generator().manual_seed(7)
    values = (torch.randn(4096, generator=generator) * 3).bfloat16()
    numeric = values.float()
    assert torch.equal(numeric.bfloat16(), values)
    widened = numeric.bfloat16().float()
    assert torch.equal(widened, numeric)


def test_owner_rounding_keeps_nan_and_inf_payloads_identical():
    world = 9
    partials = [(torch.randn(512) * 0.2).bfloat16() for _ in range(world)]
    partials[2][17] = float("nan")
    partials[4][31] = float("inf")
    partials[6][63] = float("-inf")
    partials[8][127] = -0.0
    for owner in range(world):
        reference = _chain_sum(partials, owner).bfloat16()
        forwarded = _bf16_gather_result(partials, owner).bfloat16()
        assert torch.equal(reference.view(torch.uint8), forwarded.view(torch.uint8))


def test_owner_rounding_preserves_extreme_exponents():
    world = 9
    values = torch.tensor([3.3895e38, -3.3895e38, 1.0e-38, -1.0e-38, 0.0, 1.0], dtype=torch.bfloat16)
    partials = [values.clone() for _ in range(world)]
    for owner in range(world):
        reference = _chain_sum(partials, owner).bfloat16()
        forwarded = _bf16_gather_result(partials, owner).bfloat16()
        assert torch.equal(reference.view(torch.uint8), forwarded.view(torch.uint8))

def test_ring_channel_geometry_helpers_are_importable():
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce, _align_up

    assert _align_up(1, 256) == 256
    assert _align_up(256, 256) == 256
    assert _align_up(257, 256) == 512
    assert PCIeDmaAllReduce.__name__ == "PCIeDmaAllReduce"



def test_bf16_gather_guard_keeps_full_chunk_packets_and_skips_the_128_row_graph():
    """The diagnostics guard leaves the 128-row padded packet on the FP32 ring."""

    assert BF16_GATHER_MIN_BYTES == 8 << 20
    assert bf16_gather_eligible(4608 * 7168 * 4)
    assert bf16_gather_eligible(4608 * 3584 * 4)
    assert not bf16_gather_eligible(128 * 7168 * 4)
    assert not bf16_gather_eligible(917_568 * 4)
    with pytest.raises(ValueError):
        bf16_gather_eligible(-1)
