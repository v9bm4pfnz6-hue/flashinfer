"""CPU 9-rank random-interleaving emulation of zero-copy BF16 direct, IEEE binary24 (fp24) ring, and reduce-scatter."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_pcie_dma_tree_emulator", pathlib.Path(__file__).with_name("test_pcie_dma_tree.py")
)
emulator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emulator)

WORLD = 9


def _fp24_encode_py(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact bit-level model of _pack_fp24x4 (IEEE binary32 -> RN binary24: hi16 + lo8)."""
    assert vals.dtype == torch.float32 and vals.numel() % 4 == 0
    bits = vals.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    abs_bits = bits & 0x7FFFFFFF
    is_fin = abs_bits < 0x7F800000
    is_nan = abs_bits > 0x7F800000
    lsb = (bits >> 8) & 1
    r = lsb + 0x7F
    rounded = torch.where(is_fin, (bits + r) & 0xFFFFFFFF, bits)
    rounded = torch.where(is_nan, rounded | 0x00400000, rounded)
    hi16 = ((rounded >> 16) & 0xFFFF).to(torch.uint16)
    lo8 = ((rounded >> 8) & 0xFF).to(torch.uint8)
    return hi16, lo8


def _fp24_decode_py(hi16: torch.Tensor, lo8: torch.Tensor) -> torch.Tensor:
    """Exact bit-level model of _unpack_fp24x4."""
    u = ((hi16.to(torch.int64) & 0xFFFF) << 16) | ((lo8.to(torch.int64) & 0xFF) << 8)
    return u.to(torch.uint32).view(torch.float32)


class _Tensor:
    def __init__(self, space, elems, dtype):
        self.dtype = dtype
        self.elems = elems
        self.size = torch.empty((), dtype=dtype).element_size()
        self.address = space.allocate(elems * self.size)

    @property
    def nbytes(self):
        return self.elems * self.size

    def data_ptr(self):
        return self.address

    def numel(self):
        return self.elems

    def element_size(self):
        return self.size

    def is_contiguous(self):
        return True


class _DirectMachine(emulator._Machine):
    def dma_direct_step(
        self,
        in_mode,
        out_mode,
        local_bf16_ptr,
        wire_in_ptr,
        scales_in_ptr,
        out_ptr,
        scales_out_ptr,
        elems,
        *,
        wait,
    ):
        self.emit("wait_flag", wait[0], wait[1])
        space = self.space

        def step():
            a = space.bf16(local_bf16_ptr, elems).float()
            if in_mode == "bf16":
                b = space.bf16(wire_in_ptr, elems).float()
            elif in_mode == "fp32":
                b = space.f32(wire_in_ptr, elems).clone()
            elif in_mode == "fp24":
                hi16 = torch.frombuffer(space.view(wire_in_ptr, elems * 2), dtype=torch.uint16).clone()
                lo8 = torch.frombuffer(space.view(scales_in_ptr, elems), dtype=torch.uint8).clone()
                b = _fp24_decode_py(hi16, lo8)
            else:
                raise AssertionError(in_mode)
            v = b + a
            if out_mode == "fp32":
                space.f32(out_ptr, elems).copy_(v)
            elif out_mode == "bf16":
                space.bf16(out_ptr, elems).copy_(v.to(torch.bfloat16))
            elif out_mode == "fp24":
                hi_out, lo_out = _fp24_encode_py(v)
                space.view(out_ptr, elems * 2)[:] = hi_out.view(torch.uint8).numpy()
                space.view(scales_out_ptr, elems)[:] = lo_out.numpy()
            else:
                raise AssertionError(out_mode)

        self.emit("host", step)

    def dma_direct_wait_copy(self, dst_ptr, src_ptr, elems, *, wait):
        self.emit("wait_flag", wait[0], wait[1])
        space = self.space
        self.emit("host", lambda: space.bf16(dst_ptr, elems).copy_(space.bf16(src_ptr, elems)))


def _build_world(monkeypatch, direct_mode: str, shard_elems: int, pieces: int, shard_cap_override: int | None = None):
    from b12x.comm.pcie import pcie_dma

    space = emulator._Space()
    flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
    if shard_cap_override is not None:
        capacity = shard_cap_override
    elif direct_mode == "fp24":
        capacity = ((shard_elems * 3) + 255) // 256 * 256
    else:
        capacity = ((shard_elems * 4) + 255) // 256 * 256
    slabs = [space.allocate(flags_bytes + 2 * (WORLD - 1) * capacity) for _ in range(WORLD)]
    machines, runtimes = [], []
    for rank in range(WORLD):
        machine = _DirectMachine(space)
        machine.streams = {name: emulator._Stream(machine, name) for name in ("copy", "flag", "ring")}
        rt = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        rt.world_size, rt.rank, rt.device = WORLD, rank, "fake"
        rt._tree, rt._closed = None, False
        rt._direct_mode = direct_mode
        rt.shard_capacity = capacity
        rt._direct_max_shard_elems = shard_elems
        rt._flags_base = list(slabs)
        rt._scratch_base = [base + flags_bytes for base in slabs]
        for name in ("_send_counters", "_wait_counters", "_wait_done"):
            setattr(rt, name, _Tensor(space, pcie_dma.FLAG_SLOTS, torch.int32))
        rt._direct_stage = _Tensor(space, 2 * capacity, torch.uint8)
        rt._copy_stream = machine.streams["copy"]
        rt._flag_stream = machine.streams["flag"]
        rt._ring_stream = machine.streams["ring"]
        rt._piece_events = [emulator._Event(machine) for _ in range(pcie_dma.MAX_PIECES)]
        rt._copied_events = [emulator._Event(machine) for _ in range(2 * (WORLD - 1) * pcie_dma.MAX_PIECES)]
        rt._input_ready = emulator._Event(machine)
        rt._kernels = machine
        rt._pieces_override = pieces
        machines.append(machine)
        runtimes.append(rt)
    return pcie_dma, space, machines, runtimes


def _reference_ring_bf16(inputs: torch.Tensor) -> torch.Tensor:
    """Exact ordered FP32 ring accumulation rounded once to BF16 per chunk."""
    chunks = []
    for chunk in range(WORLD):
        total = inputs[chunk, chunk].float().clone()
        for step in range(1, WORLD):
            total = total + inputs[(chunk + step) % WORLD, chunk].float()
        chunks.append(total.to(torch.bfloat16))
    return torch.stack(chunks).reshape(-1)


@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("multi_interval", [False, True])
def test_bf16_direct_is_bit_identical_to_ordered_fp32_ring(monkeypatch, pieces, in_place, multi_interval):
    shard_elems = 512
    cap = 256 * 4 if multi_interval else None
    pcie_dma, space, machines, runtimes = _build_world(monkeypatch, "fp32", shard_elems, pieces, cap)
    gen = torch.Generator().manual_seed(42 + pieces + int(in_place) * 7)
    rounds = 2
    inputs = (
        torch.randn((rounds, WORLD, WORLD, shard_elems), generator=gen)
        * torch.pow(2.0, torch.randint(-10, 11, (rounds, WORLD, WORLD, shard_elems), generator=gen).float())
    ).to(torch.bfloat16)
    inputs[0, 4, 2, 13] = float("inf")
    inputs[0, 6, 7, 29] = float("nan")

    results = [[None] * WORLD for _ in range(rounds)]
    for rank, (machine, rt) in enumerate(zip(machines, runtimes)):
        src = _Tensor(space, WORLD * shard_elems, torch.bfloat16)
        dst = src if in_place else _Tensor(space, WORLD * shard_elems, torch.bfloat16)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, m=machine: m.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        for r in range(rounds):
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, src=src: space.bf16(src.address, WORLD * shard_elems).copy_(inputs[r, rank].reshape(-1)))
            )
            rt._run_direct_bf16(src, dst, reduce_scatter_only=False)
            machine.main.operations.append(
                ("host", lambda r=r, rank=rank, dst=dst: results[r].__setitem__(rank, space.bf16(dst.address, WORLD * shard_elems).clone()))
            )
    emulator._execute(space, machines, seed=11)

    for r in range(rounds):
        expected = _reference_ring_bf16(inputs[r])
        for rank in range(WORLD):
            got = results[r][rank]
            nan_mask = torch.isnan(expected)
            assert torch.equal(torch.isnan(got), nan_mask)
            assert torch.equal(got[~nan_mask].view(torch.int16), expected[~nan_mask].view(torch.int16)), (r, rank)


@pytest.mark.parametrize("pieces", [1, 2])
@pytest.mark.parametrize("in_place", [True, False])
def test_fp24_ring_preserves_bound_across_16_binades_special_values_and_mixed_reduce_scatter(monkeypatch, pieces, in_place):
    shard_elems = 512
    pcie_dma, space, machines, runtimes = _build_world(monkeypatch, "fp24", shard_elems, pieces)
    gen = torch.Generator().manual_seed(109 + pieces)
    # 16 binades + extreme adjacent channel outliers + Inf + -Inf + Inf+(-Inf)->NaN + NaN!
    inputs = (
        (torch.rand((WORLD, WORLD, shard_elems), generator=gen, dtype=torch.float64) - 0.5)
        * 12.0
        * torch.pow(2.0, torch.randint(-8, 8, (WORLD, WORLD, shard_elems), generator=gen).double())
    ).to(torch.bfloat16)
    inputs[:, 0, 0] = 10000.0
    inputs[:, 0, 1] = 1e-4  # Quiet channel right next to massive outlier!
    inputs[4, 1, 5] = float("inf")
    inputs[2, 3, 10] = float("-inf")
    inputs[5, 6, 17] = float("inf")
    inputs[7, 6, 17] = float("-inf")  # Inf + -Inf -> NaN!
    inputs[6, 8, 31] = float("nan")

    rs_inputs = (torch.randn((WORLD, WORLD, shard_elems), generator=gen) * 3.5).to(torch.bfloat16)

    ar_out = [None] * WORLD
    rs_out = [None] * WORLD
    gather_out = [None] * WORLD
    gather_bytes = 128 * pieces
    gather_src = np.random.default_rng(77).integers(0, 256, size=(WORLD, gather_bytes), dtype=np.uint8)

    for rank, (machine, rt) in enumerate(zip(machines, runtimes)):
        src = _Tensor(space, WORLD * shard_elems, torch.bfloat16)
        dst = src if in_place else _Tensor(space, WORLD * shard_elems, torch.bfloat16)
        rs_src = _Tensor(space, WORLD * shard_elems, torch.bfloat16)
        rs_dst = _Tensor(space, shard_elems, torch.bfloat16)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, m=machine: m.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)

        # 1) all_reduce -> 2) reduce_scatter -> 3) ring all_gather -> 4) second all_reduce on the SAME channel!
        machine.main.operations.append(
            ("host", lambda rank=rank, src=src: space.bf16(src.address, WORLD * shard_elems).copy_(inputs[rank].reshape(-1)))
        )
        rt._run_direct_bf16(src, dst, reduce_scatter_only=False)
        machine.main.operations.append(
            ("host", lambda rank=rank, dst=dst: ar_out.__setitem__(rank, space.bf16(dst.address, WORLD * shard_elems).clone()))
        )

        machine.main.operations.append(
            ("host", lambda rank=rank, rs_src=rs_src: space.bf16(rs_src.address, WORLD * shard_elems).copy_(rs_inputs[rank].reshape(-1)))
        )
        rt._run_direct_bf16(rs_src, rs_dst, reduce_scatter_only=True)
        machine.main.operations.append(
            ("host", lambda rank=rank, rs_dst=rs_dst: rs_out.__setitem__(rank, space.bf16(rs_dst.address, shard_elems).clone()))
        )

        own_slot = rt._scratch_ptr(rank, WORLD - 1)
        machine.main.operations.append(
            ("host", lambda rank=rank, own_slot=own_slot: space.view(own_slot, gather_bytes).__setitem__(slice(None), gather_src[rank]))
        )
        rt._run_gather(gather_bytes, pieces=pieces)
        machine.main.operations.append(
            ("host", lambda rank=rank, rt=rt: gather_out.__setitem__(
                rank, np.stack([space.view(rt._scratch_ptr(rank, s), gather_bytes).copy() for s in range(WORLD)])
            ))
        )
        rt._release_gather()

        machine.main.operations.append(
            ("host", lambda rank=rank, src=src: space.bf16(src.address, WORLD * shard_elems).copy_(inputs[rank].reshape(-1)))
        )
        rt._run_direct_bf16(src, dst, reduce_scatter_only=False)

    emulator._execute(space, machines, seed=23)

    exact = inputs.double().sum(dim=0).reshape(-1)
    mag = inputs.double().abs().sum(dim=0).reshape(-1)
    rounded = exact.to(torch.bfloat16)
    finite = torch.isfinite(rounded) & torch.isfinite(mag)
    # 7 binary24 roundings (2^-16 each) + 1 final BF16 rounding (2^-8):
    bound = (8 * 2.0 ** -16 * mag + 2.0 ** -8 * exact.abs() + 1e-30)[finite]
    for rank in range(WORLD):
        res = ar_out[rank]
        assert torch.equal(torch.isnan(res), torch.isnan(rounded)), rank
        assert torch.equal(torch.isposinf(res), torch.isposinf(rounded)), rank
        assert torch.equal(torch.isneginf(res), torch.isneginf(rounded)), rank
        if rank > 0:
            nm = ~torch.isnan(res)
            assert torch.equal(res[nm].view(torch.int16), ar_out[0][nm].view(torch.int16))
        err = (res.double() - exact).abs()[finite]
        worst = float((err / bound).max())
        assert worst <= 1.01, f"worst error ratio {worst}"
        # Relative L2 error vs FP64 exact must match pure FP32 ring to 3+ decimal places!
        rel_l2_fp24 = float((res[finite].double() - exact[finite]).norm() / exact[finite].norm())
        rel_l2_fp32 = float((rounded[finite].double() - exact[finite]).norm() / exact[finite].norm())
        assert abs(rel_l2_fp24 - rel_l2_fp32) / rel_l2_fp32 < 0.01, (rel_l2_fp24, rel_l2_fp32)
        agree = float((res[finite].view(torch.int16) == rounded[finite].view(torch.int16)).float().mean())
        assert agree > 0.99, f"agreement {agree}"

    # Verify quiet channel next to 10000.0 outlier kept all its precision!
    assert abs(float(ar_out[0][1]) - float(rounded[1])) == 0.0

    # Check reduce_scatter ownership: rank r receives chunk r!
    rs_exact = rs_inputs.double().sum(dim=0).to(torch.bfloat16)
    for rank in range(WORLD):
        agree_rs = float((rs_out[rank].view(torch.int16) == rs_exact[rank].view(torch.int16)).float().mean())
        assert agree_rs > 0.99, (rank, agree_rs)

    # Check gather after reduce_scatter
    for rank, rt in enumerate(runtimes):
        for s in range(WORLD):
            assert np.array_equal(gather_out[rank][s], gather_src[rt.gather_rank(s)])
