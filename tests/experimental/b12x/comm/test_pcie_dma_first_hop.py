"""The BF16 broadcast ring with a BF16 first hop, emulated on the CPU.

The runtime's own stream program runs for nine simulated ranks under random interleavings, with the
semantics of test_pcie_dma_tree's emulator. A packet of BF16-representable FP32 values must reduce to
the same bits whether its first hop travels as FP32 or as BF16, and to the ordered FP32 chain rounded
once per chunk.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_pcie_dma_tree_emulator", pathlib.Path(__file__).with_name("test_pcie_dma_tree.py"))
emulator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emulator)

WORLD = 9


class _Tensor:
    def __init__(self, space, elems, dtype):
        self.dtype, self.elems, self.size = dtype, elems, torch.empty((), dtype=dtype).element_size()
        self.address = space.allocate(elems * self.size)

    def data_ptr(self):
        return self.address

    def numel(self):
        return self.elems

    def element_size(self):
        return self.size


class _Machine(emulator._Machine):
    def dma_add(self, destination, a, b, elems, dtype_code, *, wait=None):
        assert dtype_code == 2
        if wait is not None:
            self.emit("wait_flag", wait[0], wait[1])
        space = self.space
        self.emit("host", lambda: space.f32(destination, elems).copy_(space.f32(a, elems) + space.f32(b, elems)))


def _values(space, address, elems, dtype):
    return space.bf16(address, elems) if dtype == torch.bfloat16 else space.f32(address, elems)


def _run(monkeypatch, first_hop, pieces, in_place, seed, shard_elems=64, rounds=2):
    from b12x.comm.pcie import pcie_dma

    space = emulator._Space()
    flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
    capacity = shard_elems * 4
    slabs = [space.allocate(flags_bytes + 2 * (WORLD - 1) * capacity) for _ in range(WORLD)]
    generator = torch.Generator().manual_seed(seed)
    # BF16-representable values, with zeros and a wide range of magnitudes.
    inputs = torch.randn((rounds, WORLD, WORLD, shard_elems), generator=generator)
    inputs = inputs * torch.pow(2.0, torch.randint(-12, 13, inputs.shape, generator=generator).float())
    inputs[torch.rand(inputs.shape, generator=generator) < 0.05] = 0.0
    inputs = inputs.to(torch.bfloat16).float()
    machines, results = [], [[None] * WORLD for _ in range(rounds)]
    for rank in range(WORLD):
        machine = _Machine(space)
        machine.streams = {name: emulator._Stream(machine, name) for name in ("copy", "flag")}
        runtime = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        runtime.world_size, runtime.rank, runtime.device = WORLD, rank, "fake"
        runtime.shard_capacity = capacity
        runtime._flags_base = list(slabs)
        runtime._scratch_base = [base + flags_bytes for base in slabs]
        for name in ("_send_counters", "_wait_counters", "_wait_done"):
            setattr(runtime, name, _Tensor(space, pcie_dma.FLAG_SLOTS, torch.int32))
        runtime._bf16_stage = _Tensor(space, shard_elems * 2, torch.uint8)
        runtime._copy_stream, runtime._flag_stream = machine.streams["copy"], machine.streams["flag"]
        runtime._piece_events = [emulator._Event(machine) for _ in range(pcie_dma.MAX_PIECES)]
        runtime._copied_events = [emulator._Event(machine) for _ in range(2 * (WORLD - 1) * pcie_dma.MAX_PIECES)]
        runtime._input_ready, runtime._ag_ready = emulator._Event(machine), emulator._Event(machine)
        runtime._kernels, runtime._bf16_first_hop = machine, first_hop
        runtime._pick_pieces = lambda *args, **kwargs: pieces
        source = _Tensor(space, WORLD * shard_elems, torch.float32)
        output = source if in_place else _Tensor(space, WORLD * shard_elems, torch.float32)

        def view(address, elems, dtype, device, machine=machine):
            class View:
                def copy_(self, other):
                    machine.emit("host", lambda: _values(space, address, elems, dtype).copy_(
                        _values(space, other.address, other.elems, other.dtype).to(dtype)))

            made = View()
            made.address, made.elems, made.dtype = address, elems, dtype
            return made

        def add_wire(destination, addend, wire, elems, device, machine=machine):
            machine.emit("host", lambda: space.f32(destination, elems).copy_(
                space.f32(addend, elems) + space.bf16(wire, elems).float()))

        monkeypatch.setattr(pcie_dma, "_device_view", view)
        monkeypatch.setattr(pcie_dma, "_add_bf16_into_fp32", add_wire)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, machine=machine: machine.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        for index in range(rounds):
            machine.main.operations.append(("host", lambda index=index, rank=rank, source=source:
                                            space.f32(source.address, WORLD * shard_elems).copy_(inputs[index, rank].reshape(-1))))
            runtime._run_bf16_gather(source, output)
            machine.main.operations.append(("host", lambda index=index, rank=rank, output=output:
                                            results[index].__setitem__(rank, space.f32(output.address, WORLD * shard_elems).clone())))
        machines.append(machine)
    emulator._execute(space, machines, seed=seed)
    return inputs, results, machines


def _reference(inputs):
    """Chunk c: the FP32 chain in ring order from rank c, rounded to BF16 once."""
    chunks = []
    for chunk in range(WORLD):
        total = inputs[chunk, chunk].clone()
        for step in range(1, WORLD):
            total = total + inputs[(chunk + step) % WORLD, chunk]
        chunks.append(total.to(torch.bfloat16).float())
    return torch.stack(chunks).reshape(-1)


@pytest.mark.parametrize("pieces", [1, 2, 4])
@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bf16_first_hop_reduces_to_the_bits_of_the_fp32_first_hop(monkeypatch, pieces, in_place, seed):
    inputs, plain, plain_machines = _run(monkeypatch, False, pieces, in_place, seed)
    _, first, first_machines = _run(monkeypatch, True, pieces, in_place, seed)
    for index in range(inputs.shape[0]):
        expected = _reference(inputs[index])
        for rank in range(WORLD):
            # The same FP32 values on every rank in both modes: the owner of a chunk keeps its unrounded sum.
            assert torch.equal(first[index][rank].view(torch.int32), plain[index][rank].view(torch.int32)), (index, rank)
            # What the caller's BF16 store observes is the chain rounded once.
            assert torch.equal(first[index][rank].to(torch.bfloat16).view(torch.int16),
                               expected.to(torch.bfloat16).view(torch.int16)), (index, rank)
    shard_bytes = inputs.shape[-1] * 4
    for machines, hop_bytes in ((plain_machines, shard_bytes // pieces), (first_machines, shard_bytes // pieces // 2)):
        copies = [operation for operation in machines[3].streams["copy"].operations if operation[0] == "copy"]
        per_reduction = 2 * (WORLD - 1) * pieces
        assert len(copies) == 2 * per_reduction
        # The first hop of every reduction: FP32 pieces, or the same values as BF16.
        assert all(operation[3] == hop_bytes for operation in copies[:pieces] + copies[per_reduction:per_reduction + pieces])
        # Later reduce hops carry FP32 partial sums either way.
        assert all(operation[3] == shard_bytes // pieces for operation in copies[pieces:(WORLD - 1) * pieces])


def test_bf16_first_hop_keeps_its_stage_until_the_first_copies_are_done(monkeypatch):
    """The stage that feeds the first hop is rewritten for the broadcast only after those copies."""
    pieces = 2
    _, _, machines = _run(monkeypatch, True, pieces, True, seed=5, rounds=1)
    for machine in machines:
        records = [operation[1] for operation in machine.streams["copy"].operations if operation[0] == "record"]
        last_first_hop_copy = records[pieces - 1]
        waits = [index for index, operation in enumerate(machine.main.operations)
                 if operation == ("wait_event", last_first_hop_copy)]
        assert len(waits) == 1
        # Host operations of one reduction on the main stream: load, first-hop stage (per piece), first-hop
        # adds, ..., broadcast stage (per piece), ...; the wait precedes the broadcast stage conversions.
        hosts = [index for index, operation in enumerate(machine.main.operations) if operation[0] == "host"]
        stage_writes_after_wait = [index for index in hosts if index > waits[0]]
        assert len([index for index in hosts if index < waits[0]]) >= 1 + pieces + pieces
        assert len(stage_writes_after_wait) >= pieces


def test_bf16_first_hop_belongs_to_the_bf16_broadcast_ring():
    import inspect

    from b12x.comm.pcie import pcie_dma
    from b12x.comm.pcie._dma_preparation import query_from_runtime
    from types import SimpleNamespace

    assert inspect.signature(pcie_dma.PCIeDmaAllReduce.__init__).parameters["bf16_first_hop"].default is False
    base = dict(max_bytes=1 << 20, _fp8="", _bf16_gather=True, _pieces_override=0, _a2a_chunks_override=0,
                world_size=9, rank=2, shard_capacity=4096, _fp8_stage=None, _fp8_stage_stride=0,
                _bf16_stage=torch.zeros(64, dtype=torch.uint8), tree=None)
    plain = query_from_runtime(SimpleNamespace(**base), surface="DmaAllReduce.all_reduce", call={})
    first = query_from_runtime(SimpleNamespace(**base, _bf16_first_hop=True), surface="DmaAllReduce.all_reduce", call={})
    # The declaration is part of the plan's identity only when it is made.
    assert "bf16_first_hop" not in plain.call and first.call["bf16_first_hop"] is True
