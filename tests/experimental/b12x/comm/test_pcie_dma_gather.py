"""Ring all-gather on the copy engines, emulated on the CPU.

The runtime's own stream program runs for nine simulated ranks under random interleavings of their
streams, with the semantics of test_pcie_dma_tree's emulator (stream order, events, flags, copies into
the destination's scratch). The checks are the ones a GPU run cannot show directly: no interleaving
stalls, no slot is forwarded before it arrived, and a second gather does not disturb a rank that is
still reading the first.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_pcie_dma_tree_emulator", pathlib.Path(__file__).with_name("test_pcie_dma_tree.py"))
emulator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emulator)

WORLD = 9


def _ranks(monkeypatch, capacity):
    from b12x.comm.pcie import pcie_dma

    space = emulator._Space()
    flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
    slabs = [space.allocate(flags_bytes + 2 * (WORLD - 1) * capacity) for _ in range(WORLD)]
    machines, runtimes = [], []
    for rank in range(WORLD):
        machine = emulator._Machine(space)
        machine.streams = {name: emulator._Stream(machine, name) for name in ("copy", "flag")}
        runtime = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        runtime.world_size, runtime.rank, runtime.device = WORLD, rank, "fake"
        runtime._tree, runtime._closed, runtime.shard_capacity = None, False, capacity
        runtime._flags_base = list(slabs)
        runtime._scratch_base = [base + flags_bytes for base in slabs]
        runtime._send_counters = emulator._FakeTensor(space, pcie_dma.FLAG_SLOTS * 4, torch.int32)
        runtime._wait_counters = emulator._FakeTensor(space, pcie_dma.FLAG_SLOTS * 4, torch.int32)
        runtime._copy_stream, runtime._flag_stream = machine.streams["copy"], machine.streams["flag"]
        runtime._piece_events = [emulator._Event(machine) for _ in range(pcie_dma.MAX_PIECES)]
        runtime._copied_events = [emulator._Event(machine)
                                  for _ in range(2 * (WORLD - 1) * pcie_dma.MAX_PIECES)]
        runtime._input_ready = emulator._Event(machine)
        runtime._kernels = machine
        machines.append(machine)
        runtimes.append(runtime)
    return pcie_dma, space, machines, runtimes


def _issue(monkeypatch, pcie_dma, machine, body):
    monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None: machine.main)
    monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
    body()


@pytest.mark.parametrize("pieces", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_ring_gather_fills_every_slot_of_every_rank_in_consecutive_gathers(monkeypatch, pieces, seed):
    capacity, nbytes, rounds = 512, 64 * pieces, 3
    pcie_dma, space, machines, runtimes = _ranks(monkeypatch, capacity)
    generator = np.random.default_rng(100 * pieces + seed)
    segments = generator.integers(0, 256, size=(rounds, WORLD, nbytes), dtype=np.uint8)
    seen = [[None] * WORLD for _ in range(rounds)]
    for rank, (machine, runtime) in enumerate(zip(machines, runtimes)):
        def program(rank=rank, machine=machine, runtime=runtime):
            own = runtime._scratch_ptr(rank, WORLD - 1)
            for index in range(rounds):
                def write(index=index):
                    space.view(own, nbytes)[:] = segments[index, rank]

                def read(index=index):
                    seen[index][rank] = np.stack([
                        space.view(runtime._scratch_ptr(rank, slot), nbytes).copy() for slot in range(WORLD)])

                machine.main.operations.append(("host", write))
                runtime._run_gather(nbytes, pieces=pieces)
                machine.main.operations.append(("host", read))
                runtime._release_gather()

        _issue(monkeypatch, pcie_dma, machine, program)
    copies = [operation for operation in machines[4].streams["copy"].operations if operation[0] == "copy"]
    assert len(copies) == rounds * (WORLD - 1) * pieces
    # Every copy of rank 4 lands in rank 5's scratch and reads rank 4's own.
    first, last = runtimes[4]._scratch_ptr(5, 0), runtimes[4]._scratch_ptr(5, WORLD - 1)
    assert all(first <= operation[1] < last for operation in copies)
    assert all(runtimes[4]._scratch_ptr(4, 1) <= operation[2] < runtimes[4]._scratch_ptr(4, WORLD)
               for operation in copies)
    emulator._execute(space, machines, seed=seed)
    for index in range(rounds):
        for rank, runtime in enumerate(runtimes):
            for slot in range(WORLD):
                owner = runtime.gather_rank(slot)
                assert owner == (rank + 1 + slot) % WORLD
                assert np.array_equal(seen[index][rank][slot], segments[index, owner]), (index, rank, slot)


def test_ring_gather_rejects_segments_the_channel_cannot_carry(monkeypatch):
    pcie_dma, space, machines, runtimes = _ranks(monkeypatch, 512)
    runtime, machine = runtimes[0], machines[0]

    def program():
        for nbytes, pieces in ((0, 1), (528, 1), (72, 1), (96, 4), (64, 3)):
            with pytest.raises(ValueError):
                runtime._run_gather(nbytes, pieces=pieces)
        runtime._tree = object()
        with pytest.raises(ValueError):
            runtime._run_gather(64)

    _issue(monkeypatch, pcie_dma, machine, program)
    assert not machine.main.operations
