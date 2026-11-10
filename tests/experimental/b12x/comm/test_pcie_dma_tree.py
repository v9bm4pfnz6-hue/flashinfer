"""Tree all-reduce on the copy engines: the schedule, emulated on the CPU.

The emulator runs the per-rank stream programs of _dma_tree with the semantics the runtime gives
them (stream order, events, flags, copies into the destination's scratch) and the BF16/FP32
arithmetic of the accumulate kernel. It checks what a GPU run cannot show directly: that no
interleaving of ranks and streams stalls, that no scratch region is read before its copy or shared
by two transfers, and that every rank ends with the owner's single rounding of every chunk.
"""

from __future__ import annotations

import random

import pytest
import torch

from b12x.comm.pcie import _dma_tree as tree

ROUTING = tree.PAIRS_2_2_2_2_1


def _emulate(plan, inputs, *, seed, in_place):
    """inputs: [rank, chunk, piece, elems] BF16. Returns the outputs in the same layout."""
    world = plan.world_size
    outputs = inputs if in_place else torch.full_like(inputs, float("nan"))
    scratch = [dict() for _ in range(world)]
    stage = [dict() for _ in range(world)]
    flags = [set() for _ in range(world)]
    waited = [set() for _ in range(world)]
    events = [set() for _ in range(world)]
    programs = [tree.program(plan, rank)[0] for rank in range(world)]
    cursors = {(rank, name): 0 for rank in range(world) for name in tree.STREAMS}

    def source_values(rank, source):
        if source[0] == "input":
            return inputs[rank, source[1], source[2]].clone()
        if source[0] == "output":
            return outputs[rank, source[1], source[2]].clone()
        if source[0] == "stage":
            return stage[rank][source[1]].clone()
        edge = source[1]
        assert edge.destination == rank
        return scratch[rank][edge.offset].clone()

    def ready(rank, operation):
        kind = operation[0]
        if kind == "copy":
            return operation[3] is None or operation[3] in events[rank]
        if kind == "flag":
            return operation[2] in events[rank]
        if kind == "wait":
            return operation[1].slot in flags[rank]
        return True

    def run(rank, operation):
        kind = operation[0]
        if kind == "copy":
            _, edge, source, _, copied = operation
            assert edge.source == rank
            values = source_values(rank, source)
            assert values.dtype == (torch.float32 if edge.kind == tree.PARTIAL else torch.bfloat16)
            # A region belongs to one transfer of this reduction.
            assert edge.offset not in scratch[edge.destination]
            scratch[edge.destination][edge.offset] = values
            events[rank].add(copied)
        elif kind == "flag":
            flags[operation[1].destination].add(operation[1].slot)
        elif kind == "wait":
            waited[rank].add(operation[1].slot)
        elif kind == "accumulate":
            _, item, summed = operation
            assert all(edge.slot in waited[rank] for edge in (*item.incoming, *item.partials))
            arrived = {edge.source: scratch[rank][edge.offset] for edge in item.incoming}
            total = None
            for source in item.raws:
                values = (inputs[rank, item.chunk, item.piece] if source == rank else arrived[source]).float()
                total = values if total is None else total + values
            for edge in item.partials:
                total = total + scratch[rank][edge.offset]
            if item.final:
                outputs[rank, item.chunk, item.piece] = total.to(torch.bfloat16)
            else:
                assert item.stage not in stage[rank]
                stage[rank][item.stage] = total
            events[rank].add(summed)
        elif kind == "mark":
            events[rank].add(operation[1])
        elif kind == "place":
            receive = operation[1]
            assert receive.edge.slot in waited[rank]
            outputs[rank, receive.chunk, receive.piece] = scratch[rank][receive.edge.offset]
        else:
            raise AssertionError(kind)

    order = random.Random(seed)
    remaining = sum(len(programs[rank][name]) for rank in range(world) for name in tree.STREAMS)
    while remaining:
        keys = [key for key in cursors if cursors[key] < len(programs[key[0]][key[1]])]
        order.shuffle(keys)
        progressed = False
        for rank, name in keys:
            operation = programs[rank][name][cursors[(rank, name)]]
            if ready(rank, operation):
                run(rank, operation)
                cursors[(rank, name)] += 1
                remaining -= 1
                progressed = True
                if order.random() < 0.5:
                    break
        assert progressed, "the stream programs stall"
    return outputs


@pytest.mark.parametrize("pieces", [1, 2, 4])
@pytest.mark.parametrize("in_place", [False, True])
def test_tree_schedule_reduces_every_chunk_with_one_rounding(pieces, in_place):
    plan = tree.schedule(ROUTING, pieces)
    world, elems = plan.world_size, 24
    generator = torch.Generator().manual_seed(100 + pieces)
    values = (torch.randn((world, world, pieces, elems), generator=generator) * 3).to(torch.bfloat16)
    values[0, 3, 0, 0] = float("inf")
    values[5, 7, pieces - 1, 1] = float("nan")
    exact = values.double().sum(0).to(torch.bfloat16)
    for seed in range(4):
        outputs = _emulate(plan, values.clone(), seed=seed, in_place=in_place)
        # Every rank holds the same bits for every chunk.
        for rank in range(1, world):
            assert torch.equal(outputs[rank].view(torch.int16), outputs[0].view(torch.int16))
        result = outputs[0]
        finite = torch.isfinite(exact)
        assert torch.equal(torch.isnan(result), torch.isnan(exact))
        assert torch.equal(torch.isinf(result), torch.isinf(exact))
        # FP32 accumulation and one rounding: within one BF16 unit of the exact sum's rounding.
        magnitude = exact.float().abs()
        above = torch.nextafter(magnitude.to(torch.bfloat16), torch.tensor(float("inf"), dtype=torch.bfloat16))
        unit = above.float() - magnitude
        assert bool(((result.float() - exact.float()).abs()[finite] <= unit[finite]).all())
        assert float((result.view(torch.int16) == exact.view(torch.int16))[finite].float().mean()) > 0.9


@pytest.mark.parametrize("pieces", [1, 2, 4])
def test_tree_schedule_resources_and_roles(pieces):
    plan = tree.schedule(ROUTING, pieces)
    world = plan.world_size
    units = world * pieces
    for rank in range(world):
        incoming = [edge for edge in plan.edges if edge.destination == rank]
        assert sorted(edge.slot for edge in incoming) == list(range(len(incoming)))
        spans = sorted((edge.offset, edge.offset + tree.UNITS[edge.kind]) for edge in incoming)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert spans[-1][1] == plan.scratch_units[rank]
        # A rank receives the result of every chunk it does not own, once per piece.
        assert sorted((r.chunk, r.piece) for r in plan.receives[rank]) == sorted(
            (chunk, piece) for chunk in range(world) for piece in range(pieces) if chunk != rank)
        finals = [item for item in plan.accumulates[rank] if item.final]
        assert [(item.chunk, item.piece) for item in finals] == [(rank, piece) for piece in range(pieces)]
        # Sums for other owners precede this rank's own.
        kinds = [item.final for item in plan.accumulates[rank]]
        assert kinds == sorted(kinds)
    # Every contribution enters exactly one sum of its chunk and piece.
    for chunk in range(world):
        for piece in range(pieces):
            sums = [item for rank in range(world) for item in plan.accumulates[rank]
                    if (item.chunk, item.piece) == (chunk, piece)]
            assert sorted(source for item in sums for source in item.raws) == list(range(world))
            assert sum(item.final for item in sums) == 1
    # Traffic per rank in tensors: at most 2 in and 2.34 out (rank 8 forwards most results).
    sent = [0] * world
    for edge in plan.edges:
        sent[edge.source] += tree.UNITS[edge.kind]
    assert max(plan.scratch_units) / units <= 2.0 + 1e-9
    assert max(sent) / units <= 2.34
    assert set(plan.variants) == {(2, 2, True), (3, 0, False), (4, 0, False), (9, 0, True)}


def test_topology_rejects_incomplete_or_cyclic_trees():
    good = tree.PAIRS_2_2_2_2_1
    with pytest.raises(ValueError, match="one parent"):
        tree.Topology("bad", good.groups, good.reduce[:4] + (((0, 3), (1, 2)),), good.broadcast)
    with pytest.raises(ValueError, match="lead every group"):
        tree.Topology("bad", good.groups, good.reduce[:4] + (((0, 1), (1, 0), (2, 4), (3, 4)),), good.broadcast)
    with pytest.raises(ValueError, match="unknown all-reduce tree"):
        tree.topology("4+4")


# ---------------------------------------------------------------------------------------------
# The runtime itself (PCIeDmaAllReduce._run_tree) on simulated devices: its copies, flags, sums and
# placements are recorded per stream with the addresses and counters it computes, then executed in
# random interleavings over a simulated address space. An access outside an allocation fails.
# ---------------------------------------------------------------------------------------------


class _Space:
    def __init__(self):
        self.blocks, self.cursor = [], 1 << 24

    def allocate(self, nbytes):
        import numpy as np

        base = (self.cursor + 4095) // 4096 * 4096
        self.cursor = base + nbytes + 65536
        self.blocks.append((base, np.zeros(nbytes, dtype=np.uint8)))
        return base

    def view(self, address, nbytes):
        for base, block in self.blocks:
            if base <= address and address + nbytes <= base + block.size:
                return block[address - base : address - base + nbytes]
        raise AssertionError(f"access outside every allocation: {address:#x} + {nbytes}")

    def bf16(self, address, elems):
        import numpy as np

        return torch.from_numpy(self.view(address, elems * 2).view(np.int16)).view(torch.bfloat16)

    def f32(self, address, elems):
        import numpy as np

        return torch.from_numpy(self.view(address, elems * 4).view(np.float32))


class _FakeTensor:
    def __init__(self, space, nbytes, dtype=torch.bfloat16):
        self.space, self.nbytes, self.dtype = space, nbytes, dtype
        self.address = space.allocate(nbytes)

    def data_ptr(self):
        return self.address

    def numel(self):
        return self.nbytes // (2 if self.dtype == torch.bfloat16 else 4 if self.dtype == torch.int32 else 1)


class _Stream:
    def __init__(self, machine, name):
        self.machine, self.name, self.operations = machine, name, []

    def wait_event(self, event):
        self.operations.append(("wait_event", event.last))

    def wait_stream(self, other):
        self.operations.append(("join", other, len(other.operations)))


class _Event:
    def __init__(self, machine):
        self.machine, self.last = machine, None

    def record(self, stream):
        self.machine.records += 1
        # Events belong to one rank: an identity includes its machine.
        self.last = (id(self.machine), self.machine.records)
        stream.operations.append(("record", self.last))


class _Machine:
    """Stands in for torch.cuda and the DMA kernels of one rank while its operations are issued."""

    def __init__(self, space):
        self.space, self.records, self.current = space, 0, []
        self.main = _Stream(self, "main")

    def stream(self, stream):
        machine = self

        class Scope:
            def __enter__(self):
                machine.current.append(stream)

            def __exit__(self, *exc):
                machine.current.pop()

        return Scope()

    def emit(self, *operation):
        (self.current[-1] if self.current else self.main).operations.append(operation)

    # DmaKernels
    def dma_copy(self, destination, source, nbytes):
        self.emit("copy", destination, source, nbytes)

    def dma_set_flag(self, flag, counter):
        self.emit("set_flag", flag, counter)

    def dma_wait_flag(self, flag, counter):
        self.emit("wait_flag", flag, counter)

    def dma_tree_accum(self, destination, values, partials, elems, final):
        self.emit("accumulate", destination, tuple(values), tuple(partials), elems, final)


def _execute(space, machines, seed):
    """Run every rank's streams to completion in a random interleaving."""
    import numpy as np

    order = random.Random(seed)
    done_records = set()
    streams = [(rank, stream) for rank, machine in enumerate(machines)
               for stream in (machine.main, *machine.streams.values())]
    cursor = {id(stream): 0 for _, stream in streams}
    armed = {}

    def u32(address):
        return space.view(address, 4).view(np.uint32)

    def step(stream):
        operation = stream.operations[cursor[id(stream)]]
        kind = operation[0]
        if kind == "wait_event":
            if operation[1] is not None and operation[1] not in done_records:
                return False
        elif kind == "record":
            done_records.add(operation[1])
        elif kind == "join":
            if cursor[id(operation[1])] < operation[2]:
                return False
        elif kind == "copy":
            _, destination, source, nbytes = operation
            space.view(destination, nbytes)[:] = space.view(source, nbytes).copy()
        elif kind == "set_flag":
            counter = u32(operation[2])
            counter[0] += 1
            u32(operation[1])[0] = counter[0]
        elif kind == "wait_flag":
            key = (id(stream), cursor[id(stream)])
            if key not in armed:
                counter = u32(operation[2])
                counter[0] += 1
                armed[key] = int(counter[0])
            if int(u32(operation[1])[0]) - armed[key] < 0:
                return False
        elif kind == "accumulate":
            _, destination, values, partials, elems, final = operation
            total = space.bf16(values[0], elems).float()
            for address in values[1:]:
                total = total + space.bf16(address, elems).float()
            for address in partials:
                total = total + space.f32(address, elems)
            if final:
                space.bf16(destination, elems).copy_(total.to(torch.bfloat16))
            else:
                space.f32(destination, elems).copy_(total)
        elif kind == "place":
            _, destination, source, nbytes = operation
            space.view(destination, nbytes)[:] = space.view(source, nbytes).copy()
        elif kind == "host":
            operation[1]()
        else:
            raise AssertionError(kind)
        cursor[id(stream)] += 1
        return True

    remaining = sum(len(stream.operations) for _, stream in streams)
    while remaining:
        candidates = [stream for _, stream in streams if cursor[id(stream)] < len(stream.operations)]
        order.shuffle(candidates)
        progressed = 0
        for stream in candidates:
            if step(stream):
                progressed += 1
                remaining -= 1
                if order.random() < 0.4:
                    break
        assert progressed, "the runtime's streams stall"


@pytest.mark.parametrize("pieces,in_place,small_scratch", [(1, False, False), (2, False, False), (2, True, False),
                                                           (4, True, False), (2, False, True), (1, True, True)])
def test_tree_runtime_operations_reduce_two_consecutive_tensors(monkeypatch, pieces, in_place, small_scratch):
    from b12x.comm.pcie import pcie_dma

    world, chunk_elems = 9, 64 * pieces
    elements = world * chunk_elems
    plan = tree.schedule(ROUTING, pieces)
    space = _Space()
    flags_bytes = pcie_dma.FLAG_SLOTS * pcie_dma.FLAG_STRIDE
    piece_elems = chunk_elems // pieces
    # A small arena forces the reduction into several element intervals.
    interval_elems = 16 if small_scratch else piece_elems
    arena = max(plan.scratch_units) * interval_elems * 2
    stage_bytes = max(max(plan.stage_units), 1) * interval_elems * 2
    tree_flags = len(pcie_dma.TREE_PIECES) * pcie_dma.TREE_BANK_SLOTS * pcie_dma.FLAG_STRIDE
    slabs = [space.allocate(flags_bytes + arena + tree_flags) for _ in range(world)]
    monkeypatch.setattr(pcie_dma, "TREE_MIN_PIECE_BYTES", 0)
    machines, tensors = [], []
    generator = torch.Generator().manual_seed(7 + pieces)
    rounds = [(torch.randn((world, elements), generator=generator) * 3).to(torch.bfloat16) for _ in range(2)]
    rounds[0][2, 5] = float("inf")
    rounds[1][7, elements - 1] = float("nan")
    results = [dict(), dict()]
    for rank in range(world):
        machine = _Machine(space)
        runtime = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        runtime.world_size, runtime.rank, runtime.device = world, rank, "fake"
        runtime._tree = ROUTING
        runtime._pieces_override = pieces
        runtime._tree_arena_bytes, runtime._tree_stage_bytes = arena, stage_bytes
        runtime._scratch_base = [base + flags_bytes for base in slabs]
        runtime._tree_flags_base = [base + flags_bytes + arena for base in slabs]
        runtime._tree_stage = _FakeTensor(space, stage_bytes, torch.uint8)
        runtime._tree_send_counters = _FakeTensor(space, len(pcie_dma.TREE_PIECES) * pcie_dma.TREE_BANK_SENDS * 4, torch.int32)
        runtime._tree_wait_counters = _FakeTensor(space, len(pcie_dma.TREE_PIECES) * pcie_dma.TREE_BANK_SLOTS * 4, torch.int32)
        machine.streams = {name: _Stream(machine, name) for name in tree.STREAMS if name != "main"}
        runtime._tree_streams = machine.streams
        runtime._tree_events = {count: [] for count in pcie_dma.TREE_PIECES}
        runtime._input_ready = _Event(machine)
        runtime._kernels = machine
        source = _FakeTensor(space, elements * 2)
        output = source if in_place else _FakeTensor(space, elements * 2)
        machines.append(machine)
        tensors.append((runtime, source, output))

    def view(address, elems, dtype, device):
        class View:
            def copy_(self, other):
                machines[view.rank].emit("place", address, other.address, elems * 2)

        made = View()
        made.address = address
        return made

    for rank, (runtime, source, output) in enumerate(tensors):
        machine = machines[rank]
        view.rank = rank
        monkeypatch.setattr(pcie_dma, "_device_view", view)
        monkeypatch.setattr(pcie_dma.torch.cuda, "current_stream", lambda device=None, machine=machine: machine.main)
        monkeypatch.setattr(pcie_dma.torch.cuda, "stream", machine.stream)
        monkeypatch.setattr(pcie_dma.torch.cuda, "Event", lambda machine=machine: _Event(machine))
        for index in range(2):
            def load(index=index, rank=rank, source=source):
                space.bf16(source.address, elements).copy_(rounds[index][rank])

            def save(index=index, rank=rank, output=output):
                results[index][rank] = space.bf16(output.address, elements).clone()

            machine.main.operations.append(("host", load))
            runtime._run_tree(source, output)
            machine.main.operations.append(("host", save))

    intervals = -(-piece_elems // interval_elems)
    assert sum(operation[0] == "accumulate" for operation in machines[0].main.operations) == (
        2 * intervals * len(plan.accumulates[0]))
    _execute(space, machines, seed=pieces * 10 + in_place)
    for index in range(2):
        expected = _emulate(plan, rounds[index].view(world, world, pieces, piece_elems).clone(), seed=1, in_place=False)
        for rank in range(world):
            assert torch.equal(results[index][rank].view(torch.int16),
                               expected[0].reshape(-1).view(torch.int16)), (index, rank)
