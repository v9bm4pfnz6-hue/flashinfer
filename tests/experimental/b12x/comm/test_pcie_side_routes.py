"""Side-entry routes of the nine-rank rig, and the reduce-scatter that sends its results along them.

The protocol test runs a Python model of the kernel's three loops (publish, reduce owned packs,
gather the others) for every thread of nine ranks under random interleavings. A thread that finds
a slot empty waits, as the kernel polls. It checks what a GPU run cannot show directly: that no
interleaving stalls, that a slot is never written before its last reader emptied it, and that
every rank ends with every owner's result.
"""

from __future__ import annotations

import random

import pytest

from b12x.comm.pcie._side_routes import SIDE_ENTRY_RANKS, side_entry_forwards, side_entry_phases

WORLD = 9
SIDE = lambda rank: 1 if 4 <= rank < 8 else 0


def test_side_entry_routes_reach_every_rank_once_from_every_source():
    phases = side_entry_phases()
    assert len(phases) == 3
    held = {rank: {rank} for rank in range(WORLD)}
    for index, phase in enumerate(phases):
        arrived = []
        for sender, source, child in phase:
            assert source in held[sender] and (sender == source) == (index == 0)
            arrived.append((child, source))
        for child, source in arrived:
            assert source not in held[child]
            held[child].add(source)
    assert all(held[rank] == set(range(WORLD)) for rank in range(WORLD))
    for rank in range(WORLD):
        assert side_entry_forwards(rank) == {
            (source, child) for phase in phases for sender, source, child in phase if sender == rank}
        assert SIDE(SIDE_ENTRY_RANKS[rank]) != SIDE(rank)
    with pytest.raises(ValueError, match="ranks 0 to 8"):
        side_entry_forwards(9)


def _owner(index, size_packs):
    part = size_packs // WORLD
    return min(index // part, WORLD - 1) if part > 0 else WORLD - 1


def _thread(rank, first, stride, size_packs, inputs, outputs, slots, routed, stats):
    forwards = side_entry_forwards(rank) if routed else None

    def write(target, sender, index, value, result):
        # A slot is free again only after the reader of the previous launch emptied it.
        assert slots[target].get((sender, index)) is None, (rank, target, sender, index)
        slots[target][(sender, index)] = value
        stats["progress"] += 1
        if result and SIDE(rank) != SIDE(target):
            stats["crossings"][index] = stats["crossings"].get(index, 0) + 1

    for index in range(first, size_packs, stride):
        write(_owner(index, size_packs), rank, index, inputs[rank][index], False)
        yield
    for index in range(first, size_packs, stride):
        if _owner(index, size_packs) != rank:
            continue
        while any(slots[rank].get((peer, index)) is None for peer in range(WORLD)):
            yield
        total = sum(slots[rank][((rank + step) % WORLD, index)] for step in range(WORLD))
        for peer in range(WORLD):
            slots[rank][(peer, index)] = None
        outputs[rank][index] = total
        targets = ([peer for peer in range(WORLD) if peer != rank] if forwards is None
                   else [child for source, child in sorted(forwards) if source == rank])
        for peer in targets:
            write(peer, rank, index, total, True)
        yield
    for index in range(first, size_packs, stride):
        owner = _owner(index, size_packs)
        if owner == rank:
            continue
        while slots[rank].get((owner, index)) is None:
            yield
        value = slots[rank][(owner, index)]
        if forwards is not None:
            for source, child in sorted(forwards):
                if source == owner:
                    write(child, owner, index, value, True)
        outputs[rank][index] = value
        slots[rank][(owner, index)] = None
        stats["progress"] += 1
        yield


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("size_packs,threads", [(70, 8), (9, 4), (5, 4), (144, 16)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduce_scatter_with_routed_results_completes_under_any_interleaving(routed, size_packs, threads, seed):
    generator = random.Random(1000 * size_packs + 10 * seed + routed)
    slots = [dict() for _ in range(WORLD)]
    for launch in range(2):
        inputs = [[generator.randrange(1, 1 << 20) for _ in range(size_packs)] for _ in range(WORLD)]
        outputs = [[None] * size_packs for _ in range(WORLD)]
        stats = {"progress": 0, "crossings": {}}
        live = [_thread(rank, first, threads, size_packs, inputs, outputs, slots, routed, stats)
                for rank in range(WORLD) for first in range(threads)]
        while live:
            before, finished = stats["progress"], []
            generator.shuffle(live)
            for thread in live:
                try:
                    next(thread)
                except StopIteration:
                    finished.append(thread)
            live = [thread for thread in live if thread not in finished]
            assert not live or finished or stats["progress"] > before, "every thread waits: the protocol stalled"
        expected = [sum(inputs[rank][index] for rank in range(WORLD)) for index in range(size_packs)]
        assert all(outputs[rank] == expected for rank in range(WORLD))
        assert all(value is None for table in slots for value in table.values())
        for index in range(size_packs):
            owner = _owner(index, size_packs)
            # A result crosses between ranks 0-3,8 and ranks 4-7 once when routed, and once per rank of
            # the other side when the owner pushes it to every peer.
            assert stats["crossings"][index] == (1 if routed else 4 if SIDE(owner) == 0 else 5)


def test_routed_result_launch_is_a_nine_rank_reduce_scatter():
    from b12x.comm.pcie._oneshot_cute import _OneshotLaunch

    for rank in range(WORLD):
        launch = _OneshotLaunch("bfloat16", world_size=9, rank=rank, stage_input=True, device_slot_selection=True,
                                slot_bias=0, transport="remote_push_rsag_side", threads=256)
        assert launch._rsag and launch._side_forwards == side_entry_forwards(rank)
        assert any(source == rank for source, _ in launch._side_forwards)
    plain = _OneshotLaunch("bfloat16", world_size=9, rank=0, stage_input=True, device_slot_selection=True,
                           slot_bias=0, transport="remote_push_rsag", threads=256)
    assert plain._rsag and not plain._side_forwards
    full = _OneshotLaunch("bfloat16", world_size=9, rank=0, stage_input=True, device_slot_selection=True,
                          slot_bias=0, transport="remote_push_full", threads=256)
    assert not full._rsag and not full._side_forwards
    with pytest.raises(ValueError, match="nine ranks"):
        _OneshotLaunch("bfloat16", world_size=8, rank=0, stage_input=True, device_slot_selection=True,
                       slot_bias=0, transport="remote_push_rsag_side", threads=256)
    with pytest.raises(ValueError, match="requires staged BF16 input"):
        _OneshotLaunch("float16", world_size=9, rank=0, stage_input=True, device_slot_selection=True,
                       slot_bias=0, transport="remote_push_rsag_side", threads=256)
