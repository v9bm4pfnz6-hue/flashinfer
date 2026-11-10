"""Schedule of a single-rounding BF16 all-reduce routed over a PCIe switch tree.

The copy-engine ring forwards every owner chunk through all ranks: eight FP32
hops to reduce and eight BF16 hops to broadcast, 2.67 times the tensor on every
link per direction. On a tree of leaf switches most of those hops are not
needed. A rank's BF16 contribution can travel as it is (half an FP32 partial)
to a rank that sums several contributions, a partial sum crosses a shared link
once, and the rounded result is forwarded inside a leaf instead of entering it
twice.

This module holds the routing only. For every chunk (one per owner rank) and
piece it lists

- raw transfers: a rank's BF16 values to the rank that sums them;
- accumulations: FP32 sums of BF16 values and FP32 partials in a fixed order,
  stored as an FP32 partial (a combiner) or rounded once to BF16 (the owner);
- partial transfers from a combiner to the owner (or to another combiner);
- result transfers of the owner's BF16 rounding along a broadcast tree.

Every rank computes the same schedule. Only the owner rounds, so all ranks hold
identical bits. The arithmetic is FP32 accumulation with one rounding, as the
ring's; the grouping of the FP32 additions differs from the ring's order.

The routing of a topology is a pair of trees over its leaf groups per owner
group: the parent of a group in the reduce tree receives that group's values
(two BF16 values of a leaf pair, or one FP32 partial of a group that combines),
and the parent in the broadcast tree sends it the result. The tables below
were chosen by an integer program over all trees to minimise the busiest link
(lead/bin/tree_allreduce_milp.py in the Kimi K3.1 deployment tree).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

RAW, PARTIAL, RESULT = "raw", "partial", "result"
# Scratch and stage sizes are counted in BF16 pieces: an FP32 partial takes two.
UNITS = {RAW: 1, PARTIAL: 2, RESULT: 1}


@dataclass(frozen=True)
class Topology:
    """Leaf groups of ranks and, per owner group, the reduce and broadcast trees over the groups."""

    name: str
    groups: tuple[tuple[int, ...], ...]
    reduce: tuple[tuple[tuple[int, int], ...], ...]
    broadcast: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        ranks = sorted(rank for group in self.groups for rank in group)
        if ranks != list(range(len(ranks))) or any(not 1 <= len(g) <= 2 for g in self.groups):
            raise ValueError("a tree topology partitions ranks 0..W-1 into groups of one or two")
        count = len(self.groups)
        if len(self.reduce) != count or len(self.broadcast) != count:
            raise ValueError("a tree topology declares one reduce and one broadcast tree per group")
        for owner in range(count):
            for tree in (dict(self.reduce[owner]), dict(self.broadcast[owner])):
                if sorted(tree) != [g for g in range(count) if g != owner]:
                    raise ValueError("a routing tree gives every other group one parent")
                for group in tree:
                    seen, node = set(), group
                    while node != owner:
                        if node in seen or node not in tree:
                            raise ValueError("a routing tree must lead every group to the owner")
                        seen.add(node)
                        node = tree[node]

    @property
    def world_size(self) -> int:
        return sum(len(group) for group in self.groups)


def _trees(*maps: dict[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(tuple(sorted(tree.items())) for tree in maps)


# Nine ranks on four leaf pairs and one single rank: ranks (0,1), (2,3), (4,5), (6,7) share a
# leaf each and rank 8 has its own; the leaves of rank 8, (0,1) and (2,3) hang off one switch
# whose single uplink joins them to the leaves of (4,5) and (6,7). Group 0 is the single rank.
# Busiest link 2.33 tensors per direction (ring 2.67; bound over all mixtures of trees 2.24).
PAIRS_2_2_2_2_1 = Topology(
    name="2+2+2+2+1",
    groups=((8,), (0, 1), (2, 3), (4, 5), (6, 7)),
    reduce=_trees(
        {1: 0, 2: 0, 3: 0, 4: 0},
        {0: 2, 2: 1, 3: 1, 4: 3},
        {0: 1, 1: 2, 3: 4, 4: 2},
        {0: 4, 1: 3, 2: 1, 4: 3},
        {0: 3, 1: 2, 2: 4, 3: 4},
    ),
    broadcast=_trees(
        {1: 0, 2: 0, 3: 0, 4: 3},
        {0: 2, 2: 1, 3: 1, 4: 0},
        {0: 2, 1: 0, 3: 4, 4: 1},
        {0: 3, 1: 3, 2: 0, 4: 3},
        {0: 4, 1: 2, 2: 0, 3: 0},
    ),
)

TOPOLOGIES = {PAIRS_2_2_2_2_1.name: PAIRS_2_2_2_2_1}


def topology(name: str) -> Topology:
    try:
        return TOPOLOGIES[name]
    except KeyError:
        raise ValueError(f"unknown all-reduce tree topology {name!r}; known: {sorted(TOPOLOGIES)}") from None


@dataclass(frozen=True)
class Edge:
    """One transfer of a piece: its flag slot and scratch offset belong to the destination."""

    chunk: int
    piece: int
    kind: str
    source: int
    destination: int
    slot: int
    offset: int


@dataclass(frozen=True)
class Accumulate:
    """FP32 sum at `rank`: BF16 values of `raws` in ascending rank order (its own included), then
    the FP32 partials of `partials` in ascending source order."""

    rank: int
    chunk: int
    piece: int
    final: bool
    raws: tuple[int, ...]
    incoming: tuple[Edge, ...]
    partials: tuple[Edge, ...]
    stage: int
    outgoing: tuple[Edge, ...]


@dataclass(frozen=True)
class Receive:
    """Arrival of a result at a rank that does not own the chunk, and the copies it forwards."""

    rank: int
    chunk: int
    piece: int
    edge: Edge
    relays: tuple[Edge, ...]


@dataclass(frozen=True)
class Schedule:
    world_size: int
    pieces: int
    edges: tuple[Edge, ...]
    raw_sends: tuple[tuple[Edge, ...], ...]
    accumulates: tuple[tuple[Accumulate, ...], ...]
    receives: tuple[tuple[Receive, ...], ...]
    scratch_units: tuple[int, ...]
    stage_units: tuple[int, ...]
    flag_slots: int
    variants: tuple[tuple[int, int, bool], ...]


@lru_cache(maxsize=None)
def schedule(routing: Topology, pieces: int) -> Schedule:
    """Transfers and accumulations of one reduction split into `pieces` per owner chunk."""

    if pieces not in (1, 2, 4, 8):
        raise ValueError("a tree reduction uses one, two, four or eight pieces per chunk")
    groups = routing.groups
    world = routing.world_size
    group_of = {rank: index for index, group in enumerate(groups) for rank in group}
    slots, offsets, stages = [0] * world, [0] * world, [0] * world
    edges: list[Edge] = []

    def edge(chunk, piece, kind, source, destination):
        made = Edge(chunk, piece, kind, source, destination, slots[destination], offsets[destination])
        slots[destination] += 1
        offsets[destination] += UNITS[kind]
        edges.append(made)
        return made

    raw_sends = [[] for _ in range(world)]
    accumulates = [[] for _ in range(world)]
    receives = [[] for _ in range(world)]
    for piece in range(pieces):
        for chunk in range(world):
            owner_group = group_of[chunk]
            reduce_parent = dict(routing.reduce[owner_group])
            broadcast_parent = dict(routing.broadcast[owner_group])

            def active(group):
                # The member that sums, receives and forwards for its group alternates with the
                # chunk and the piece, which balances the two links of a pair.
                if group == owner_group:
                    return chunk
                members = groups[group]
                return members[(chunk + piece + group) % len(members)]

            def depth(group, parent):
                steps = 0
                while group != owner_group:
                    group, steps = parent[group], steps + 1
                return steps

            children = {group: sorted(g for g, p in reduce_parent.items() if p == group)
                        for group in range(len(groups))}
            # Deepest groups first, so a combiner's inputs exist before its partial is routed.
            order = sorted(range(len(groups)), key=lambda group: (-depth(group, reduce_parent), group))
            inputs = {group: ([], []) for group in range(len(groups))}
            for group in order:
                combiner = active(group)
                combines = group == owner_group or bool(children[group])
                if not combines:
                    target = active(reduce_parent[group])
                    for member in groups[group]:
                        made = edge(chunk, piece, RAW, member, target)
                        raw_sends[member].append(made)
                        inputs[reduce_parent[group]][0].append(made)
                    continue
                raws, partials = inputs[group]
                for member in groups[group]:
                    if member != combiner:
                        made = edge(chunk, piece, RAW, member, combiner)
                        raw_sends[member].append(made)
                        raws.append(made)
                raws.sort(key=lambda item: item.source)
                partials.sort(key=lambda item: item.source)
                final = group == owner_group
                outgoing = ()
                stage = -1
                if not final:
                    made = edge(chunk, piece, PARTIAL, combiner, active(reduce_parent[group]))
                    inputs[reduce_parent[group]][1].append(made)
                    outgoing = (made,)
                    stage = stages[combiner]
                    stages[combiner] += UNITS[PARTIAL]
                accumulates[combiner].append(Accumulate(
                    rank=combiner, chunk=chunk, piece=piece, final=final,
                    raws=tuple(sorted([combiner, *(item.source for item in raws)])),
                    incoming=tuple(raws), partials=tuple(partials), stage=stage, outgoing=outgoing))

            # Broadcast: the owner, then every group's receiving member, sends to its partner and
            # to the receiving member of each child group.
            send_children = {group: sorted(g for g, p in broadcast_parent.items() if p == group)
                             for group in range(len(groups))}

            def fan_out(group, sender):
                made = [edge(chunk, piece, RESULT, sender, member)
                        for member in groups[group] if member != sender]
                made += [edge(chunk, piece, RESULT, sender, active(child)) for child in send_children[group]]
                return made

            owner_edges = fan_out(owner_group, chunk)
            final = accumulates[chunk][-1]
            assert final.final and final.chunk == chunk and final.piece == piece
            accumulates[chunk][-1] = Accumulate(**{**final.__dict__, "outgoing": tuple(owner_edges)})
            pending = list(owner_edges)
            while pending:
                arrived = pending.pop(0)
                receiver = arrived.destination
                group = group_of[receiver]
                relays = fan_out(group, receiver) if receiver == active(group) and group != owner_group else []
                receives[receiver].append(Receive(receiver, chunk, piece, arrived, tuple(relays)))
                pending += relays

    variants = sorted({(len(item.raws), len(item.partials), item.final)
                       for rank in range(world) for item in accumulates[rank]})
    # A rank sums what combines for others before its own chunk, piece by piece: its own sum
    # waits for partials, which wait only for values every rank sends first.
    ordered = tuple(tuple(sorted(items, key=lambda item: (item.final, item.piece, item.chunk)))
                    for items in accumulates)
    return Schedule(world_size=world, pieces=pieces, edges=tuple(edges),
                    raw_sends=tuple(map(tuple, raw_sends)), accumulates=ordered,
                    receives=tuple(map(tuple, receives)), scratch_units=tuple(offsets),
                    stage_units=tuple(stages), flag_slots=max(slots), variants=tuple(variants))


# Streams of one rank. Copies of each kind queue on their own stream so that a copy waiting for
# a sum or an arrival does not hold back copies whose data is ready; every copy stream has a
# flag stream that publishes a copy's flag once that copy has completed.
COPY_STREAMS = ("raw", "partial", "result", "relay")
STREAMS = ("main", "receive", *COPY_STREAMS, *(name + "_flag" for name in COPY_STREAMS))


@lru_cache(maxsize=None)
def program(plan: Schedule, rank: int) -> tuple[dict[str, tuple[tuple, ...]], int]:
    """Operations of `rank` per stream, in issue order, and the number of events they use.

    Operations (events are indices into a per-channel pool, recorded on the issuing stream):
      ("copy", edge, source, after, event)  copy a piece into the destination's scratch once
                                            event `after` (None: the input) is complete; `source`
                                            is ("input", chunk, piece), ("stage", units),
                                            ("output", chunk, piece) or ("scratch", edge)
      ("flag", edge, after)                 publish the flag of `edge` after its copy
      ("wait", edge)                        wait for the flag of an incoming edge
      ("accumulate", item, event)           sum (see Accumulate), then record `event`
      ("mark", event)                       record an arrival
      ("place", receive)                    copy an arrived result from scratch into the output
    Every rank waits for results in the same (piece, chunk) order, and sums for other owners
    before it sums its own chunks, so the waits of all ranks form no cycle.
    """

    streams: dict[str, list[tuple]] = {name: [] for name in STREAMS}
    events = 0

    def event():
        nonlocal events
        events += 1
        return events - 1

    def send(stream, edge, source, after):
        copied = event()
        streams[stream].append(("copy", edge, source, after, copied))
        streams[stream + "_flag"].append(("flag", edge, copied))

    for edge in plan.raw_sends[rank]:
        send("raw", edge, ("input", edge.chunk, edge.piece), None)
    for item in plan.accumulates[rank]:
        for edge in (*item.incoming, *item.partials):
            streams["main"].append(("wait", edge))
        summed = event()
        streams["main"].append(("accumulate", item, summed))
        for edge in item.outgoing:
            if item.final:
                send("result", edge, ("output", item.chunk, item.piece), summed)
            else:
                send("partial", edge, ("stage", item.stage), summed)
    for receive in plan.receives[rank]:
        streams["receive"].append(("wait", receive.edge))
        if receive.relays:
            arrived = event()
            streams["receive"].append(("mark", arrived))
            for edge in receive.relays:
                send("relay", edge, ("scratch", receive.edge), arrived)
        streams["receive"].append(("place", receive))
    return {name: tuple(operations) for name, operations in streams.items()}, events


__all__ = ["Accumulate", "COPY_STREAMS", "Edge", "PAIRS_2_2_2_2_1", "Receive", "STREAMS", "Schedule",
           "TOPOLOGIES", "Topology", "program", "schedule", "topology"]
