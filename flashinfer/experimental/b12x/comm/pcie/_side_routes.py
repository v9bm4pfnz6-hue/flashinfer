"""Routes of the nine-rank collectives that cross between the two sides of the rig once.

Topology contract: ranks 2i and 2i+1 share a leaf switch (i < 4) whose one uplink carries
everything that leaves the pair, rank 8 has a leaf of its own, and ranks 0-3 and 8 sit on one side
of a single link whose other side holds ranks 4-7.
"""

from __future__ import annotations

import functools

# The rank on the other side that receives each rank's segment: of all assignments, the one with
# the fewest copies on the busiest link and then the smallest sum of the three phases' busiest
# links.
SIDE_ENTRY_RANKS = (5, 4, 6, 7, 8, 2, 8, 1, 7)


@functools.cache
def side_entry_phases() -> tuple:
    """Every ``(sender, source, child)`` of the side-entry routes, by phase.

    First a rank sends its segment to its partner, to one member of each other leaf of its side
    and to one rank of the other side (``SIDE_ENTRY_RANKS``). Second, the members that
    received it forward it to their partners, and the entry rank forwards it to its partner and
    to one member of each other leaf of its side. Third, those members forward it to their
    partners. Every rank receives every other rank's segment once.
    """
    leaves = ((0, 1), (2, 3), (4, 5), (6, 7), (8,))
    sides = ((8, 0, 1, 2, 3), (4, 5, 6, 7))
    leaf_of = {rank: index for index, members in enumerate(leaves) for rank in members}
    phases = ([], [], [])
    for source in range(9):
        mine, other = sides if source in sides[0] else sides[::-1]

        def spread(origin, ranks):
            # The partner of `origin` and one member of each other leaf among `ranks`.
            targets = []
            for index, members in enumerate(leaves):
                here = [rank for rank in members if rank in ranks]
                if index == leaf_of[origin]:
                    targets += [rank for rank in here if rank != origin]
                elif here:
                    targets.append(here[(source + index) % len(here)])
            return targets

        entry = SIDE_ENTRY_RANKS[source]
        for target in (*spread(source, mine), entry):
            phases[0].append((source, source, target))
        for sender, targets, phase in (
            (source, spread(source, mine), 1),
            (entry, spread(entry, other), 2),
        ):
            for target in targets:
                if sender == entry:
                    phases[1].append((entry, source, target))
                if leaf_of[target] != leaf_of[sender]:
                    for partner in leaves[leaf_of[target]]:
                        if partner != target:
                            phases[phase].append((target, source, partner))
    return tuple(tuple(phase) for phase in phases)


@functools.cache
def side_entry_forwards(rank: int) -> frozenset:
    """Every ``(source, child)`` that ``rank`` sends in the side-entry routes, in any phase."""
    if not 0 <= rank < 9:
        raise ValueError("the side-entry routes are defined for ranks 0 to 8")
    return frozenset(
        (source, child)
        for phase in side_entry_phases()
        for sender, source, child in phase
        if sender == rank
    )
