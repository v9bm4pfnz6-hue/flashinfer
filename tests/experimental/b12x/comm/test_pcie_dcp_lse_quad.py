"""Nine-rank LSE reduce-scatter with one FP32 partial per destination between the quads.

The kernel cannot run without GPUs. These tests cover what it takes from tables: the roles of every
rank (who writes which rows into which group of whose staging, and behind which barrier), and the
arithmetic of the two-stage combination, emulated in FP32 in the kernel's statement order and driven
by the same role tables. The launcher itself is lowered offline for every rank and compared with the
pushed reduce-scatter on nine GPUs (lead/bin/topo_lse_bench.py).
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.comm.pcie._dcp_a2a_cute import lse_quad_roles

WORLD = 9
SIDE = lambda rank: 1 if 4 <= rank < 8 else 0


def test_quad_roles_deliver_every_partial_behind_a_shared_barrier():
    roles = [lse_quad_roles(rank) for rank in range(WORLD)]
    staged = [dict() for _ in range(WORLD)]  # group -> (kind, writer, head owner)
    crossings = {0: 0, 1: 0}
    barriers = [[0] * WORLD for _ in range(WORLD)]
    for phase, mask_name in (("first", "first_mask"), ("second", "second_mask")):
        for rank, role in enumerate(roles):
            mask = role[mask_name]
            # A barrier relation is symmetric, so both ranks of a pair pass the same barriers.
            for peer in range(WORLD):
                if mask >> peer & 1:
                    assert roles[peer][mask_name] >> rank & 1 and peer != rank
                    barriers[rank][peer] += 1
            for destination, head_owner, group in role[phase]:
                # The writer and the owner of the staging share the barrier that ends this phase.
                assert mask >> destination & 1 and destination != rank
                assert group not in staged[destination]
                staged[destination][group] = ("rows", rank, head_owner)
                crossings[SIDE(rank)] += SIDE(rank) != SIDE(destination)
        if phase == "first":
            # Before its second phase a combiner holds its quad peers' rows for the rank across.
            for rank, role in enumerate(roles[:8]):
                assert [staged[rank][group] for group in role["duty"]] == [
                    ("rows", peer, role["across"]) for peer in (rank ^ 1, rank ^ 2, rank ^ 3)]
    for rank, role in enumerate(roles[:8]):
        across = role["across"]
        assert across == rank ^ 4 and roles[across]["across"] == rank
        assert role["second_mask"] >> across & 1
        for group in role["partial"]:
            assert group not in staged[across]
            staged[across][group] = ("partial", rank, across)
        crossings[SIDE(rank)] += 2
    assert barriers == [list(row) for row in zip(*barriers)]
    for rank, role in enumerate(roles):
        # Every group of a staging is used once, so no write of a launch lands on another.
        assert sorted(staged[rank]) == ([*range(8)] if rank == 8 else [*range(WORLD)])
        covered = {rank}
        for term in role["terms"][1:]:
            if term[0] == "rows":
                kind, writer, head_owner = staged[rank][term[1]]
                assert (kind, head_owner) == ("rows", rank)
                covered.add(writer)
            else:
                first, second = (staged[rank][group] for group in term[1:])
                assert first == second == ("partial", rank ^ 4, rank)
                covered |= {rank ^ 4, rank ^ 5, rank ^ 6, rank ^ 7}
        assert role["terms"][0] == ("own",) and covered == set(range(WORLD))
    assert roles[8]["across"] is None and roles[8]["second"] == () and roles[8]["second_mask"] == 0
    # Row units over the link between ranks 0-3,8 and ranks 4-7 (an FP32 partial counts two):
    # 12 in each direction where the pushed reduce-scatter sends 20.
    assert crossings == {0: 12, 1: 12}
    with pytest.raises(ValueError, match="ranks 0 to 8"):
        lse_quad_roles(9)


def _weights(lses):
    """The kernel's weights: exp(LSE - maximum) for finite LSEs, zero otherwise; maximum and sum."""
    stacked = torch.stack(lses)
    finite = torch.isfinite(stacked)
    clean = torch.where(finite, stacked, torch.full_like(stacked, -math.inf))
    maximum = clean.max(0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.where(finite, torch.exp(clean - maximum), torch.zeros_like(clean))
    total = torch.zeros_like(maximum)
    for weight in weights:
        total = total + weight
    return weights, maximum, total


def _combine(values, lses):
    """Weighted mean in term order; rows of zero weight are not read."""
    weights, maximum, total = _weights(lses)
    inverse = 1.0 / torch.clamp(total, min=1.0e-10)
    result = torch.zeros_like(values[0])
    for value, weight in zip(values, weights):
        normalized = (weight * inverse).unsqueeze(-1)
        result = torch.where(normalized != 0, normalized * value + result, result)
    return result, maximum, total


def _pushed(partials, lses, heads):
    outputs = []
    for rank in range(WORLD):
        own = slice(rank * heads, (rank + 1) * heads)
        order = [(rank + step) % WORLD for step in range(WORLD)]
        result, _, _ = _combine([partials[source][:, own].float() for source in order],
                                [lses[source][:, own] for source in order])
        outputs.append(result.to(torch.bfloat16))
    return outputs


def _quad(partials, lses, heads):
    roles = [lse_quad_roles(rank) for rank in range(WORLD)]
    staged = [dict() for _ in range(WORLD)]

    def push(rank, pushes):
        for destination, head_owner, group in pushes:
            columns = slice(head_owner * heads, (head_owner + 1) * heads)
            staged[destination][group] = (partials[rank][:, columns].float(), lses[rank][:, columns])

    for rank, role in enumerate(roles):
        push(rank, role["first"])
    formed = []
    for rank, role in enumerate(roles[:8]):
        across = role["across"]
        columns = slice(across * heads, (across + 1) * heads)
        sources = [(partials[rank][:, columns].float(), lses[rank][:, columns])]
        sources += [staged[rank][group] for group in role["duty"]]
        partial, maximum, total = _combine([value for value, _ in sources], [lse for _, lse in sources])
        combined = torch.where(total > 0, maximum + torch.log(total), torch.full_like(total, -math.inf))
        formed.append((across, role["partial"][0], partial, combined))
    for rank, role in enumerate(roles):
        push(rank, role["second"])
    for across, group, partial, combined in formed:
        staged[across][group] = (partial, combined)
    outputs = []
    for rank, role in enumerate(roles):
        own = slice(rank * heads, (rank + 1) * heads)
        sources = [(partials[rank][:, own].float(), lses[rank][:, own])]
        sources += [staged[rank][term[1]] for term in role["terms"][1:]]
        result, _, _ = _combine([value for value, _ in sources], [lse for _, lse in sources])
        outputs.append(result.to(torch.bfloat16))
    return outputs


def _exact(partials, lses, heads):
    outputs = []
    for rank in range(WORLD):
        own = slice(rank * heads, (rank + 1) * heads)
        stacked = torch.stack([lses[source][:, own].double() for source in range(WORLD)])
        stacked = torch.where(torch.isfinite(stacked), stacked, torch.full_like(stacked, -math.inf))
        weights = torch.softmax(stacked, 0)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        values = torch.stack([partials[source][:, own].double() for source in range(WORLD)])
        values = torch.where((weights == 0).unsqueeze(-1), torch.zeros_like(values), values)
        outputs.append((weights.unsqueeze(-1) * values).sum(0))
    return outputs


@pytest.mark.parametrize("spread", [3.0, 30.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quad_combination_is_the_weighted_mean_of_all_nine_partials(spread, seed):
    rows, heads, width = 6, 3, 64
    generator = torch.Generator().manual_seed(100 * seed + int(spread))
    partials = [torch.randn((rows, WORLD * heads, width), generator=generator).to(torch.bfloat16) for _ in range(WORLD)]
    lses = [torch.randn((rows, WORLD * heads), generator=generator) * spread for _ in range(WORLD)]
    for rank in range(WORLD):
        # A shard without visible context reports -inf and may hold NaN partials.
        lses[rank][0, (7 * rank) % (WORLD * heads)] = -math.inf
        partials[rank][0, (7 * rank) % (WORLD * heads)] = math.nan
        lses[rank][1, 5] = -math.inf                       # no rank sees context for head 5 of row 1
        if rank < 4:
            lses[rank][2, 4 * heads:8 * heads] = -math.inf   # a whole quad is empty for the other quad's heads
            partials[rank][2, 4 * heads:8 * heads] = math.nan
        if rank != 6:
            lses[rank][3, :] = -math.inf                     # one rank holds everything
        if rank == 2:
            lses[rank][4, :] = math.nan                      # a NaN LSE weighs nothing
    pushed, quad, exact = (form(partials, lses, heads) for form in (_pushed, _quad, _exact))
    differing = elements = 0
    for rank in range(WORLD):
        assert not torch.isnan(quad[rank].float()).any() and not torch.isnan(pushed[rank].float()).any()
        reference = exact[rank]
        ulp = torch.pow(2.0, torch.floor(torch.log2(reference.abs().clamp(min=1e-30))) - 7)
        for form in (pushed[rank], quad[rank]):
            assert ((form.double() - reference).abs() <= ulp).all()
        differing += int((quad[rank].view(torch.int16) != pushed[rank].view(torch.int16)).sum())
        elements += quad[rank].numel()
    # A head nobody weighs is zero in both forms.
    owner, local = divmod(5, heads)
    assert not quad[owner][1, local].any() and not pushed[owner][1, local].any()
    # FP32 products and sums group differently: a small share of results rounds to the other neighbour.
    assert differing < elements // 50


def test_quad_switch_selects_the_launcher_and_its_compile_identity(monkeypatch):
    from b12x._lib.compile_plan import ProgramKey
    from b12x.comm.pcie import _dcp_a2a_cute as kernels

    monkeypatch.delenv("B12X_PCIE_DCP_LSE_PUSH_TP9", raising=False)
    assert not kernels.lse_quad_tp9_enabled()
    monkeypatch.setenv("B12X_PCIE_DCP_LSE_PUSH_TP9", "1")
    assert kernels.lse_push_tp9_enabled() and not kernels.lse_quad_tp9_enabled()
    monkeypatch.setenv("B12X_PCIE_DCP_LSE_PUSH_TP9", "2")
    assert kernels.lse_push_tp9_enabled() and kernels.lse_quad_tp9_enabled()
    compiled = []

    def fake_compile(launch, *args, compile_spec, **kwargs):
        def raw(*a, **k):
            return None

        raw.__b12x_programs__ = (ProgramKey("cute", "test", "dcp-lse-quad-tp9"),)
        compiled.append((launch, compile_spec))
        return raw

    monkeypatch.setattr(kernels, "b12x_compile", fake_compile)
    monkeypatch.setattr(kernels, "raise_if_kernel_resolution_frozen", lambda *args, **kwargs: None)
    for name in ("_u32_ptr", "_f32_ptr", "_u8_ptr"):
        monkeypatch.setattr(kernels, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernels, "current_cuda_stream", lambda: 0)
    kernels._get_compiled_lse_reduce_scatter.__wrapped__(9, 3, "bf16", 512, True)
    kernels._get_compiled_lse_reduce_scatter.__wrapped__(9, 3, "bf16", 512, True, True)
    (pushed, pushed_spec), (quad, quad_spec) = compiled
    # The transport is a plan declaration: the switch alone leaves the launcher on the push.
    assert type(pushed) is kernels._LseReduceScatterLaunch and pushed._push
    assert type(quad) is kernels._LseQuadReduceScatterLaunch and quad._roles == lse_quad_roles(3)
    assert '"push"' in pushed_spec.json_key and '"quad"' in quad_spec.json_key
    assert pushed_spec.json_key != quad_spec.json_key
    with pytest.raises(ValueError, match="nine ranks"):
        kernels._get_compiled_lse_reduce_scatter.__wrapped__(8, 3, "bf16", 512, True, True)
