"""2-island hierarchical BF16 two-shot all-reduce (transports island / island_overlap / island_eighth).

Unit tests run anywhere. The GPU test needs eight CUDA devices ordered as two
PCIe islands (ranks 0-3, 4-7; partner = rank ^ 4) and is opt-in:

    B12X_RUN_PCIE_TWOSHOT_ISLAND_TEST=1 pytest tests/experimental/b12x/comm/test_pcie_twoshot_island.py
"""

from __future__ import annotations

import os
import socket

import pytest
import torch

from b12x.comm.pcie import _twoshot_preparation as tp
from b12x.comm.pcie._twoshot_bf16_cute import (
    _ISLAND_OVERLAP_STAGES,
    get_twoshot_bf16_island_launcher,
    island_overlap_fits,
)

WORLD_SIZE = 8
ROW_ELEMS = 7168
MAX_ROWS = 3072
ROWS = (1, 7, 300, 3072)
GPU_TRANSPORTS = ("island", "island_overlap", "island_eighth")


@pytest.mark.parametrize("transport", tp.ISLAND_TRANSPORTS)
def test_island_transports_are_bf16_all_reduce_only(transport):
    assert tp._transport("PCIeTwoShotBF16.all_reduce", transport) == transport
    with pytest.raises(ValueError, match="pull and island"):
        tp._transport("PCIeTwoShotBF16.reduce_scatter", transport)


def test_island_eligibility_needs_world_eight_and_whole_quarters():
    assert tp.island_eligible(8, 896)
    assert not tp.island_eligible(4, 896)
    assert not tp.island_eligible(8, 898)
    assert not tp.island_eligible(8, 0)


@pytest.mark.parametrize(
    "env, expected",
    (("0", "pull"), ("", "pull"), ("1", "island"), ("fp32", "island"),
     ("bf16", "island_bf16"), ("overlap", "island_overlap"), ("eighth", "island_eighth")),
)
def test_env_remaps_large_world_eight_reductions(monkeypatch, env, expected):
    monkeypatch.setenv("B12X_PCIE_TWOSHOT_ISLAND", env)
    monkeypatch.delenv("B12X_PCIE_TWOSHOT_ISLAND_MIN_BYTES", raising=False)
    big = (1 << 20) // 16
    surface = "PCIeTwoShotBF16.all_reduce"
    assert tp._island_transport(surface, "pull", 8, big) == expected
    assert tp._island_transport(surface, "push", 8, big) == (
        "push" if expected == "pull" else expected
    )
    # Below the size gate, at other world sizes and on partial quarters: unchanged.
    assert tp._island_transport(surface, "pull", 8, big - 4) == "pull"
    assert tp._island_transport(surface, "pull", 4, big) == "pull"
    assert tp._island_transport(surface, "pull", 8, big + 2) == "pull"
    assert tp._island_transport("PCIeTwoShotBF16.all_gather", "push", 8, big) == "push"


def test_env_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("B12X_PCIE_TWOSHOT_ISLAND", "fast")
    with pytest.raises(ValueError, match="must be 0, 1"):
        tp._island_transport("PCIeTwoShotBF16.all_reduce", "pull", 8, 1 << 20)


def test_min_bytes_gate_is_configurable(monkeypatch):
    monkeypatch.setenv("B12X_PCIE_TWOSHOT_ISLAND", "1")
    monkeypatch.setenv("B12X_PCIE_TWOSHOT_ISLAND_MIN_BYTES", "0")
    assert tp._island_transport("PCIeTwoShotBF16.all_reduce", "pull", 8, 4) == "island"


def test_overlap_staging_fits_push_slot():
    # Quarter of the largest payload against the push slot (8 * pack_stride packs).
    total = MAX_ROWS * ROW_ELEMS // 8
    pack_stride = -(-(-(-total // WORLD_SIZE)) // 16) * 16
    assert island_overlap_fits(total // 4, pack_stride)
    assert island_overlap_fits(ROW_ELEMS // 8 // 4, pack_stride)
    assert not island_overlap_fits(64 * _ISLAND_OVERLAP_STAGES, 16)


@pytest.mark.parametrize("world", (2, 4, 6, 10))
def test_island_launcher_requires_world_eight(world):
    with pytest.raises(ValueError, match="world size 8"):
        get_twoshot_bf16_island_launcher(world, 0, False, 0, 512, 8, 0)


def test_overlap_launcher_rejects_bf16_partials():
    with pytest.raises(ValueError, match="FP32 partials only"):
        get_twoshot_bf16_island_launcher(8, 0, False, 0, 512, 8, 0, True, True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, port: int) -> None:
    import hashlib

    import torch.distributed as dist
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD_SIZE
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    runtime = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=dist.group.WORLD, device=device, max_rows=MAX_ROWS, row_elems=ROW_ELEMS
    )

    def prepare(transport, inp, out, name):
        query = tp.query_from_runtime(
            runtime, surface="PCIeTwoShotBF16.all_reduce",
            call=dict(inp=inp, out=out, transport=transport),
        )
        assert query.call["transport"] == transport
        plan = tp.plan(query, runtime=runtime)
        session.prepare(
            (plan.request(name=name, collective=CollectiveRequirement(key=name, ranks=tuple(range(WORLD_SIZE))),
                          prepare_call=lambda state: tp.prepared_call(state, payload=inp, out=out)),),
            coordinator=lambda progress: progress.ready_collectives[0].key if progress.ready_collectives else None,
        )
        call = plan.prepared.state.query.call
        return lambda: runtime.all_reduce(
            inp, plan=plan, out=out, threads=call["threads"], block_limit=call["block_limit"]
        )

    def digest(tensor):
        return hashlib.sha256(tensor.view(torch.int16).cpu().numpy().tobytes()).hexdigest()

    try:
        for rows in ROWS:
            generator = torch.Generator().manual_seed(rows)
            exact = torch.randint(-64, 64, (WORLD_SIZE, rows, ROW_ELEMS), generator=generator).to(torch.bfloat16)
            noisy = torch.randn((WORLD_SIZE, rows, ROW_ELEMS), generator=generator).to(torch.bfloat16)
            results = {}
            for kind, operands in (("exact", exact), ("noisy", noisy)):
                reference = operands.double().sum(0)
                for transport in GPU_TRANSPORTS:
                    for alias in (False, True):
                        inp = operands[rank].to(device)
                        out = inp if alias else torch.full_like(inp, float("nan"))
                        run = prepare(transport, inp, out, f"{transport}-{kind}-{rows}-{alias}")
                        # Preparation may execute the call once; restore the in-place input.
                        inp.copy_(operands[rank].to(device))
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        run()
                        torch.cuda.synchronize(device)
                        result = out.cpu()
                        case = (rank, transport, kind, rows, alias)
                        if kind == "exact":  # integer sums are exact in FP32; one BF16 rounding
                            expected = reference.float().to(torch.bfloat16)
                            bad = (result.view(torch.int16) != expected.view(torch.int16)).nonzero()
                            assert bad.numel() == 0, (
                                case, bad.shape[0], bad[:3].tolist(),
                                result[tuple(bad[0])].item() if bad.numel() else None,
                                expected[tuple(bad[0])].item() if bad.numel() else None,
                            )
                        else:  # single rounding of an FP32 sum: within one BF16 ulp of the exact sum
                            error = (result.double() - reference).abs()
                            # one BF16 ulp of the sum, plus FP32 accumulation slack near zero
                            ulp = (reference.abs() * 2.0 ** -7).clamp_min(2.0 ** -20)
                            assert bool((error <= ulp).all()), (case, float(error.max()))
                        digests = [None] * WORLD_SIZE
                        dist.all_gather_object(digests, digest(result))
                        assert len(set(digests)) == 1, (transport, kind, rows, alias)
                        results.setdefault(kind, set()).add(digests[0])
                        if rows == ROWS[-1] and not alias and kind == "noisy":
                            graph = torch.cuda.CUDAGraph()
                            with runtime.capture(), torch.cuda.graph(graph):
                                run()
                            for _ in range(200):
                                graph.replay()
                            torch.cuda.synchronize(device)
                            assert torch.equal(out.cpu(), result)
                            graph.reset()
            # The pipelined variant reproduces the serial island result bit for bit.
            assert all(len(values) == 1 for values in results.values())
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        runtime.close()
        session.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TWOSHOT_ISLAND_TEST") != "1",
    reason="set B12X_RUN_PCIE_TWOSHOT_ISLAND_TEST=1 to run the 8-GPU island test",
)
def test_world_eight_island_all_reduce_matches_reference_on_all_ranks():
    import torch.multiprocessing as mp

    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        pytest.skip("needs exactly eight CUDA devices in island order")
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD_SIZE, join=True)


# ---- owner-eighth (island_eighth) protocol emulator: CPU-only check of the kernel's schedule ----


def _emulate_island_eighth(operands, *, immediate_remote: bool, footprint: dict, descending: bool = True):
    """NumPy model of `_TwoShotIslandEighthAllReduceLaunch` (same indices).

    operands: float32 array (8, packs, 8) holding BF16-exact values. Each
    iteration runs all ranks; remote stores are applied immediately (in rank
    order 7..0, the adversarial order for reads of rank 0..7) or deferred to
    the barrier. A missing barrier (read of a same-iteration remote write) or a
    premature overwrite (write-after-read across ranks) changes the result in
    one of the two modes. ``footprint`` collects the half index j touching each
    (rank, buffer, unit) to check the per-CTA contract.
    """
    import numpy as np
    from b12x.comm.pcie import _twoshot_bf16_cute as cute_mod

    def bf16(values):
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(torch.bfloat16).float().numpy()

    world, total, _ = operands.shape
    quarter = total // 4
    stages = cute_mod._ISLAND_OVERLAP_STAGES
    sub = cute_mod.island_eighth_sub_packs(quarter)
    half = sub // 2
    units = cute_mod._EIGHTH_AREA_UNITS * sub
    staging = np.full((world, units, 8), np.nan, dtype=np.float32)
    output = np.full((world, total, 8), np.nan, dtype=np.float32)
    deferred = []

    def touch(rank, buffer, unit, j):
        key = (rank, buffer, int(unit))
        assert footprint.setdefault(key, j) == j, ("per-CTA footprint violated", key, j, footprint[key])

    def store(writer, rank, unit, value, j):
        touch(rank, "staging", unit, j)
        if rank != writer and not immediate_remote:
            deferred.append((rank, unit, value.copy()))
        else:
            staging[rank, unit] = value

    def load(rank, unit, j):
        touch(rank, "staging", unit, j)
        value = staging[rank, unit].copy()
        assert not np.isnan(value).any(), ("read before write", rank, unit)
        return value

    def out_store(rank, pack, value, j):
        touch(rank, "output", pack, j)
        output[rank, pack] = value

    order = list(range(world - 1, -1, -1)) if descending else list(range(world))
    for step in range(stages + 4):
        for rank in order:
            base, local, partner, island_a = rank & 4, rank & 3, rank ^ 4, rank < 4
            mine, theirs = (0, half) if island_a else (half, 0)
            for j in range(half):
                if step < stages:  # A
                    first, parity = step * sub, step & 1
                    for hop in (1, 2, 3):
                        peer_local = (local + hop) & 3
                        for position in (j, j + half):
                            if first + position < quarter:
                                store(rank, base + peer_local,
                                      (cute_mod._EIGHTH_RS + parity * 4 + local) * sub + position,
                                      operands[rank, peer_local * quarter + first + position], j)
                if 1 <= step <= stages:  # B
                    chunk_id = step - 1
                    first, parity = chunk_id * sub, chunk_id & 1
                    for side, offset in ((0, mine), (1, theirs)):
                        position = j + offset
                        if first + position < quarter:
                            acc = None
                            for source_local in range(4):
                                if source_local == local:
                                    value = operands[rank, local * quarter + first + position]
                                else:
                                    value = load(rank, (cute_mod._EIGHTH_RS + parity * 4 + source_local) * sub + position, j)
                                acc = value.astype(np.float32) if acc is None else (acc + value).astype(np.float32)
                            target_rank = rank if side == 0 else partner
                            area = (cute_mod._EIGHTH_OWN if side == 0 else cute_mod._EIGHTH_PIN) + parity
                            store(rank, target_rank, area * sub + j, np.concatenate([acc[:4], [0.0] * 4]).astype(np.float32), j)
                            store(rank, target_rank, area * sub + half + j, np.concatenate([acc[4:], [0.0] * 4]).astype(np.float32), j)
                if 2 <= step <= stages + 1:  # C
                    chunk_id = step - 2
                    first, parity = chunk_id * sub, chunk_id & 1
                    position = j + mine
                    if first + position < quarter:
                        own = np.concatenate([load(rank, (cute_mod._EIGHTH_OWN + parity) * sub + j, j)[:4],
                                              load(rank, (cute_mod._EIGHTH_OWN + parity) * sub + half + j, j)[:4]])
                        peer = np.concatenate([load(rank, (cute_mod._EIGHTH_PIN + parity) * sub + j, j)[:4],
                                               load(rank, (cute_mod._EIGHTH_PIN + parity) * sub + half + j, j)[:4]])
                        final = bf16((own + peer).astype(np.float32))
                        out_store(rank, local * quarter + first + position, final, j)
                        store(rank, partner, cute_mod._EIGHTH_FIN * sub + parity * half + j, final, j)
                        for hop in (1, 2, 3):
                            peer_local = (local + hop) & 3
                            store(rank, base + peer_local,
                                  (cute_mod._EIGHTH_AG + (chunk_id % 3) * 4 + local) * sub + position, final, j)
                if 3 <= step <= stages + 2:  # D
                    chunk_id = step - 3
                    first, parity = chunk_id * sub, chunk_id & 1
                    position = j + theirs
                    if first + position < quarter:
                        final = load(rank, cute_mod._EIGHTH_FIN * sub + parity * half + j, j)
                        out_store(rank, local * quarter + first + position, final, j)
                        for hop in (1, 2, 3):
                            peer_local = (local + hop) & 3
                            store(rank, base + peer_local,
                                  (cute_mod._EIGHTH_AG + (chunk_id % 3) * 4 + local) * sub + position, final, j)
                if step >= 4:  # E
                    chunk_id = step - 4
                    first = chunk_id * sub
                    for hop in (1, 2, 3):
                        source_local = (local + hop) & 3
                        for position in (j, j + half):
                            if first + position < quarter:
                                out_store(rank, source_local * quarter + first + position,
                                          load(rank, (cute_mod._EIGHTH_AG + (chunk_id % 3) * 4 + source_local) * sub + position, j), j)
        for rank, unit, value in deferred:  # barrier
            staging[rank, unit] = value
        deferred.clear()
    return output


@pytest.mark.parametrize("packs", (896, 4 * 13 * 7, 4 * 896 * 3, 4 * 37))
def test_island_eighth_protocol_emulation_is_exact_and_race_free(packs):
    import numpy as np

    generator = torch.Generator().manual_seed(packs)
    operands = torch.randn((WORLD_SIZE, packs, 8), generator=generator).to(torch.bfloat16).float().numpy()
    quarter = packs // 4
    results = []
    for immediate, descending in ((False, True), (True, True), (True, False)):
        footprint = {}
        out = _emulate_island_eighth(
            operands, immediate_remote=immediate, footprint=footprint, descending=descending
        )
        assert not np.isnan(out).any()
        results.append(out)
    # No same-iteration cross-rank dependency (RAW or WAR) in either issue order.
    assert np.array_equal(results[0], results[1]) and np.array_equal(results[0], results[2])
    out = results[0]
    assert all(np.array_equal(out[0], out[rank]) for rank in range(WORLD_SIZE))  # identical ranks
    # One rounding of (island A partial in order 0..3) + (island B partial in order 0..3), FP32.
    a = operands[0].astype(np.float32)
    for r in (1, 2, 3):
        a = (a + operands[r]).astype(np.float32)
    b = operands[4].astype(np.float32)
    for r in (5, 6, 7):
        b = (b + operands[r]).astype(np.float32)
    expected = torch.from_numpy((a + b).astype(np.float32)).to(torch.bfloat16).float().numpy()
    assert np.array_equal(out[0], expected)
    assert quarter * 4 == packs


def test_island_eighth_staging_fits_push_slot_and_env():
    from b12x.comm.pcie._twoshot_bf16_cute import island_eighth_fits, island_eighth_sub_packs

    total = MAX_ROWS * ROW_ELEMS // 8
    pack_stride = -(-(-(-total // WORLD_SIZE)) // 16) * 16
    for rows in (1, 7, 300, 1536, MAX_ROWS):
        quarter = rows * ROW_ELEMS // 8 // 4
        assert island_eighth_sub_packs(quarter) % 2 == 0
        assert island_eighth_sub_packs(quarter) * _ISLAND_OVERLAP_STAGES >= quarter
        assert island_eighth_fits(quarter, pack_stride)
    assert not island_eighth_fits(4096, 16)


def test_island_eighth_launcher_rejects_conflicting_variants():
    with pytest.raises(ValueError, match="FP32 partials only"):
        get_twoshot_bf16_island_launcher(8, 0, False, 0, 512, 8, 0, True, False, True)
    with pytest.raises(ValueError, match="one island pipeline variant"):
        get_twoshot_bf16_island_launcher(8, 0, False, 0, 512, 8, 0, False, True, True)
