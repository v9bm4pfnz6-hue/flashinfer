"""Collective exchanges preserve identical candidate schedules across ranks."""

from dataclasses import replace

import pytest

from b12x.preparation import CollectiveTuningRequirement, FrozenMapping
from b12x.preparation._collective import exchange, _same
from b12x.preparation.types import CollectiveRequirement
from .test_session import request
from .test_race_elimination import _Timer, _race, host_timing
from b12x.preparation._measurement import measure_race_steps


def test_family_benchmark_is_used_by_default_and_caller_can_override():
    req = request(name="source")
    builtin, supplied = lambda state: None, lambda state: state
    plan = replace(req.plan, _benchmark_call=builtin)
    assert (
        plan.request(name="a", prepare_call=req.prepare_call).benchmark_call is builtin
    )
    assert (
        plan.request(
            name="b", prepare_call=req.prepare_call, benchmark_call=supplied
        ).benchmark_call
        is supplied
    )


def test_exchange_preserves_rank_order_and_rejects_different_catalogs():
    requirement = CollectiveRequirement("reduce", (2, 5))
    steps = exchange(requirement, "catalog", candidates=(1, 2))
    local = next(steps)
    assert local == CollectiveTuningRequirement(
        "reduce", (2, 5), "catalog", FrozenMapping({"candidates": (1, 2)})
    )
    with pytest.raises(StopIteration) as result:
        steps.send(
            (local, replace(local, payload=FrozenMapping({"candidates": (1, 3)})))
        )
    with pytest.raises(ValueError, match="candidates"):
        _same(result.value.value, "candidates")


def test_dcp_candidates_coalesce_effective_grids_and_preserve_overrides():
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        TUNING,
    )
    from b12x.preparation import autotune_effort

    native = dict(
        surface="DcpAllToAll.all_gather_heads",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=dict(threads=256, block_limit=16, shape=(1, 4, 512)),
        setup=dict(total_heads=16, threads_override=0, blocks_override=0),
    )
    query = CollectiveQuery(
        operation="all_gather_heads",
        world_size=4,
        rank=0,
        alternatives=FrozenMapping({"dcp": native}),
        default=CollectiveConfig(implementation="dcp", threads=256, blocks=16),
    )
    with autotune_effort("default"):
        values = list(TUNING.iterate(TUNING.configure(query, device=None)))
    assert len(values) == 3
    assert {config.threads for _, config in values} == {128, 256, 512}
    native["setup"].update(threads_override=256, blocks_override=16)
    query = replace(query, alternatives=FrozenMapping({"dcp": native}))
    with autotune_effort("default"):
        values = list(TUNING.iterate(TUNING.configure(query, device=None)))
    assert len(values) == 1
    assert values[0][1] == query.default


@pytest.mark.parametrize("shape", [(3, 1536), (513, 2560), (8,)])
@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 9, 10))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_plain_catalog_uses_storage_bounds_instead_of_shape_policy(shape, world, dtype):
    from math import prod
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        TUNING,
        oneshot_alternatives,
    )
    from b12x.comm.pcie._tuning import PcieQuery, TUNING as NATIVE
    from b12x.preparation import autotune_effort

    packs = prod(shape) // 8
    capacity = max(packs, 64)
    native = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=world,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                dtype=dtype,
                transport="pull",
                threads=256,
                blocks=8,
                eager_threads=256,
                eager_blocks=8,
                stage_input=True,
                size_packs=packs,
                remote_push_region_packs=capacity,
                payload_ready_region_packs=(1 + 2 * (world - 1)) * capacity,
            )
        ),
        setup=FrozenMapping(
            dict(
                shape=shape,
                variants=((False, 0), (True, 0), (True, 1)),
                eager_buffer_bytes=capacity * 16,
                eager_storage_shards=1 + 2 * (world - 1) + world,
            )
        ),
    )
    alternatives = oneshot_alternatives(native, object())
    expected = {"pull", "remote_push_ll", "remote_push_simple"}
    if world == 2:
        expected |= {"tp2_remote_push", "tp2_remote_push_simple"}
    if dtype == "bfloat16":
        expected |= {"remote_push_full", "remote_push_rsag"}
        if world == 9:
            # Nine ranks also race the reduce-scatter whose results cross between the sides once.
            expected |= {"remote_push_rsag_side"}
    assert set(alternatives) == expected
    query = CollectiveQuery(
        operation="all_reduce",
        world_size=world,
        rank=0,
        alternatives=FrozenMapping(
            {k: NATIVE.encode_query(v[0]) for k, v in alternatives.items()}
        ),
        default=CollectiveConfig(
            implementation="remote_push_ll", threads=128, blocks=36
        ),
    )
    with autotune_effort("default"):
        configs = [
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        ]
    assert {config.implementation for config in configs} == set(alternatives)
    assert all(1 <= config.blocks <= 36 for config in configs)
    assert (
        CollectiveConfig(implementation="remote_push_ll", threads=128, blocks=36)
        in configs
    )
    with pytest.raises(ValueError, match="parameter space"):
        TUNING.validate_config(
            query, CollectiveConfig(implementation="pull", threads=128, blocks=37), None
        )
    small = replace(
        native,
        call=FrozenMapping(
            {**native.call, "size_packs": 1, "remote_push_region_packs": 1}
        ),
        setup=FrozenMapping({**native.setup, "shape": (8,), "eager_buffer_bytes": 16}),
    )
    small_expected = {"pull", "remote_push_ll"}
    if world == 2:
        small_expected.add("tp2_remote_push")
    assert set(oneshot_alternatives(small, object())) == small_expected
    absent = replace(
        native, setup=FrozenMapping({**native.setup, "eager_storage_shards": 1})
    )
    assert set(oneshot_alternatives(absent, object())) == {"pull"}

    for default in {config.implementation: config for config in configs}.values():
        other = replace(query, default=default)
        with autotune_effort("default"):
            measured = {
                config
                for _, config in TUNING.iterate(TUNING.configure(other, device=None))
            }
        assert measured == set(configs)

    for world_size, rank in ((world, -1), (world, world), (5, 0)):
        with pytest.raises(ValueError, match="world size and rank"):
            TUNING.validate_query(
                replace(query, world_size=world_size, rank=rank), None
            )


def test_collective_signature_distinguishes_shapes_but_not_rank_ordinals():
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        _collective_signature,
    )

    def query(rank, rows):
        return CollectiveQuery(
            operation="all_reduce",
            world_size=4,
            rank=rank,
            default=CollectiveConfig(implementation="pull", threads=256, blocks=16),
            alternatives=FrozenMapping(
                {
                    "pull": dict(
                        rank=rank,
                        world_size=4,
                        call=dict(shape=(rows, 5120), dtype="bf16", device_index=rank),
                        setup=dict(device_index=rank),
                    )
                }
            ),
        )

    assert _collective_signature(query(0, 8)) == _collective_signature(query(3, 8))
    assert _collective_signature(query(0, 8)) != _collective_signature(query(0, 16))


def test_collective_samples_control_repetitions_and_elimination(
    host_timing, monkeypatch
):
    from b12x.preparation import _measurement

    monkeypatch.setattr(_measurement, "ROUND_BUDGET_US", 64)
    schedules, counts = [], []
    for rank in range(2):
        timers = (_Timer(1 if rank == 0 else 8), _Timer(4 if rank == 0 else 2))
        schedule = []

        def reduce(indices, turn, repetition, samples):
            schedule.append((indices, turn, repetition))
            yield "exchange"
            return tuple((8.0 if index == 0 else 4.0,) for index in indices)

        prepared = _race(timers)
        steps = measure_race_steps(
            prepared, device_ordinal=0, rounds=3, eliminate=True, sample_exchange=reduce
        )
        while True:
            try:
                next(steps)
            except StopIteration as result:
                assert result.value.latencies_us == (8.0, 4.0)
                break
        schedules.append(schedule)
        counts.append(tuple(timer.replays for timer in timers))
    assert schedules[0] == schedules[1]
    assert counts[0] == counts[1] == (8, 48)


def _adaptive_measure(timers, sample_exchange=None, *, rounds=15):
    def identity(indices, turn, repetition, samples):
        yield
        return samples

    prepared = _race(timers)
    steps = measure_race_steps(
        prepared,
        device_ordinal=0,
        rounds=rounds,
        eliminate=True,
        adaptive_collective_pair=True,
        adaptive_repeats=False,
        sample_exchange=sample_exchange or identity,
    )
    while True:
        try:
            next(steps)
        except StopIteration as result:
            return prepared, result.value


@pytest.mark.parametrize("difference", (-2.0, 2.0))
def test_adaptive_collective_clear_pair_stops_after_three_rounds(
    host_timing, difference
):
    timers = (_Timer(10.0), _Timer(10.0 + difference))
    prepared, result = _adaptive_measure(timers)
    assert prepared.completed_rounds == 3
    assert tuple(timer.replays for timer in timers) == (3, 3)
    assert result.latencies_us == (10.0, 10.0 + difference)


def test_adaptive_collective_noisy_pair_collects_more_rounds(host_timing):
    timers = (_Timer(10.0), _Timer(9.0, 11.0, 10.0, *([11.0] * 12)))
    prepared, result = _adaptive_measure(timers)
    assert 3 < prepared.completed_rounds <= 15
    assert timers[0].replays == timers[1].replays == prepared.completed_rounds
    assert result.latencies_us == (10.0, 11.0)


def test_adaptive_collective_exact_tie_reaches_bound(host_timing):
    timers = (_Timer(10.0), _Timer(10.0))
    prepared, result = _adaptive_measure(timers)
    assert prepared.completed_rounds == prepared.planned_rounds == 15
    assert tuple(timer.replays for timer in timers) == (15, 15)
    assert result.latencies_us == (10.0, 10.0)


def test_adaptive_collective_mean_cannot_override_opposite_median():
    from b12x.preparation._measurement import _collective_pair_separated

    first = [100.0 * (index + 1) for index in range(15)]
    second = [value + 10 for value in first]
    second[7] = first[7] - 20
    assert sum(second) > sum(first)
    assert sorted(second)[7] < sorted(first)[7]
    assert not _collective_pair_separated((first, second))


def test_adaptive_collective_rank_reduction_controls_shared_stopping(host_timing):
    schedules = []
    for rank in range(2):
        schedule = []

        def reduce(indices, turn, repetition, samples):
            schedule.append((indices, turn, repetition))
            yield
            return tuple(
                (10.0 if index == 0 else (9.0, 11.0, 10.0)[turn % 3],)
                for index in indices
            )

        prepared, _ = _adaptive_measure(
            (_Timer(1.0 + rank * 50), _Timer(20.0 - rank * 10)), reduce
        )
        assert prepared.completed_rounds == 15
        schedules.append(schedule)
    assert schedules[0] == schedules[1]


@pytest.mark.parametrize("failure", ("cancel", "close"))
def test_adaptive_collective_extended_race_closes_resources(
    host_timing, monkeypatch, failure
):
    from contextlib import nullcontext
    from threading import Event
    from types import SimpleNamespace
    from b12x.preparation import _measurement
    from b12x.preparation._collective import _Cancelled
    from b12x.preparation.session import PreparationJob

    closed = []
    timers = (_Timer(10.0), _Timer(10.0))
    for index, timer in enumerate(timers):
        timer.close = lambda index=index: closed.append(index)
    prepared = _race(timers)

    def prepare(*args, **kwargs):
        return prepared
        yield

    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    job = PreparationJob.__new__(PreparationJob)
    job.session = SimpleNamespace(
        device=SimpleNamespace(ordinal=0),
        samples=2,
        rounds=15,
        cache_policy="cold",
        _stop=Event(),
        _pool=None,
    )
    job._race_eviction = object()
    job._timing = SimpleNamespace(span=lambda name: nullcontext())

    def exchange(indices, turn, repetition, samples):
        if failure == "cancel" and turn == 3:
            raise _Cancelled
        yield
        return samples

    steps = job._measure(
        [SimpleNamespace(call=timer.call) for timer in timers],
        champion=True,
        adaptive_collective_pair=True,
        sample_exchange=exchange,
    )
    if failure == "cancel":
        with pytest.raises(_Cancelled):
            while True:
                next(steps)
    else:
        while prepared.completed_rounds < 3:
            next(steps)
        steps.close()
    assert prepared._closed
    assert sorted(closed) == [0, 1]


@pytest.mark.parametrize("rounds", (1, 2, 4, 7, 20))
def test_adaptive_collective_explicit_round_limit(host_timing, rounds):
    timers = (_Timer(10.0), _Timer(10.0))
    prepared, _ = _adaptive_measure(timers, rounds=rounds)
    assert prepared.completed_rounds == min(rounds, 15)
    assert tuple(timer.replays for timer in timers) == (min(rounds, 15),) * 2


@pytest.mark.parametrize("cached", [(), (3,), (3, 5, 7)])
def test_preparation_progress_counts_collective_and_local_measurements(
    tmp_path, monkeypatch, cached
):
    from b12x.preparation import PreparedCall
    from b12x.preparation import _collective
    from .test_defaults import Query, contract
    from .test_session import _deterministic_timer, session

    _deterministic_timer(monkeypatch)
    monkeypatch.setattr(
        _collective, "capture_benchmark_call", lambda call, **kwargs: call
    )

    def requests(rows):
        items = []
        for row in rows:
            tuning = contract(values=(1, 2, 4))
            collective = None
            if row != 7:
                tuning = replace(tuning, collective_signature=lambda query: query.rows)
                collective = CollectiveRequirement(f"group/{row}", (0, 1))
            item = request(
                name=f"rows-{row}",
                tuning=tuning,
                collective=collective,
                benchmark=lambda state: PreparedCall(
                    run=lambda: state.value, produce=lambda: None
                ),
            )
            items.append(replace(item, plan=replace(item.plan, query=Query(row))))
        return tuple(items)

    def coordinate(progress):
        if progress.ready_collective_tuning is not None:
            exchange = progress.ready_collective_tuning
            return (exchange, exchange)
        if progress.ready_collectives:
            return progress.ready_collectives[0].key
        return None

    if cached:
        with session(tmp_path) as engine:
            engine.prepare(requests(cached), coordinator=coordinate)
    snapshots = []
    with session(tmp_path) as engine:
        result = engine.prepare(
            requests((3, 5, 7)), coordinator=coordinate, progress=snapshots.append
        )
        expected = 3 * (3 - len(cached))
        assert result.benchmarked_candidates == expected
        assert result.cache_hits == len(cached)
        assert result.restored_candidates == 0
        for row in (3, 5, 7):
            assert result.selections[f"rows-{row}"].source == (
                "cached" if row in cached else "tuned"
            )
        final = snapshots[-1]
        assert final.completed_candidates == final.total_candidates == expected
        assert all(
            snapshot.completed_candidates <= snapshot.total_candidates
            for snapshot in snapshots
            if snapshot.total_candidates is not None
        )


def test_cancelled_collective_progress_retains_completed_measurements(
    tmp_path, monkeypatch
):
    from b12x.preparation import PreparedCall
    from b12x.preparation import _collective
    from .test_defaults import contract
    from .test_session import _deterministic_timer, session

    _deterministic_timer(monkeypatch)
    monkeypatch.setattr(
        _collective, "capture_benchmark_call", lambda call, **kwargs: call
    )
    item = request(
        name="partial",
        tuning=replace(contract(), collective_signature=lambda query: query.rows),
        collective=CollectiveRequirement("group/partial", (0, 1)),
        benchmark=lambda state: PreparedCall(
            run=lambda: state.value, produce=lambda: None
        ),
    )

    def coordinate(progress):
        exchange = progress.ready_collective_tuning
        if exchange is not None:
            if exchange.step == "bound/1":
                exchange = replace(
                    exchange,
                    payload=FrozenMapping({**exchange.payload, "cancel": True}),
                )
            return (exchange, exchange)
        if progress.ready_collectives:
            return progress.ready_collectives[0].key
        return None

    snapshots = []
    with session(tmp_path) as engine:
        result = engine.prepare(
            (item,), coordinator=coordinate, progress=snapshots.append
        )
        assert result.selections["partial"].source == "default"
        assert result.benchmarked_candidates == 1
        assert snapshots[-1].completed_candidates == snapshots[-1].total_candidates == 1


def test_collective_catalog_rejects_rank_divergent_finalist_policy():
    from threading import Event
    from types import SimpleNamespace
    from b12x.preparation import FinalistPolicy, PreparedCall
    from b12x.preparation._collective import select
    from .test_defaults import Config, contract

    tuning = replace(
        contract(), collective_signature=lambda query: {"rows": query.rows}
    )
    req = replace(
        request(
            name="collective",
            tuning=tuning,
            collective=CollectiveRequirement("reduce", (0, 1)),
            benchmark=lambda state: PreparedCall(run=lambda: None),
        ),
        finalist_policy=FinalistPolicy(family_fields=("width",)),
    )
    job = SimpleNamespace(
        _choice_key=lambda obligation, selections: "local-query",
        session=SimpleNamespace(
            device=SimpleNamespace(uuid="rank-device"),
            cache_only=False,
            rounds=7,
            samples=2,
            _stop=Event(),
        ),
    )
    obligation = SimpleNamespace(
        request=req, candidates=((FrozenMapping({"width": 1}), Config(1)),)
    )
    steps = select(job, obligation, {})
    local = next(steps)
    assert local.payload["finalist_policy"] == (("width",), 1, 16)
    remote = replace(
        local, payload=FrozenMapping({**local.payload, "finalist_policy": None})
    )
    with pytest.raises(ValueError, match="disagree on finalist_policy"):
        steps.send((local, remote))


def test_collective_finalist_rerace_preserves_ordinary_winner(tmp_path, monkeypatch):
    from b12x.preparation import FinalistPolicy, PreparedCall
    from b12x.preparation import _collective, _measurement
    from .test_defaults import contract
    from .test_session import _deterministic_timer, session

    batches = []
    closed = []
    _deterministic_timer(monkeypatch, batches=batches)
    monkeypatch.setattr(
        _collective, "capture_benchmark_call", lambda call, **kwargs: call
    )
    ordinary_measure = _measurement.measure_race_steps

    def measure(race, **kwargs):
        result = yield from ordinary_measure(race, **kwargs)
        if len(race.calls) == 3:
            return _measurement.RaceMeasurements((1.0, 5.0, 2.0), 0)
        return result

    monkeypatch.setattr(_measurement, "measure_race_steps", measure)

    def make(name):
        return replace(
            request(
                name=name,
                tuning=replace(
                    contract(), collective_signature=lambda query: query.rows
                ),
                collective=CollectiveRequirement("finalists", (0, 1)),
                benchmark=lambda state: PreparedCall(
                    run=lambda: state.value,
                    produce=lambda: None,
                    close=lambda: closed.append(state.value),
                ),
            ),
            finalist_policy=FinalistPolicy(family_fields=("width",)),
        )

    def coordinate(progress):
        exchange = progress.ready_collective_tuning
        if exchange is not None:
            return (exchange, exchange)
        if progress.ready_collectives:
            return progress.ready_collectives[0].key
        return None

    with session(tmp_path) as engine:
        result = engine.prepare((make("first"),), coordinator=coordinate)
        assert result.selections["first"].config.width == 2
        assert result.benchmarked_candidates == 3
        finalists = result.finalists["first"]
        assert {item.selection.config.width for item in finalists} == {1, 2, 4}
        assert (
            min(finalists, key=lambda item: item.latency_us).selection.config.width == 1
        )
    assert batches == [(3,), (3, 6), (6, 12), (3, 6, 12)]
    assert sorted(closed) == [3, 3, 6, 12, 12]
    with session(tmp_path) as engine:
        cached = engine.prepare((make("restored"),), coordinator=coordinate)
        assert cached.cache_hits == 1
        assert cached.benchmarked_candidates == 0
        assert cached.selections["restored"].config.width == 2
        assert cached.finalists["restored"] == tuple(
            replace(item, selection=replace(item.selection, source="cached"))
            for item in finalists
        )
