"""Logical PCIe operations with native transport and launch candidates."""

from dataclasses import asdict, dataclass, replace

from b12x.preparation import (
    AutotuneEffort,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    Plan,
    TuningContract,
)
from b12x.preparation._efficiency import capture_autotune_effort

from ._tuning import PcieQuery, TUNING as NATIVE
from ._joint_profile import JOINT_FAMILIES, profile_records


@dataclass(frozen=True, kw_only=True)
class CollectiveConfig:
    implementation: str
    threads: int
    blocks: int
    submode: str = ""
    active_blocks: int = 0


@dataclass(frozen=True, kw_only=True)
class CollectiveQuery:
    operation: str
    world_size: int
    rank: int
    alternatives: FrozenMapping
    default: CollectiveConfig
    inplace: bool = False


def _configs(query, device=None):
    for name, payload in query.alternatives.items():
        call = payload["call"]
        blocks = int(call.get("blocks", call.get("block_limit", 16)))
        threads = int(call["threads"])
        if name in ("dcp_phased", "dcp_ring", "dcp_recursive", "dcp_recursive_ready"):
            # Automatic admission covers the supported SM120 geometry. Phased
            # kernels also have to pass the native cluster occupancy check.
            if (
                device is None
                or device.vendor != "nvidia"
                or device.compute_capability != (12, 0)
                or device.sm_count < 64
            ):
                continue
            counts = (
                (threads,)
                if payload["setup"].get("threads_override", 0)
                else (
                    (128,)
                    if name in ("dcp_ring", "dcp_recursive", "dcp_recursive_ready")
                    else (128, 256, 512)
                )
            )
            for count in counts:
                yield CollectiveConfig(
                    implementation=name,
                    threads=count,
                    blocks=8,
                    submode="barrier",
                    active_blocks=8,
                )
            continue
        if name in JOINT_FAMILIES:
            cluster_size = call.get("joint_cluster_size", 1)
            if cluster_size != 1:
                from ._joint_profile import JOINT_MODES, validate_cluster_geometry

                validate_cluster_geometry(blocks, threads, cluster_size)
                modes = {value: mode for mode, value in JOINT_MODES.items()}
                for mode, active in profile_records(
                    name, blocks, cluster_size=cluster_size
                ):
                    yield CollectiveConfig(
                        implementation=name,
                        threads=threads,
                        blocks=blocks,
                        submode=modes[mode],
                        active_blocks=active,
                    )
                continue
            for count_threads in (32, 64, 128, 256, 512):
                for mode in JOINT_FAMILIES[name]:
                    units = int(call["size_packs"])
                    if mode in ("bulk_push", "bulk_pull"):
                        units = -(-units // query.world_size)
                    covering = max(1, -(-units // count_threads))
                    for active in dict.fromkeys(
                        min(count, blocks)
                        for count in (1, 2, 4, 8, 16, 32, 36, blocks, covering)
                    ):
                        yield CollectiveConfig(
                            implementation=name,
                            threads=count_threads,
                            blocks=blocks,
                            submode=mode,
                            active_blocks=active,
                        )
            continue
        yield CollectiveConfig(implementation=name, threads=threads, blocks=blocks)
        if (
            query.world_size in (2, 3, 4, 6, 8, 9, 10)
            and query.operation == "all_reduce"
            and call.get("stage_input")
            and call.get("dtype") in ("float16", "bfloat16")
            and name
            in (
                "pull",
                "tp2_remote_push",
                "tp2_remote_push_simple",
                "remote_push_ll",
                "remote_push_simple",
                "remote_push_full",
                "remote_push_rsag",
            )
        ):
            from .pcie_oneshot import _MAX_BLOCKS

            units = int(call["size_packs"]) * (2 if name == "tp2_remote_push" else 1)
            for count_threads in (32, 64, 128, 256, 512):
                grids = (
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    _MAX_BLOCKS,
                    min(_MAX_BLOCKS, max(1, -(-units // count_threads))),
                )
                for count_blocks in dict.fromkeys(grids):
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count_blocks
                    )
        elif name == "dcp_word":
            from math import prod

            element_bytes = 1 if call["dtype"] == "torch.float8_e4m3fn" else 2
            chunks = (prod(call["shape"]) * element_bytes // 4 + 63) // 64
            setup = payload["setup"]
            thread_counts = (
                (threads,) if setup.get("threads_override", 0) else (32, 64, 128, 256)
            )
            for count_threads in thread_counts:
                warps = count_threads // 32
                if warps > 1 << (chunks - 1).bit_length():
                    continue
                grids = (
                    (blocks,)
                    if setup.get("blocks_override", 0)
                    else sorted({1, (chunks + warps - 1) // warps})
                )
                for count_blocks in grids:
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count_blocks
                    )
        elif name == "twoshot":
            from math import prod

            packs = prod(call["shape"]) // 8
            shard = -(-packs // query.world_size)
            for count_threads in (32, 64, 128, 256, 512):
                for count in dict.fromkeys(
                    (
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        min(64, max(1, -(-shard // count_threads))),
                    )
                ):
                    yield CollectiveConfig(
                        implementation=name, threads=count_threads, blocks=count
                    )
        elif name == "dcp":
            setup = payload["setup"]
            thread_values = (
                (threads,)
                if setup["threads_override"]
                else (
                    (128, 256, 512, 1024)
                    if query.operation == "all_gather_heads" and query.world_size == 2
                    else (128, 256, 512)
                )
            )
            block_values = (blocks,) if setup["blocks_override"] else (8, 16, 32, 64)
            for threads in thread_values:
                for count in block_values:
                    yield CollectiveConfig(
                        implementation=name, threads=threads, blocks=count
                    )


def _efficient_plain_grid(query, value):
    name = value["implementation"]
    if query.operation != "all_reduce" or name not in (
        "pull",
        "tp2_remote_push",
        "tp2_remote_push_simple",
        "remote_push_ll",
        "remote_push_simple",
        "remote_push_full",
        "remote_push_rsag",
    ):
        return True
    call = query.alternatives[name]["call"]
    if not call.get("stage_input") or call.get("dtype") not in ("float16", "bfloat16"):
        return True
    from .pcie_oneshot import _MAX_BLOCKS

    # Retain maximum grids at every thread count even for arbitrarily large payloads.
    if value["blocks"] == _MAX_BLOCKS:
        return True
    units = int(call["size_packs"]) * (2 if name == "tp2_remote_push" else 1)
    # Only exclude extreme serialization; idle blocks can affect synchronization latency.
    return units <= 1024 * value["threads"] * value["blocks"]


def _parameters(query, device):
    configs = tuple(dict.fromkeys(_configs(query, device)))
    effort = capture_autotune_effort()
    if effort is AutotuneEffort.OFF:
        eligible = {config.implementation for config in configs}
        configs = tuple(
            dict.fromkeys(
                CollectiveConfig(
                    implementation=name,
                    threads=int(payload["call"]["threads"]),
                    submode=payload["call"].get("submode", ""),
                    active_blocks=int(payload["call"].get("active_blocks", 0)),
                    blocks=int(
                        payload["call"].get(
                            "blocks", payload["call"].get("block_limit", 16)
                        )
                    ),
                )
                for name, payload in query.alternatives.items()
                if name in eligible
            )
        )
    legal_configs = frozenset(configs)
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "implementation": tuple(dict.fromkeys(c.implementation for c in configs)),
            "threads": tuple(dict.fromkeys(c.threads for c in configs)),
            "blocks": tuple(dict.fromkeys(c.blocks for c in configs)),
            "submode": tuple(dict.fromkeys(c.submode for c in configs)),
            "active_blocks": tuple(dict.fromkeys(c.active_blocks for c in configs)),
        },
        predicates=(lambda value: CollectiveConfig(**value) in legal_configs,),
        # Plain reductions measure pull alongside the payload-ready transports.
        efficiency_predicates=(
            lambda value: (
                value["implementation"] != "pull"
                or query.default.implementation == "pull"
                or query.operation == "all_reduce"
            ),
            lambda value: (
                effort is AutotuneEffort.OFF or _efficient_plain_grid(query, value)
            ),
        ),
        autotune_effort=effort,
    )


def _validate_query(query, device):
    if not isinstance(query, CollectiveQuery) or not query.alternatives:
        raise ValueError("collective query requires native alternatives")
    if (
        query.world_size not in (2, 3, 4, 6, 8, 9, 10, 12, 16)
        or not 0 <= query.rank < query.world_size
    ):
        raise ValueError("collective tuning requires a supported world size and rank")
    for name, payload in query.alternatives.items():
        native = PcieQuery(**payload)
        NATIVE.validate_query(native, device)
        if (native.world_size, native.rank) != (query.world_size, query.rank):
            raise ValueError("native alternatives must use the same rank group")
        if name in (
            "dcp_phased",
            "dcp_ring",
            "dcp_recursive",
            "dcp_recursive_ready",
        ) and (
            query.operation != "all_gather_heads"
            or query.world_size not in ((16,) if name == "dcp_phased" else (8, 16))
            or not native.surface.endswith(".all_gather_heads")
            or native.call.get("gather_transport")
            != {
                "dcp_ring": "selected_ring",
                "dcp_recursive": "selected_recursive",
                "dcp_recursive_ready": "selected_recursive_ready",
                "dcp_phased": "selected_phased_barrier",
            }[name]
        ):
            raise ValueError("selected gather tuning requires its supported barrier profile")


def _validate_config(query, config, device):
    if not isinstance(config, CollectiveConfig) or config not in tuple(
        _configs(query, device)
    ):
        raise ValueError(
            "configuration is outside the native collective parameter space"
        )


def _equivalence_key(query, device, config):
    if config.implementation == "twoshot":
        from math import prod

        packs = prod(query.alternatives["twoshot"]["call"]["shape"]) // 8
        shard = -(-packs // query.world_size)
        return (
            config.implementation,
            config.threads,
            min(config.blocks, max(1, -(-shard // config.threads))),
        )
    if config.implementation != "dcp":
        return asdict(config)
    native = query.alternatives["dcp"]
    heads = native["setup"]["total_heads"]
    if query.operation == "lse_reduce_scatter":
        heads //= query.world_size
    rows = native["call"]["shape"][0] * heads
    warps = config.threads // 32
    return (
        config.implementation,
        config.threads,
        min(config.blocks, (rows + warps - 1) // warps),
    )


def _collective_signature(query):
    alternatives = {}
    for name, payload in query.alternatives.items():
        native = payload.to_dict()
        del native["rank"]
        for section in ("call", "setup"):
            native[section].pop("device_index", None)
        alternatives[name] = native
    return dict(
        operation=query.operation,
        world_size=query.world_size,
        alternatives=alternatives,
        inplace=query.inplace,
    )


def _family_bucket(query):
    if query.operation != "all_reduce":
        return None
    from math import prod

    alternatives = {}
    sizes = set()
    for name, payload in query.alternatives.items():
        call, setup = payload["call"], payload["setup"]
        if call["dtype"] not in ("float16", "bfloat16"):
            return None
        sizes.add(prod(call["shape"] if name == "twoshot" else setup["shape"]) * 2)
        alternatives[name] = dict(
            surface=payload["surface"],
            topology=payload["topology"],
            dtype=call["dtype"],
            stage_input=call.get("stage_input"),
            remote_push_region_packs=call.get("remote_push_region_packs"),
            payload_ready_region_packs=call.get("payload_ready_region_packs"),
            eager_buffer_bytes=setup.get("eager_buffer_bytes"),
            eager_storage_shards=setup.get("eager_storage_shards"),
            registered=setup.get("registered"),
            shard_storage={
                field: setup[field]
                for field in ("max_rows", "pack_stride", "slot_bytes", "slots")
                if field in setup
            },
        )
        if "joint_cluster_size" in call:
            alternatives[name]["joint_cluster_size"] = call["joint_cluster_size"]
    if len(sizes) != 1:
        raise ValueError("collective alternatives disagree on payload size")
    size = sizes.pop()
    units = -(-size // 8192)
    ladder = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
    upper = next((value for value in ladder if value >= units), None)
    if upper is None:
        upper = 1 << (units - 1).bit_length()
    return (
        "implementation",
        dict(
            version=1,
            operation=query.operation,
            world_size=query.world_size,
            inplace=query.inplace,
            upper_bytes=upper * 8192,
            alternatives=alternatives,
        ),
        size,
    )


TUNING = TuningContract(
    component_id="comm.pcie",
    query_schema_version=7,
    candidate_contract_version=5,
    config_schema_version=2,
    query_fields=frozenset(
        ("operation", "world_size", "rank", "alternatives", "default", "inplace")
    ),
    config_fields=frozenset(
        ("implementation", "threads", "blocks", "submode", "active_blocks")
    ),
    encode_query=lambda q: dict(
        operation=q.operation,
        world_size=q.world_size,
        rank=q.rank,
        alternatives=q.alternatives,
        default=asdict(q.default),
        inplace=q.inplace,
    ),
    encode_config=asdict,
    decode_config=lambda value: CollectiveConfig(**value),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda q, device: q.default,
    knobs=(
        Knob(name="implementation", values=None),
        Knob(name="threads", values=None),
        Knob(name="blocks", values=None, binding=ParameterBinding.RUNTIME),
        Knob(name="submode", values=None, binding=ParameterBinding.RUNTIME),
        Knob(name="active_blocks", values=None, binding=ParameterBinding.RUNTIME),
    ),
    parameters=_parameters,
    collective_signature=_collective_signature,
    collective_family_bucket=_family_bucket,
    equivalence_key=_equivalence_key,
)


def plan(
    alternatives, *, default, inplace=False, invocation=FrozenMapping(), override=None
):
    from ._preparation import _provider
    from ._benchmark import benchmark_call

    first = next(iter(alternatives.values()))[0]
    default_call = alternatives[default][0].call
    query = CollectiveQuery(
        operation=first.surface.split(".", 1)[1],
        world_size=first.world_size,
        rank=first.rank,
        alternatives=FrozenMapping(
            {
                name: NATIVE.encode_query(q)
                for name, (q, runtime) in alternatives.items()
            }
        ),
        default=CollectiveConfig(
            implementation=default,
            submode=default_call.get("submode", ""),
            active_blocks=int(default_call.get("active_blocks", 0)),
            threads=int(default_call["threads"]),
            blocks=int(default_call.get("blocks", default_call.get("block_limit", 16))),
        ),
        inplace=inplace,
    )
    TUNING.validate_query(query, None)
    native_plans = {}

    def native(config):
        if config not in native_plans:
            q, runtime = alternatives[config.implementation]
            call = dict(q.call)
            call["threads"] = config.threads
            call["blocks" if "blocks" in call else "block_limit"] = config.blocks
            if config.implementation in JOINT_FAMILIES or config.implementation in (
                "dcp_phased",
                "dcp_ring",
                "dcp_recursive",
                "dcp_recursive_ready",
            ):
                call.update(submode=config.submode, active_blocks=config.active_blocks)
            q = replace(q, call=FrozenMapping(call))
            native_plans[config] = _provider(q.surface).plan(q, runtime=runtime)
        return native_plans[config]

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, device: native(config)._compile_jobs(
            config, device
        ),
        _memory_requirements=lambda config, device: native(config)._memory_requirements(
            config, device
        ),
        _materialize=lambda selection, device: native(selection.config)._materialize(
            selection, device
        ),
        _benchmark_call=lambda state: benchmark_call(state, inplace=inplace),
        _device=next(iter(alternatives.values()))[1].device,
    )


def oneshot_alternatives(query, runtime):
    alternatives = {}
    field = "transport" if query.surface.endswith(".all_reduce") else "mode"
    selected = query.call[field]
    alternatives[selected] = (query, runtime)
    if not query.call["stage_input"]:
        return alternatives
    modes = ["pull"] if field == "transport" else ["stage_pull"]
    if field == "transport":
        capacity = int(query.setup.get("eager_buffer_bytes") or 0) // 16
        packs = int(query.call["size_packs"])
        graph_slots = all(
            variant in tuple(tuple(v) for v in query.setup["variants"])
            for variant in ((True, 0), (True, 1))
        )
        valid_graph = (
            query.world_size in (2, 3, 4, 6, 8, 9, 10)
            and query.call["dtype"] in ("float16", "bfloat16")
            and graph_slots
            and 0 < packs <= min(capacity, 0x7FFFFFFF)
        )
        if (
            valid_graph
            and query.call["remote_push_region_packs"] > 0
            and query.call["remote_push_region_packs"]
            + 2 * (query.world_size - 1) * capacity
            <= int(query.setup["eager_storage_shards"]) * capacity
        ):
            from .pcie_oneshot import _MAX_BLOCKS

            modes.append("remote_push_ll")
            if query.world_size == 2:
                modes.append("tp2_remote_push")
            if (
                capacity >= _MAX_BLOCKS
                and packs * 16 <= capacity * 32 - _MAX_BLOCKS * 16
            ):
                modes.append("remote_push_simple")
                if query.world_size == 2:
                    modes.append("tp2_remote_push_simple")
        if (
            valid_graph
            and query.call["dtype"] == "bfloat16"
            and query.call.get("payload_ready_region_packs", 0) > 0
            and query.call["payload_ready_region_packs"] + query.world_size * capacity
            <= int(query.setup["eager_storage_shards"]) * capacity
        ):
            modes += ["remote_push_full", "remote_push_rsag"]
            if query.setup.get("joint_max_blocks", 0) > 0:
                modes += list(JOINT_FAMILIES)
    elif query.world_size == 4:
        modes += ["stage_remote_push"]
        # The payload-ready fused specialization uses an isolated incoming region.
        if selected == "stage_bf16_sentinel":
            modes += ["stage_bf16_sentinel"]
    for mode in modes:
        call = {**query.call, field: mode}
        if field == "transport" and mode == "pull":
            call.update(threads=call["eager_threads"], blocks=call["eager_blocks"])
        if field == "transport" and mode in ("tp2_remote_push", "remote_push_ll"):
            call.update(threads=128, blocks=36)
        if field == "transport" and mode in (
            "tp2_remote_push_simple",
            "remote_push_simple",
        ):
            call.update(threads=128, blocks=min(36, -(-call["size_packs"] // 128)))
        if field == "transport" and mode in ("remote_push_full", "remote_push_rsag"):
            call.update(threads=256, blocks=16)
        if mode in JOINT_FAMILIES:
            call.update(
                threads=256,
                blocks=query.setup["joint_max_blocks"],
                submode=JOINT_FAMILIES[mode][0],
                active_blocks=min(16, query.setup["joint_max_blocks"]),
            )
        if mode not in alternatives:
            alternatives[mode] = (replace(query, call=FrozenMapping(call)), runtime)
    return alternatives
