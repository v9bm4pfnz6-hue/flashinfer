"""Frozen input-transform geometry controls storage without changing arithmetic."""

from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe import _impl, _preparation
from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    TUNING,
    _validate_query,
)
from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig,
    write_exl3_checkpoint,
)
from b12x.preparation import FrozenMapping
from b12x.preparation.session import _declaration_key


def _helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/experimental/b12x/benchmark_trellis_shared_input.py"
    )
    spec = importlib.util.spec_from_file_location("trellis_scratch_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _caps(rows, experts=896, topk=16, frozen=None):
    plan = _impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="exl3",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=3584,
        intermediate_size=384,
        trellis_bits=2,
        trellis_codebook="lut_e4m3",
        trellis_tile_config=(128, 128, 128, 128),
        intermediate_hadamard=True,
        intermediate_hadamard_blocks=(512, 128),
    )
    return _impl.TPMoEScratchCaps(
        max_tokens=rows,
        num_topk=topk,
        device="cpu",
        weight_plan=plan,
        quant_mode="w4a16",
        trellis_input_scale_rows=frozen,
        decode_config=MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        ),
    )


def _phase_saving(plan):
    sizes = {s.name: math.prod(s.shape) * s.dtype.itemsize for s in plan.tensor_specs}
    before = sum(
        sizes[name]
        for name in (
            "fc1_c_tmp",
            "fc2_c_tmp",
            "rotation_a_gate",
            "full_rotation_output",
        )
    )
    return before - max(
        sizes["fc1_c_tmp"] + sizes["rotation_a_gate"],
        sizes["fc2_c_tmp"],
        sizes["full_rotation_output"],
    )


@pytest.mark.parametrize("rows", [1, 8, 128, 1024])
@pytest.mark.parametrize("experts,topk", [(1, 1), (1, 4), (17, 4), (896, 16)])
def test_compact_size_changes_only_input_transform_storage(
    monkeypatch, rows, experts, topk
):
    monkeypatch.setattr(_impl, "get_num_sm", lambda _: 188)
    plans = [
        _impl.plan_tp_moe_scratch(
            _caps(rows, experts, topk, frozen), prewarm_launches=False
        )
        for frozen in (None, 1, experts)
    ]
    generic, shared, per_expert = plans
    assert generic.layout.total_nbytes - shared.layout.total_nbytes == rows * (
        topk - 1
    ) * 3584 * 2 - _phase_saving(generic._core_workspace_plan) + _phase_saving(
        shared._core_workspace_plan
    )
    assert (
        per_expert.layout.total_nbytes
        == (shared if experts == 1 else generic).layout.total_nbytes
    )
    a, b = (
        dict((s.name, (s.shape, s.dtype)) for s in p._core_workspace_plan.tensor_specs)
        for p in (generic, shared)
    )
    assert a.pop("rotation_a_gate") == ((rows * topk, 3584), torch.float16)
    assert b.pop("rotation_a_gate") == ((rows, 3584), torch.float16)
    assert a == b


@pytest.mark.parametrize("value", [0, -1, 2, True, "1"])
def test_frozen_rows_reject_invalid_physical_geometry(value):
    with pytest.raises(ValueError, match="frozen input scale rows"):
        _caps(128, frozen=value)


def test_legacy_pool_rejects_compact_layout_before_allocation(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("legacy pool allocated before rejecting compact layout")

    monkeypatch.setattr(_impl, "_allocate_core_arena", unexpected)
    with pytest.raises(NotImplementedError, match="plan_tp_moe_scratch"):
        _impl.materialize_tp_moe_arena_workspaces(object(), caps=_caps(128, frozen=1))


def test_optional_row_geometry_roundtrip_and_identity():
    query = MoeDecodeQuery(
        quant_mode="w4a16",
        quant_modes=("w4a16",),
        source_format="b12x_trellis",
        activation="situ",
        io_dtype="bfloat16",
        num_experts=896,
        hidden_size=3584,
        intermediate_size=384,
        top_k=16,
        num_tokens=1024,
        routed_rows=16384,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        w13_layout="w31",
        weight_layouts=("trellis_native",),
        w4a16_weight_layout="trellis_t256",
        w4a16_scale_format="e4m3_k32",
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=None,
        controls=FrozenMapping(),
    )
    assert "trellis_input_scale_rows" not in query.to_dict()
    assert _preparation._decode_query(query.to_dict()) == query
    identities = []
    declarations = []
    compiled_queries = []
    for value in (None, 1, 896):
        item = replace(query, trellis_input_scale_rows=value)
        _validate_query(item, None)
        assert _preparation._decode_query(item.to_dict()) == item
        identities.append(FrozenMapping(item.to_dict()))
        declarations.append(
            _declaration_key(
                SimpleNamespace(
                    contract=TUNING,
                    query=item,
                    invocation=FrozenMapping(),
                    override=None,
                    _device=None,
                )
            )
        )
        compiled_queries.append(
            FrozenMapping(_preparation._program_query_payload(item))
        )
    assert len(set(identities)) == 3
    assert len(set(declarations)) == 3
    assert len(set(compiled_queries)) == 3


def _source(helper, path, experts):
    manifest = write_exl3_checkpoint(
        path,
        Exl3SynthConfig(
            codebook="lut_e4m3",
            num_experts=experts,
            hidden_size=512,
            intermediate_size=512,
            moe_layer_indices=(0,),
            bits=2,
            intermediate_hadamard=True,
            pre_block=512,
            post_block=128,
            seed=4337,
        ),
    )
    layer = helper.read_exl3_layer(path, manifest, 0, first_slot=0, slot_count=8)
    return helper.trellis_from_exl3(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("experts,topk", [(1, 1), (17, 4)])
def test_frozen_and_generic_bindings_preserve_shapes_and_values(
    tmp_path, dtype, experts, topk
):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    helper = _helpers()
    source, weights = _source(helper, tmp_path, experts)
    geometry = helper.fused_moe.MoEGeometry(
        num_experts=experts, hidden_size=512, intermediate_size=256
    )
    prepared = helper.prepare_experts(source, weights, geometry, dtype)
    x, ids, routes = helper.inputs(128, 512, experts, topk, dtype, seed=4391)
    with (
        helper.prepared_runner(
            prepared, x, ids, routes, freeze_transform=False
        ) as generic,
        helper.prepared_runner(prepared, x, ids, routes) as compact,
    ):
        assert generic.scratch_plan.caps.trellis_input_scale_rows is None
        assert compact.scratch_plan.caps.trellis_input_scale_rows == 1
        helper.qualify_pair(
            generic,
            compact,
            x,
            ids,
            routes,
            experts,
            seed=4409,
            scratch_saving_bytes=(
                128 * (topk - 1) * 512 * 2
                - _phase_saving(generic.scratch_plan._core_workspace_plan)
                + _phase_saving(compact.scratch_plan._core_workspace_plan)
            ),
        )
        graphs = []
        for rows in (3, 128, 1, 17, 125, 8):
            pair = []
            for runner in (generic, compact):
                binding = runner.scratch_plan.bind(
                    scratch=runner.scratch,
                    a=x[:rows],
                    experts=prepared._impl,
                    topk_ids=ids[:rows],
                    topk_weights=routes[:rows],
                    output=runner.output[:rows],
                )
                step = helper.Runner(binding, runner.output[:rows], runner.scratch)
                step.capture()
                pair.append(step)
            graphs.append((rows, pair))
        for rows, pair in reversed(graphs):
            generic.output.fill_(7)
            compact.output.fill_(7)
            x.mul_(-0.5)
            ids[:rows, -1] = -1
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            for step in pair:
                step.graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert torch.equal(generic.output, compact.output)
            assert torch.all(compact.output[rows:] == 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_full_rotation_resets_barrier_for_every_shared_arena_replay(tmp_path):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    helper = _helpers()
    source, weights = _source(helper, tmp_path, 17)
    geometry = helper.fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    experts = helper.prepare_experts(source, weights, geometry, torch.bfloat16)
    with ExitStack() as stack:
        runners = {}
        for rows in (128, 4096):
            x, ids, routes = helper.inputs(
                rows, 512, 17, 4, torch.bfloat16, seed=4463 + rows
            )
            runners[rows] = stack.enter_context(
                helper.prepared_runner(
                    experts, x, ids, routes, freeze_transform=False
                )
            )
        storage = torch.empty(
            max(
                runner.scratch_plan.layout.total_nbytes
                for runner in runners.values()
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        for rows, runner in runners.items():
            x, ids, routes = helper.inputs(
                rows, 512, 17, 4, torch.bfloat16, seed=4463 + rows
            )
            runner.binding = runner.scratch_plan.bind(
                scratch=storage,
                a=x,
                experts=experts._impl,
                topk_ids=ids,
                topk_weights=routes,
                output=runner.output,
            )
            runner.capture()

        small = runners[128]
        small.graph.replay()
        torch.cuda.synchronize()
        expected = small.output.clone()
        assert small.binding.kernel_workspace[-2:].tolist() == [0, 3]

        runners[4096].graph.replay()
        torch.cuda.synchronize()
        small.binding.kernel_workspace.fill_(0xA5)
        small.graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(small.output, expected)
        assert small.binding.kernel_workspace[-2:].tolist() == [0, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_frozen_layout_mismatch_rejected_before_scratch_mapping(tmp_path):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    helper = _helpers()
    source, weights = _source(helper, tmp_path, 17)
    other_source, other_weights = helper.expanded_input_scales(source, weights, 17)
    geometry = helper.fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    a = helper.prepare_experts(source, weights, geometry, torch.bfloat16)
    b = helper.prepare_experts(other_source, other_weights, geometry, torch.bfloat16)
    x, ids, routes = helper.inputs(8, 512, 17, 4, torch.bfloat16, seed=4433)
    for accepted, rejected in ((a, b), (b, a)):
        with helper.prepared_runner(accepted, x, ids, routes) as runner:
            with pytest.raises(ValueError, match="input scale rows"):
                runner.scratch_plan.bind(
                    scratch=None,
                    a=x,
                    experts=rejected._impl,
                    topk_ids=ids,
                    topk_weights=routes,
                    output=runner.output,
                )
    with (
        helper.prepared_runner(a, x, ids, routes, freeze_transform=False) as generic,
        helper.prepared_runner(b, x, ids, routes) as per_expert,
    ):
        assert per_expert.scratch_plan.caps.trellis_input_scale_rows == 17
        binding = generic.scratch_plan.bind(
            scratch=generic.scratch,
            a=x,
            experts=b._impl,
            topk_ids=ids,
            topk_weights=routes,
            output=generic.output,
        )
        helper.qualify_pair(
            helper.Runner(binding, generic.output, generic.scratch),
            per_expert,
            x,
            ids,
            routes,
            17,
            seed=4441,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_effective_row_layout_includes_per_expert_gains(tmp_path):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    from b12x.moe.fused_moe.config import TrellisConfig
    from b12x.moe.fused_moe.weights import ScaleFactors

    helper = _helpers()
    source, weights = _source(helper, tmp_path, 17)
    config = source.config.to_dict()
    config["scale"]["input_scales"]["gains"] = "per_expert"
    source = replace(source, config=TrellisConfig.from_dict(config))
    weights = replace(
        weights,
        input_scales=ScaleFactors(
            weights.input_scales.vectors, torch.ones(17, dtype=torch.float16)
        ),
    )
    geometry = helper.fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    prepared = helper.prepare_experts(source, weights, geometry, torch.bfloat16)
    table = prepared._impl.representation_for("w4a16").gate_suh
    assert table.shape == (17, 512)
    assert torch.equal(table[0], table[1])
    x, ids, routes = helper.inputs(8, 512, 17, 4, torch.bfloat16, seed=4457)
    with helper.prepared_runner(prepared, x, ids, routes) as runner:
        assert runner.scratch_plan.caps.trellis_input_scale_rows == 17
        allocation = next(
            s
            for s in runner.scratch_plan._core_workspace_plan.tensor_specs
            if s.name == "rotation_a_gate"
        )
        assert allocation.shape == (32, 512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_legacy_per_expert_pool_stays_conservative_and_exact(tmp_path):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    from b12x.moe.fused_moe.config import TrellisConfig
    from b12x.moe.fused_moe.weights import ScaleFactors

    helper = _helpers()
    source, weights = _source(helper, tmp_path, 17)
    source, weights = helper.expanded_input_scales(source, weights, 17)
    config = source.config.to_dict()
    config["scale"]["output_scales"]["vectors"] = "per_expert"
    source = replace(source, config=TrellisConfig.from_dict(config))
    weights = replace(
        weights,
        output_scales=ScaleFactors(
            weights.output_scales.vectors.expand(17, 512).contiguous()
        ),
    )
    geometry = helper.fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    experts = helper.prepare_experts(source, weights, geometry, torch.bfloat16)
    x, ids, routes = helper.inputs(8, 512, 17, 4, torch.bfloat16, seed=4493)
    with helper.prepared_runner(
        experts, x, ids, routes, output_dtype=torch.float32
    ) as reference:
        reference.capture()
        for row_contract in (None, 17):
            caps = replace(
                reference.scratch_plan.caps, trellis_input_scale_rows=row_contract
            )
            layout = _impl._plan_tp_moe_arena_layout_from_caps(caps)
            arena = torch.zeros(layout.total_nbytes, dtype=torch.uint8, device=x.device)
            pool = _impl.allocate_tp_moe_workspace_pool(
                shared_arena=arena,
                route_workspace_nbytes=layout.route_workspace_nbytes,
                core_workspace_nbytes=layout.core_workspace_nbytes,
            )
            _impl.materialize_tp_moe_arena_workspaces(pool, caps=caps)
            assert all(
                value.trellis_input_scale_rows is None
                for value in pool.workspaces.values()
            )
            assert all(
                value.plan.trellis_input_scale_rows is None
                for value in pool.core_arenas.values()
            )
            assert layout.total_nbytes == _impl.tp_moe_required_nbytes(
                replace(caps, trellis_input_scale_rows=None)
            )
            output = torch.empty_like(reference.output)
            binding = _impl.build_tp_moe_fp4_binding(
                scratch=pool,
                a=x,
                experts=experts._impl,
                topk_ids=ids,
                topk_weights=routes,
                output=output,
                decode_config=caps.decode_config,
            )
            legacy = helper.Runner(binding, output, (arena,))
            legacy.capture()
            for _ in range(3):
                x.mul_(-0.5)
                reference.graph.replay()
                legacy.graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(reference.output, output)
