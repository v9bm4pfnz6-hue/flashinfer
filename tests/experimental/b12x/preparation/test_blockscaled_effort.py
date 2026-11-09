"""Packed precision effort widens performance choices without weakening legality."""

from dataclasses import replace

import pytest

from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING
from b12x.preparation import DeviceIdentity


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
LEVELS = ("default", "audit")


def query(recipe="nvfp4", m=8, n=128, k=256, **kwargs):
    return BlockscaledQuery(
        recipe=recipe,
        num_tokens=m,
        in_features=k,
        padded_in_features=k,
        out_features=n,
        **kwargs,
    )


def _hashable(value):
    if hasattr(value, "items"):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def candidates(q, level):
    q = replace(q, autotune_effort=level)
    return {
        _hashable(TUNING.equivalence_key(q, DEVICE, config))
        for _, config in TUNING.choices(q, DEVICE)
    }


@pytest.mark.parametrize(
    "recipe", ("nvfp4", "mxfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0")
)
@pytest.mark.parametrize(
    "m,n,k",
    ((1, 8, 256), (8, 64, 256), (17, 128, 256), (17, 32768, 5120), (65, 64, 256)),
)
def test_levels_are_nested_and_naturally_include_heuristic(recipe, m, n, k):
    q = query(recipe, m, n, k)
    default = TUNING.default_config(q, DEVICE)
    TUNING.validate_config(q, default, DEVICE)
    key = _hashable(TUNING.equivalence_key(q, DEVICE, default))
    previous = set()
    for level in LEVELS:
        choices = candidates(q, level)
        assert previous <= choices
        assert key in choices
        previous = choices


@pytest.mark.parametrize(
    "q,tile_m,tile_n,tile_k,mma_warps,rule",
    (
        (query(m=8), 64, 64, 64, None, "efficiency"),
        (query("mxfp8", m=16, n=64), 16, 128, 64, 4, "efficiency"),
        # N4 tiles beyond eight tokens are dominated at every effort.
        (query("iq2_xs", m=16), 4, 4, 256, None, "correctness"),
    ),
)
def test_performance_filters_lift_at_audit_and_dominance_rules_hold(
    q, tile_m, tile_n, tile_k, mma_warps, rule
):
    choice = dict(
        mode="a16",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=1,
        mma_warps=mma_warps,
    )
    with pytest.raises(ValueError, match=rule):
        TUNING.lower(replace(q, autotune_effort="default"), DEVICE, choice)
    audit = replace(q, autotune_effort="audit")
    if rule == "efficiency":
        TUNING.lower(audit, DEVICE, choice)
    else:
        with pytest.raises(ValueError, match=rule):
            TUNING.lower(audit, DEVICE, choice)


@pytest.mark.parametrize("level", LEVELS)
def test_mxfp8_row_support_precision_and_workspace_stay_mandatory(level):
    q = query("mxfp8", m=64, autotune_effort=level)
    choice = dict(mode="a16", tile_m=32, tile_n=64, tile_k=64, split_k=1, mma_warps=4)
    with pytest.raises(ValueError):
        TUNING.lower(q, DEVICE, choice)
    with pytest.raises(ValueError):
        TUNING.lower(
            replace(q, activation_mode="quantized"), DEVICE, choice | dict(tile_m=16)
        )
    with pytest.raises(ValueError, match="workspace"):
        TUNING.lower(
            replace(q, workspace_nbytes=0), DEVICE, choice | dict(tile_m=16, split_k=4)
        )


def test_effort_is_captured_in_declaration_but_excluded_from_program_key(monkeypatch):
    from b12x.gemm.blockscaled._preparation import _compile_query_payload, _decode_query

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    q = query()
    assert q.autotune_effort == "audit"
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "off")
    assert q.autotune_effort == "audit"
    assert TUNING.encode_query(q) != TUNING.encode_query(
        replace(q, autotune_effort="off")
    )
    payload = _compile_query_payload(q)
    assert payload == _compile_query_payload(replace(q, autotune_effort="off"))
    assert "autotune_effort" not in payload
    assert _decode_query(payload).autotune_effort == "default"


def test_packed_a16_backend_is_raced_at_default_for_verification_rows():
    verification = BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=128,
        in_features=7168,
        padded_in_features=7168,
        out_features=1408,
        activation_mode="a16",
        allow_bf16_atomic_reduction=False,
        workspace_form="provided",
        workspace_nbytes=64 << 20,
    )
    space = TUNING.parameter_space(verification, DEVICE)
    assert "packed" in {
        assignment["a16_backend"] for assignment in space.configurations()
    }
    assert "packed" in {
        assignment["a16_backend"]
        for assignment in replace(space, autotune_effort="audit").configurations()
    }

    # The expanded BF16 lowering wins once the weight expansion is amortized;
    # the prefill bucket keeps the packed backend audit-only.
    prefill = replace(verification, num_tokens=4608)
    space = TUNING.parameter_space(prefill, DEVICE)
    assert "packed" not in {
        assignment["a16_backend"] for assignment in space.configurations()
    }
    assert "packed" in {
        assignment["a16_backend"]
        for assignment in replace(space, autotune_effort="audit").configurations()
    }
