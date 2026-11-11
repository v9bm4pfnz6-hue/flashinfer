"""Scope of the opt-in Trellis launch switches."""

import pytest

from b12x.moe._shared.kernels.w4a16.kernel.config import _trellis_launch_switch


def test_trellis_switch_scope_limits_switches_to_decode_launches(monkeypatch):
    monkeypatch.setenv("B12X_TRELLIS_FLOW_FLAGS", "1")
    monkeypatch.delenv("B12X_TRELLIS_SWITCH_SCOPE", raising=False)
    # Without a scope a switch applies to every launch, as before.
    assert [_trellis_launch_switch("B12X_TRELLIS_FLOW_FLAGS", m) for m in (1, 127, 128, 4608)] == ["1"] * 4
    monkeypatch.setenv("B12X_TRELLIS_SWITCH_SCOPE", "decode")
    assert [_trellis_launch_switch("B12X_TRELLIS_FLOW_FLAGS", m) for m in (1, 32, 127)] == ["1"] * 3
    assert [_trellis_launch_switch("B12X_TRELLIS_FLOW_FLAGS", m) for m in (128, 2048, 4608)] == ["0"] * 3
    # An unset switch reads its default in both scopes.
    assert _trellis_launch_switch("B12X_TRELLIS_LUT_CP_ASYNC", 4) == "0"
    monkeypatch.setenv("B12X_TRELLIS_SWITCH_SCOPE", "prefill")
    with pytest.raises(ValueError, match="B12X_TRELLIS_SWITCH_SCOPE"):
        _trellis_launch_switch("B12X_TRELLIS_FLOW_FLAGS", 4)


def test_prefill_safe_scope_passes_table_staging_and_side_prefetch_to_large_launches(monkeypatch):
    switches = ("B12X_TRELLIS_LUT_CP_ASYNC", "B12X_TRELLIS_PREFETCH_SIDES",
                "B12X_TRELLIS_INKERNEL_ROUTE_PACK", "B12X_TRELLIS_FLOW_FLAGS")
    for name in switches:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("B12X_TRELLIS_SWITCH_SCOPE", "prefill_safe")
    # Launches of fewer than 128 tokens read every switch, as in the decode scope.
    for name in switches:
        assert [_trellis_launch_switch(name, m) for m in (1, 32, 127)] == ["1"] * 3
    # Larger launches read the table staging and the side-table prefetch; route packing and the
    # per-block counters keep their defaults there.
    for m in (128, 2048, 4608):
        assert [_trellis_launch_switch(name, m) for name in switches] == ["1", "1", "0", "0"]
    # A switch that is not set reads its default in this scope too.
    monkeypatch.delenv("B12X_TRELLIS_PREFETCH_SIDES")
    assert [_trellis_launch_switch("B12X_TRELLIS_PREFETCH_SIDES", m) for m in (4, 4608)] == ["0", "0"]
