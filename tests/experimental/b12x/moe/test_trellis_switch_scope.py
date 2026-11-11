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
