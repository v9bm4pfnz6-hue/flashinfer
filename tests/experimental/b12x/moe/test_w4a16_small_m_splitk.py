"""The per-launch stripe split-K choice of the full-rotation Trellis kernel."""

import pytest

from flashinfer.experimental.b12x.moe._shared.kernels.w4a16.kernel.config import (
    _w4a16_small_m_splitk_enabled,
    _w4a16_trellis_decode_splitk,
)

SWITCH = "B12X_W4A16_SMALL_M_SPLITK"


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_other_switch_values_leave_the_choice_to_the_kernel_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert all(_w4a16_trellis_decode_splitk(m) is None for m in (1, 24, 127, 128, 4608))
    assert _w4a16_small_m_splitk_enabled() is (value == "1")


def test_decode_mode_splits_decode_launches_and_keeps_prefill_whole_tile(monkeypatch):
    monkeypatch.setenv(SWITCH, "decode")
    assert [_w4a16_trellis_decode_splitk(m) for m in (1, 6, 24, 32, 127)] == [True] * 5
    assert [_w4a16_trellis_decode_splitk(m) for m in (128, 256, 2048, 4608)] == [False] * 4
    # Callers that do not resolve a launch keep whole-tile scheduling in this mode.
    assert _w4a16_small_m_splitk_enabled() is False
