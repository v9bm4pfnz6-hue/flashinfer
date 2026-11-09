"""CPU checks for the FP8 dense-MLA probability-scale derivation.

The FP8 PV epilogue derives the HIGH/LOW quantization maximum of a 64-candidate
chunk from the online-softmax block scale exp2(block_max - new_max) instead of a
shared atomic maximum over the stored probabilities. The identity holds because
the largest probability of a warp is exactly one: its argmax element evaluates
exp2(0). Therefore the maximum of probability * warp_scale over the chunk equals
the block scale of the argmax warp, and the scale and reciprocal that feed the
HIGH/LOW quantization are the same floats as before.

These tests model the kernel's four-warp, eight-head, 64-candidate layout in
NumPy and compare the two derivations bit for bit, including chunks whose scores
are masked, chunks that raise the running maximum, and chunks that underflow.
They do not execute the CUDA kernel; the one-GPU ticket
optimization-team/tickets/mla/dense-mla-pv-sync proves the kernel-level equality
by comparing output, final LSE, split partials and partial LSE against the
pre-change kernel on the serving shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

HEADS_PER_TILE = 8
MATH_WARPS_PER_QUERY = 4
CANDIDATES_PER_CHUNK = 64
CANDIDATES_PER_WARP = CANDIDATES_PER_CHUNK // MATH_WARPS_PER_QUERY
FP8_MAX = np.float32(448.0)
MASK = np.float32(-1.0e30)
FTZ_MIN = np.float32(2.0**-126)
SCALE_CLAMP = np.float32(1.0e-10)


def _exp2(value) -> np.float32:
    """Deterministic monotone model of ex2.approx.ftz.f32.

    The kernel relies on the same function being applied to the same argument,
    not on the function's accuracy, so a correctly rounded float32 exponential
    with flush-to-zero reproduces the identity.
    """
    result = np.float32(np.exp2(np.float32(value)))
    if result < FTZ_MIN:
        result = np.float32(0.0)
    return result


def _fmax(left, right) -> np.float32:
    return np.float32(np.maximum(np.float32(left), np.float32(right)))


def _warp_local_maxima(scores: np.ndarray) -> np.ndarray:
    """Per (warp, head) maximum of a (heads, candidates) score chunk."""
    warps = scores.reshape(HEADS_PER_TILE, MATH_WARPS_PER_QUERY, CANDIDATES_PER_WARP)
    return warps.max(axis=2)


def _atomic_maximum(scores: np.ndarray, global_max: np.ndarray, new_max: np.ndarray) -> np.ndarray:
    """The pre-change maximum: max over candidates of |probability * warp_scale|."""
    local = _warp_local_maxima(scores)
    maximum = np.zeros(HEADS_PER_TILE, dtype=np.float32)
    for warp in range(MATH_WARPS_PER_QUERY):
        warp_scale = np.array(
            [_exp2(local[head, warp] - new_max[head]) for head in range(HEADS_PER_TILE)],
            dtype=np.float32,
        )
        for head in range(HEADS_PER_TILE):
            candidate_scores = scores[head, warp * CANDIDATES_PER_WARP:(warp + 1) * CANDIDATES_PER_WARP]
            probabilities = np.array(
                [_exp2(score - local[head, warp]) for score in candidate_scores],
                dtype=np.float32,
            )
            weights = np.array([probabilities * warp_scale[head]], dtype=np.float32)[0]
            maximum[head] = _fmax(maximum[head], np.abs(weights).max())
    return maximum


def _block_scale_maximum(scores: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    """The derived maximum: the online-softmax block scale of each head."""
    local = _warp_local_maxima(scores)
    maximum = np.zeros(HEADS_PER_TILE, dtype=np.float32)
    for head in range(HEADS_PER_TILE):
        block_max = np.float32(local[head].max())
        merged_max = _fmax(global_max[head], block_max)
        maximum[head] = _exp2(block_max - merged_max)
    return maximum


def _scale_and_inverse(maximum: np.ndarray):
    scale = np.array(
        [np.float32(_fmax(value, SCALE_CLAMP)) * np.float32(1.0 / FP8_MAX) for value in maximum],
        dtype=np.float32,
    )
    inverse = np.array([np.float32(1.0) / value for value in scale], dtype=np.float32)
    return scale, inverse


def _new_max(scores: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    local = _warp_local_maxima(scores)
    return np.array(
        [_fmax(global_max[head], np.float32(local[head].max())) for head in range(HEADS_PER_TILE)],
        dtype=np.float32,
    )


def _check_chunk(scores: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    merged_max = _new_max(scores, global_max)
    atomic = _atomic_maximum(scores, global_max, merged_max)
    derived = _block_scale_maximum(scores, global_max)
    assert atomic.tobytes() == derived.tobytes(), (atomic, derived)
    atomic_scale, atomic_inverse = _scale_and_inverse(atomic)
    derived_scale, derived_inverse = _scale_and_inverse(derived)
    assert atomic_scale.tobytes() == derived_scale.tobytes()
    assert atomic_inverse.tobytes() == derived_inverse.tobytes()
    return merged_max


def _scores(seed: int, *, masked: int = 0, spread: float = 8.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = (generator.random((HEADS_PER_TILE, CANDIDATES_PER_CHUNK)) * spread).astype(np.float32)
    if masked:
        indices = generator.choice(CANDIDATES_PER_CHUNK, size=masked, replace=False)
        values[:, indices] = MASK
    return values


@pytest.mark.parametrize("seed", [0, 1, 7, 20261110])
def test_block_scale_is_the_exact_probability_maximum(seed: int) -> None:
    global_max = np.full(HEADS_PER_TILE, MASK, dtype=np.float32)
    for chunk in range(4):
        scores = _scores(seed * 10 + chunk)
        global_max = _check_chunk(scores, global_max)


@pytest.mark.parametrize("masked", [1, 16, 48, 64])
def test_masked_candidates_keep_the_identity(masked: int) -> None:
    global_max = np.full(HEADS_PER_TILE, MASK, dtype=np.float32)
    global_max = _check_chunk(_scores(11, masked=masked), global_max)
    global_max = _check_chunk(_scores(12, masked=0), global_max)
    global_max = _check_chunk(_scores(13, masked=masked), global_max)


def test_late_maximum_rescale_keeps_the_identity() -> None:
    """A dominant key in a later chunk must still produce equal maxima."""
    global_max = np.full(HEADS_PER_TILE, MASK, dtype=np.float32)
    global_max = _check_chunk(_scores(21, spread=0.25), global_max)
    late = _scores(22, spread=0.25)
    late[:, CANDIDATES_PER_CHUNK // 2] = np.float32(24.0)
    global_max = _check_chunk(late, global_max)
    global_max = _check_chunk(_scores(23, spread=0.25), global_max)


def test_underflowing_chunk_uses_the_same_clamped_scale() -> None:
    """After a very large maximum, a much smaller chunk underflows in both paths."""
    global_max = np.array([np.float32(200.0)] * HEADS_PER_TILE, dtype=np.float32)
    scores = np.array([np.float32(-50.0)] * (HEADS_PER_TILE * CANDIDATES_PER_CHUNK), dtype=np.float32)
    scores = scores.reshape(HEADS_PER_TILE, CANDIDATES_PER_CHUNK)
    _check_chunk(scores, global_max)


def test_all_masked_chunk_matches_previous_scale() -> None:
    global_max = np.full(HEADS_PER_TILE, MASK, dtype=np.float32)
    first = np.full((HEADS_PER_TILE, CANDIDATES_PER_CHUNK), MASK, dtype=np.float32)
    global_max = _check_chunk(first, global_max)
    # A second all-masked chunk must keep both derivations at the same value.
    _check_chunk(first, global_max)


def test_serving_shared_memory_layout_is_unchanged() -> None:
    """The optimization keeps every serving specialization's SMEM footprint."""
    pytest.importorskip("cutlass")
    pytest.importorskip("torch")
    from b12x.attention.dense_mla._layout import make_smem_layout

    serving = {
        "target-decode-fp8": dict(query_tile=1, fp8=True),
        "target-prefill-fp8": dict(query_tile=4, fp8=True),
        "drafter-bf16": dict(query_tile=1, fp8=False),
    }
    for name, spec in serving.items():
        layout = make_smem_layout(**spec)
        assert layout.total_bytes <= 101_376, (name, layout.total_bytes)
        # The FP8 reciprocal slots of the probability-scale group remain reserved
        # even though the fused path only writes the eight scale values; the BF16
        # path keeps its single scale float.
        per_tile = 16 * 4 if spec["fp8"] else 4
        assert layout.weight_scale_group_bytes == per_tile, name
        assert layout.weight_scale_bytes == spec["query_tile"] * per_tile, name
