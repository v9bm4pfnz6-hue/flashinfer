# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the pitched copy-engine primitive."""

from __future__ import annotations

import pytest

from b12x.comm.pcie._dma_kernels import DmaKernels


class _RecordingIPC:
    def __init__(self) -> None:
        self.calls = []

    def cudaMemcpy2DAsync(self, *arguments) -> None:
        self.calls.append(arguments)


def test_pitched_copy_reports_its_arguments():
    ipc = _RecordingIPC()
    kernels = DmaKernels(ipc)
    kernels.dma_copy_2d(4096, 11232, 8192, 1248, 1248, 1024, stream=7)
    assert len(ipc.calls) == 1
    dst, dst_pitch, src, src_pitch, width, height, stream = ipc.calls[0]
    assert (dst, dst_pitch, src, src_pitch, width, height, stream) == (
        4096, 11232, 8192, 1248, 1248, 1024, 7,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 0, 0, 0, 0, 0),
        (0, 1132, 0, 1132, 1126, 8),
        (0, 1130, 0, 1136, 1124, 8),
        (0, 1128, 0, 1200, 1216, 8),
    ],
)
def test_pitched_copy_rejects_unaligned_or_inconsistent_geometry(arguments):
    ipc = _RecordingIPC()
    kernels = DmaKernels(ipc)
    with pytest.raises(ValueError):
        kernels.dma_copy_2d(*arguments, stream=0)
    assert ipc.calls == []
