"""CuTeDSL kernels for single-rounding BF16 PCIe two-shot collectives.

The reduce-scatter and all-gather launchers mirror :mod:`_twoshot_cute` (the
fp8-transport variant): phase one pushes this rank's shard into every peer's
IPC staging slot (posted PCIe writes), a per-CTA flag barrier follows, and
phase two either reduces the local rank's shard (reduce_scatter) or copies the
staged shards into place (all_gather). The pull all-reduce launcher instead
stages locally and reads its peers' slabs (:class:`_TwoShotPullAllReduceLaunch`).
Payload packs are 16 bytes = 8 bf16 values; reductions accumulate in fp32 in a
fixed rank order (local rank first, then ``(local + i) % world`` for
``i = 1..world-1``) and round once to bf16, so a given rank's output is
deterministic across runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics.memory import (
    ld_global_nc_v4_u32,
    ld_global_v4_u32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._cute_intrinsics import (
    f32_as_u32,
    graph_epoch_arrive_serialized,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
    unpack_bf16x2,
)
from ._twoshot_cute import (
    _GRAPH_BLOCKS_ARRIVED_OFFSET,
    _GRAPH_EPOCH_OFFSET,
    _MAX_BLOCKS,
    _MAX_RANKS,
    _FLAG_STRIDE,
    _SELF_COUNTER_BYTES,
    _fence_sc_sys,
    _ld_generic_v4_u32,
    _ld_global_u32,
    _st_generic_v4_u32,
    _st_global_u32,
    _st_relaxed_sys_u32,
)

_PACK_ELEMS = 8  # bf16 values per 16-byte pack
_SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8, 9, 10)


@dsl_user_op
def _ld_acquire_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Barrier flag load that acquires the peer's ``fence.sc.sys`` release."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _TwoShotBf16Launch:
    def __init__(
        self,
        operation: str,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
    ) -> None:
        if operation not in ("reduce_scatter", "all_gather"):
            raise ValueError(f"invalid two-shot operation {operation!r}")
        self._operation = operation
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)
        self._row_elems = int(row_elems)

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        slot_bytes: Int64,
        shard_packs: Int64,
        total_packs: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            output,
            rank,
            pack_stride,
            slot_bytes,
            shard_packs,
            total_packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _select_address(
        self,
        pointers: Sequence[cute.Pointer],
        index: Int32,
    ) -> Int64:
        """Select one scalar launch pointer without unrolling peer work."""

        address = Int64(pointers[0].toint())
        for peer in cutlass.range_constexpr(1, self._world_size):
            if index == Int32(peer):
                address = Int64(pointers[peer].toint())
        return address

    @cute.jit
    def _barrier(
        self,
        signals: Sequence[cute.Pointer],
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.barrier()
        if tidx < Int32(self._world_size):
            _fence_sc_sys()
            if cutlass.const_expr(self._operation == "all_gather"):
                self_base = Int64(signals[self._rank].toint())
            else:
                self_base = self._select_address(signals, rank)
            self_counter_address = self_base + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            value = _ld_global_u32(self_counter_address) + Uint32(1)
            _st_global_u32(self_counter_address, value)

            flag_slot = Int64(value % Uint32(2))
            peer_base = self._select_address(signals, tidx)
            peer_counter_address = (
                peer_base
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    (flag_slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(rank) * Int64(_FLAG_STRIDE)
                )
                * Int64(4)
            )
            self_counter_address = (
                self_base
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    (flag_slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(tidx) * Int64(_FLAG_STRIDE)
                )
                * Int64(4)
            )
            _st_relaxed_sys_u32(peer_counter_address, value)
            # A relaxed load followed by the CTA barrier would not order this
            # block's later peer-data loads after the peer's release; the
            # system-scope acquire does, and the CTA barrier extends it to
            # every thread.
            observed = _ld_acquire_sys_u32(self_counter_address)
            while observed != value:
                observed = _ld_acquire_sys_u32(self_counter_address)
        cute.arch.barrier()

    @cute.jit
    def _accumulate_words(
        self,
        accumulator: cute.Tensor,
        words,
        initialize: cutlass.Constexpr = False,
    ) -> None:
        for word_index in cutlass.range_constexpr(4):
            lo, hi = unpack_bf16x2(words[word_index])
            element = word_index * 2
            if cutlass.const_expr(initialize):
                # An ordered sum starts with its first input, not an implicit
                # positive zero. Preserve negative zero when every input is -0.
                accumulator[element] = lo
                accumulator[element + 1] = hi
            else:
                accumulator[element] = accumulator[element] + lo
                accumulator[element + 1] = accumulator[element + 1] + hi

    @cute.jit
    def _load_accumulate_pack_global_nc(
        self,
        accumulator: cute.Tensor,
        address: Int64,
        initialize: cutlass.Constexpr = False,
    ) -> None:
        words = ld_global_nc_v4_u32(address)
        self._accumulate_words(accumulator, words, initialize)

    @cute.jit
    def _load_accumulate_pack_generic(
        self,
        accumulator: cute.Tensor,
        address: Int64,
    ) -> None:
        words = _ld_generic_v4_u32(address)
        self._accumulate_words(accumulator, words)

    @cute.jit
    def _store_pack(self, output_address: Int64, accumulator: cute.Tensor) -> None:
        st_global_v4_u32(
            output_address,
            pack_f32x2_to_bf16x2(accumulator[0], accumulator[1]),
            pack_f32x2_to_bf16x2(accumulator[2], accumulator[3]),
            pack_f32x2_to_bf16x2(accumulator[4], accumulator[5]),
            pack_f32x2_to_bf16x2(accumulator[6], accumulator[7]),
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        slot_bytes: Int64,
        shard_packs: Int64,
        total_packs: Int64,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        block_threads, _, _ = cute.arch.block_dim()
        if cutlass.const_expr(self._operation == "all_gather"):
            local_rank = Int32(self._rank)
        else:
            local_rank = rank
        chunk = (shard_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > shard_packs:
            end = shard_packs

        payload_address = Int64(payload.toint())
        staging_slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            if cutlass.const_expr(self._operation == "all_gather"):
                self_signal = Int64(signals[self._rank].toint())
            else:
                self_signal = self._select_address(signals, local_rank)
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )

        # Phase one: push remote shards (posted PCIe writes), rank-staggered.
        peer_index = Int32(1)
        while peer_index < Int32(self._world_size):
            destination = (local_rank + peer_index) % Int32(self._world_size)
            destination_base = (
                self._select_address(staging, destination) + staging_slot_offset
            )
            destination_payload = destination_base + (
                Int64(local_rank) * pack_stride * Int64(16)
            )
            if cutlass.const_expr(self._operation == "reduce_scatter"):
                source_pack = Int64(destination) * shard_packs
            else:
                source_pack = Int64(0)

            index = begin + Int64(tidx)
            while index < end:
                valid_pack = source_pack + index
                if cutlass.const_expr(self._operation == "all_gather"):
                    valid_pack = Int64(local_rank) * shard_packs + index
                if valid_pack < total_packs:
                    words = ld_global_nc_v4_u32(
                        payload_address + (source_pack + index) * Int64(16)
                    )
                    _st_generic_v4_u32(
                        destination_payload + index * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                index += Int64(block_threads)
            peer_index += Int32(1)

        self._barrier(signals, local_rank)

        if cutlass.const_expr(self._operation == "all_gather"):
            self_base = Int64(staging[self._rank].toint())
        else:
            self_base = self._select_address(staging, local_rank)
        self_base += staging_slot_offset
        output_address = Int64(output.toint())
        if cutlass.const_expr(self._operation == "reduce_scatter"):
            index = begin + Int64(tidx)
            while index < end:
                accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                for lane in cutlass.range_constexpr(_PACK_ELEMS):
                    accumulator[lane] = Float32(0.0)

                local_pack = Int64(local_rank) * shard_packs + index
                if local_pack < total_packs:
                    self._load_accumulate_pack_global_nc(
                        accumulator,
                        payload_address + local_pack * Int64(16),
                        True,
                    )

                    for peer_index in cutlass.range(
                        Int32(1),
                        Int32(self._world_size),
                        Int32(1),
                        unroll=1,
                    ):
                        source_rank = (local_rank + peer_index) % Int32(
                            self._world_size
                        )
                        staged_pack = (
                            self_base
                            + Int64(source_rank) * pack_stride * Int64(16)
                            + index * Int64(16)
                        )
                        self._load_accumulate_pack_generic(accumulator, staged_pack)

                self._store_pack(output_address + index * Int64(16), accumulator)
                index += Int64(self._threads)
        else:
            first_index = begin + Int64(tidx)
            iteration_count = Int32(
                (end - first_index + Int64(self._threads - 1)) // Int64(self._threads)
            )
            peer_index = Int32(0)
            index = Int64(0)
            while peer_index < Int32(self._world_size):
                source_rank = (local_rank + peer_index) % Int32(self._world_size)
                source_payload_base = Int64(0)
                if source_rank == local_rank:
                    source_payload_base = payload_address
                else:
                    source_payload_base = self_base + Int64(
                        source_rank
                    ) * pack_stride * Int64(16)
                destination_base = output_address + Int64(
                    source_rank
                ) * shard_packs * Int64(16)
                for iteration in cutlass.range(
                    Int32(0),
                    iteration_count,
                    Int32(1),
                    unroll=1,
                ):
                    index = first_index + Int64(iteration) * Int64(self._threads)
                    if Int64(source_rank) * shard_packs + index < total_packs:
                        # Generic addressing serves both the local payload and the
                        # IPC-mapped peer slabs.
                        words = _ld_generic_v4_u32(
                            source_payload_base + index * Int64(16)
                        )
                        st_global_v4_u32(
                            destination_base + index * Int64(16),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                peer_index += Int32(1)


def _bf16_process_key(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        "bf16",
        str(operation),
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(row_elems),
        int(device_index),
    )


@program_cache
def get_twoshot_bf16_launcher(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile and return one static world/operation/thread specialization."""
    process_key = _bf16_process_key(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index  # part of the process-local cache key
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"BF16 two-shot launchers support world sizes {_SUPPORTED_WORLD_SIZES}, "
            f"got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    slot_bias = int(slot_bias) & 1
    launch = _TwoShotBf16Launch(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
    )
    cache_key = (
        "bf16",
        operation,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for _ in range(16)
        ),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for _ in range(16)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        0,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot_bf16.{operation}",
            3,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        pack_stride: int,
        slot_bytes: int,
        shard_packs: int,
        total_packs: int,
        grid_x: int,
    ) -> None:
        if len(staging_addresses) != 16 or len(signal_addresses) != 16:
            raise ValueError("two-shot scalar pointer ABI requires sixteen peers")
        raw_args = (
            make_ptr(
                cutlass.Uint32,
                payload_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for address in staging_addresses
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for address in signal_addresses
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            rank,
            pack_stride,
            slot_bytes,
            shard_packs,
            total_packs,
            grid_x,
            current_cuda_stream(),
        )
        raw(*raw_args)

    return attach_programs(run, raw)


class _TwoShotPullAllReduceLaunch(_TwoShotBf16Launch):
    """Single-launch BF16 all-reduce with one rounding, using remote reads.

    The kernel copies the shards its peers reduce from this rank's payload into
    its own IPC slab (a local copy; the local shard is reduced straight from
    the payload). After the first barrier, every rank pulls its shard (P/world
    for a per-rank payload of P bytes) from every peer's staging, accumulates
    in fp32 (rank order: self, then ``(rank + i) % world``), rounds once to
    bf16 and writes the reduced shard both to the output and to its slab's
    reduced region. After the second barrier, every rank pulls the other
    ranks' reduced shards into the output. PCIe read volume per rank is
    2P*(world - 1)/world, which is 1.5P for world size 4; synchronization uses
    two barriers.

    The barriers pair block ``b`` only with block ``b`` of each peer, so every
    phase walks the same shard-local pack indices: shard-local pack ``i`` of
    every shard is staged, reduced and published by the block that owns ``i``
    on every rank.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
    ) -> None:
        super().__init__(
            "reduce_scatter",  # barrier addressing uses the dynamic-rank path
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
        )

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        reduced_offset: Int64,
        slot_bytes: Int64,
        shard_packs: Int64,
        total_packs: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            output,
            rank,
            reduced_offset,
            slot_bytes,
            shard_packs,
            total_packs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _load_accumulate_pack_remote(
        self,
        accumulator: cute.Tensor,
        address: Int64,
    ) -> None:
        words = ld_global_v4_u32(address)
        self._accumulate_words(accumulator, words)

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        reduced_offset: Int64,
        slot_bytes: Int64,
        shard_packs: Int64,
        total_packs: Int64,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        local_rank = rank
        threads = Int64(self._threads)
        grid_threads = Int64(gdim) * threads
        flat = Int64(bidx) * threads + Int64(tidx)

        payload_address = Int64(payload.toint())
        output_address = Int64(output.toint())
        staging_slot_offset = Int64(0)
        self_signal = self._select_address(signals, local_rank)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )
        self_base = self._select_address(staging, local_rank) + staging_slot_offset

        # Stage the peers' shards. Pack ``index`` of each shard must come from
        # the block that reduces ``index`` on the peer: the barrier waits only
        # for the same block index on every peer, so a full-payload grid
        # stride would leave remote reads unordered against their staging.
        # The last shard may end early (packed tail): packs at or past
        # ``total_packs`` are neither staged, reduced nor copied.
        index = flat
        while index < shard_packs:
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                shard_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                pack = Int64(shard_rank) * shard_packs + index
                if pack < total_packs:
                    words = ld_global_nc_v4_u32(payload_address + pack * Int64(16))
                    st_global_v4_u32(
                        self_base + pack * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Pull and reduce this rank's shard from every peer's staging.
        shard_base = Int64(local_rank) * shard_packs
        index = flat
        while index < shard_packs:
            if shard_base + index < total_packs:
                accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                for lane in cutlass.range_constexpr(_PACK_ELEMS):
                    accumulator[lane] = Float32(0.0)
                # Issue the local and all remote pack loads back-to-back (the peer
                # loop is unrolled at compile time), then accumulate in fixed
                # rank order: self, then (rank + i) % world for i = 1..world-1.
                pack_offset = (shard_base + index) * Int64(16)
                local_words = ld_global_nc_v4_u32(payload_address + pack_offset)
                peer_words = []
                for peer_index in cutlass.range_constexpr(1, self._world_size):
                    source_rank = (local_rank + Int32(peer_index)) % Int32(
                        self._world_size
                    )
                    peer_base = (
                        self._select_address(staging, source_rank) + staging_slot_offset
                    )
                    peer_words.append(ld_global_v4_u32(peer_base + pack_offset))
                self._accumulate_words(accumulator, local_words, True)
                for peer_index in cutlass.range_constexpr(self._world_size - 1):
                    self._accumulate_words(accumulator, peer_words[peer_index])
                self._store_pack(
                    output_address + (shard_base + index) * Int64(16), accumulator
                )
                self._store_pack(
                    self_base + reduced_offset + index * Int64(16), accumulator
                )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Copy the other ranks' published reduced shards into the output.
        for peer_index in cutlass.range_constexpr(1, self._world_size):
            source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
            peer_reduced = (
                self._select_address(staging, source_rank)
                + staging_slot_offset
                + reduced_offset
            )
            destination = output_address + Int64(source_rank) * shard_packs * Int64(16)
            index = flat
            while index < shard_packs:
                if Int64(source_rank) * shard_packs + index < total_packs:
                    words = ld_global_v4_u32(peer_reduced + index * Int64(16))
                    st_global_v4_u32(
                        destination + index * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                index += grid_threads


@program_cache
def get_twoshot_bf16_allreduce_launcher(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile the single-launch pull-based bf16 all-reduce specialization."""
    process_key = _bf16_process_key(
        "all_reduce_pull",
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index, process_key
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"BF16 two-shot launchers support world sizes {_SUPPORTED_WORLD_SIZES}, "
            f"got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    slot_bias = int(slot_bias) & 1
    launch = _TwoShotPullAllReduceLaunch(
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
    )
    cache_key = (
        "bf16",
        "all_reduce_pull",
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for _ in range(16)
        ),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for _ in range(16)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        0,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.twoshot_bf16.all_reduce_pull",
            5,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        reduced_offset: int,
        slot_bytes: int,
        shard_packs: int,
        total_packs: int,
        grid_x: int,
    ) -> None:
        if len(staging_addresses) != 16 or len(signal_addresses) != 16:
            raise ValueError("two-shot scalar pointer ABI requires sixteen peers")
        raw_args = (
            make_ptr(
                cutlass.Uint32,
                payload_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for address in staging_addresses
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for address in signal_addresses
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            rank,
            reduced_offset,
            slot_bytes,
            shard_packs,
            total_packs,
            grid_x,
            current_cuda_stream(),
        )
        raw(*raw_args)

    return attach_programs(run, raw)


__all__ = ["get_twoshot_bf16_launcher", "get_twoshot_bf16_allreduce_launcher"]


class _TwoShotIslandAllReduceLaunch(_TwoShotBf16Launch):
    """Single-launch 2-island hierarchical BF16 all-reduce for world size 8.

    Topology contract (B12X_PCIE_TWOSHOT_ISLAND): ranks 0-3 share one PCIe
    switch (island A), ranks 4-7 the other (island B), and rank ``r``'s
    cross-island partner is ``r ^ 4``. Only partner pairs cross the island
    uplink, once, carrying each island's FP32 partial for one quarter.

    Quarter ``q`` (``shard_packs`` = total/4 packs) is owned by island-local
    rank ``q`` in both islands. Phases, each followed by the per-CTA barrier
    (block ``b`` pairs only with block ``b`` of every peer, and every phase
    walks the same quarter-local pack indices):

    1. push quarter ``m`` of the payload into island peer ``m``'s staging
       region ``local`` (posted writes, island-local only);
    2. FP32 island partial of the owned quarter, summed in island-local rank
       order 0..3 over the original BF16 inputs; written as FP32 into this
       rank's own output (low half into quarter ``local``, high half into
       quarter ``local + 1``; both are overwritten later);
    3. push the FP32 partial (or its BF16 rounding when ``wire_bf16``) into
       the partner's staging regions 0 (low half) and 1 (high half) at the
       same 16-byte pack index, so every CTA touches only its own packs;
    4. final = own partial + partner partial (two-term FP32 sums are
       commutative, so both partners round the identical value), rounded once
       to BF16 into the owned output quarter;
    5. push the owned quarter into the island peers' staging region ``local``;
       after the barrier, copy the three received quarters into the output.

    All eight ranks therefore hold bitwise-identical results. Staging use stays
    within the push two-shot slot (4 regions of 2*pack_stride packs).
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
        wire_bf16: bool = False,
    ) -> None:
        if int(world_size) != 8:
            raise ValueError("island two-shot all-reduce requires world size 8")
        super().__init__(
            "reduce_scatter",  # barrier addressing uses the dynamic-rank path
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
        )
        self._wire_bf16 = bool(wire_bf16)

    @cute.jit
    def _final_store_bf16(self, address: Int64, values: cute.Tensor) -> None:
        st_global_v4_u32(
            address,
            pack_f32x2_to_bf16x2(values[0], values[1]),
            pack_f32x2_to_bf16x2(values[2], values[3]),
            pack_f32x2_to_bf16x2(values[4], values[5]),
            pack_f32x2_to_bf16x2(values[6], values[7]),
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        staging9: cute.Pointer,
        staging10: cute.Pointer,
        staging11: cute.Pointer,
        staging12: cute.Pointer,
        staging13: cute.Pointer,
        staging14: cute.Pointer,
        staging15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        slot_bytes: Int64,
        shard_packs: Int64,
        total_packs: Int64,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            staging9,
            staging10,
            staging11,
            staging12,
            staging13,
            staging14,
            staging15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        island_base = self._rank & 4
        local = self._rank & 3
        partner = self._rank ^ 4
        next_local = (local + 1) & 3
        # shard_packs = packs per quarter (total_packs = 4 * shard_packs).
        chunk = (shard_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > shard_packs:
            end = shard_packs
        region_bytes = Int64(2) * pack_stride * Int64(16)

        payload_address = Int64(payload.toint())
        output_address = Int64(output.toint())
        staging_slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            self_signal = Int64(signals[self._rank].toint())
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )
        self_base = Int64(staging[self._rank].toint()) + staging_slot_offset
        partner_base = Int64(staging[partner].toint()) + staging_slot_offset

        # Phase 1: island reduce-scatter pushes (rank-staggered, local peers only).
        for step in cutlass.range_constexpr(1, 4):
            peer_local = (local + step) & 3
            destination = (
                Int64(staging[island_base + peer_local].toint())
                + staging_slot_offset
                + Int64(local) * region_bytes
            )
            source = payload_address + Int64(peer_local) * shard_packs * Int64(16)
            index = begin + Int64(tidx)
            while index < end:
                words = ld_global_nc_v4_u32(source + index * Int64(16))
                _st_generic_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += Int64(self._threads)
        self._barrier(signals, rank)

        # Phase 2: FP32 island partial of the owned quarter, fixed order 0..3.
        own_low = output_address + Int64(local) * shard_packs * Int64(16)
        own_high = output_address + Int64(next_local) * shard_packs * Int64(16)
        index = begin + Int64(tidx)
        while index < end:
            accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
            for source_local in cutlass.range_constexpr(4):
                if cutlass.const_expr(source_local == local):
                    words = ld_global_nc_v4_u32(
                        payload_address
                        + (Int64(local) * shard_packs + index) * Int64(16)
                    )
                else:
                    words = _ld_generic_v4_u32(
                        self_base
                        + Int64(source_local) * region_bytes
                        + index * Int64(16)
                    )
                self._accumulate_words(accumulator, words, source_local == 0)
            if cutlass.const_expr(self._wire_bf16):
                packed = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = pack_f32x2_to_bf16x2(
                        accumulator[word * 2], accumulator[word * 2 + 1]
                    )
                st_global_v4_u32(
                    own_low + index * Int64(16), packed[0], packed[1], packed[2], packed[3]
                )
            else:
                st_global_v4_u32(
                    own_low + index * Int64(16),
                    f32_as_u32(accumulator[0]),
                    f32_as_u32(accumulator[1]),
                    f32_as_u32(accumulator[2]),
                    f32_as_u32(accumulator[3]),
                )
                st_global_v4_u32(
                    own_high + index * Int64(16),
                    f32_as_u32(accumulator[4]),
                    f32_as_u32(accumulator[5]),
                    f32_as_u32(accumulator[6]),
                    f32_as_u32(accumulator[7]),
                )
            index += Int64(self._threads)
        self._barrier(signals, rank)

        # Phase 3: one cross-island push of the partial to the partner.
        index = begin + Int64(tidx)
        while index < end:
            low = ld_global_v4_u32(own_low + index * Int64(16))
            if cutlass.const_expr(self._wire_bf16):
                _st_generic_v4_u32(
                    partner_base + index * Int64(16), low[0], low[1], low[2], low[3]
                )
            else:
                high = ld_global_v4_u32(own_high + index * Int64(16))
                _st_generic_v4_u32(
                    partner_base + index * Int64(16), low[0], low[1], low[2], low[3]
                )
                _st_generic_v4_u32(
                    partner_base + region_bytes + index * Int64(16),
                    high[0],
                    high[1],
                    high[2],
                    high[3],
                )
            index += Int64(self._threads)
        self._barrier(signals, rank)

        # Phase 4: final = own + partner (commutative two-term FP32 sum), one rounding.
        index = begin + Int64(tidx)
        while index < end:
            total = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
            if cutlass.const_expr(self._wire_bf16):
                mine = ld_global_v4_u32(own_low + index * Int64(16))
                theirs = _ld_generic_v4_u32(self_base + index * Int64(16))
                self._accumulate_words(total, mine, True)
                self._accumulate_words(total, theirs)
            else:
                mine_low = ld_global_v4_u32(own_low + index * Int64(16))
                mine_high = ld_global_v4_u32(own_high + index * Int64(16))
                theirs_low = _ld_generic_v4_u32(self_base + index * Int64(16))
                theirs_high = _ld_generic_v4_u32(
                    self_base + region_bytes + index * Int64(16)
                )
                for lane in cutlass.range_constexpr(4):
                    total[lane] = u32_as_f32(mine_low[lane]) + u32_as_f32(
                        theirs_low[lane]
                    )
                    total[lane + 4] = u32_as_f32(mine_high[lane]) + u32_as_f32(
                        theirs_high[lane]
                    )
            self._final_store_bf16(own_low + index * Int64(16), total)
            index += Int64(self._threads)
        self._barrier(signals, rank)

        # Phase 5: island all-gather pushes of the owned quarter.
        for step in cutlass.range_constexpr(1, 4):
            peer_local = (local + step) & 3
            destination = (
                Int64(staging[island_base + peer_local].toint())
                + staging_slot_offset
                + Int64(local) * region_bytes
            )
            index = begin + Int64(tidx)
            while index < end:
                words = ld_global_v4_u32(own_low + index * Int64(16))
                _st_generic_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += Int64(self._threads)
        self._barrier(signals, rank)
        for step in cutlass.range_constexpr(1, 4):
            source_local = (local + step) & 3
            source = self_base + Int64(source_local) * region_bytes
            destination = output_address + Int64(source_local) * shard_packs * Int64(16)
            index = begin + Int64(tidx)
            while index < end:
                words = _ld_generic_v4_u32(source + index * Int64(16))
                st_global_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += Int64(self._threads)


@program_cache
def get_twoshot_bf16_island_launcher(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
    wire_bf16: bool = False,
) -> Callable[..., None]:
    """Compile the single-launch 2-island hierarchical all-reduce (world 8)."""
    del device_index  # part of the process-local cache key
    if int(world_size) != 8:
        raise ValueError("island two-shot all-reduce requires world size 8")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % _PACK_ELEMS != 0:
        raise ValueError("row_elems must be a positive multiple of 8")
    slot_bias = int(slot_bias) & 1
    launch = _TwoShotIslandAllReduceLaunch(
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        wire_bf16,
    )
    name = "all_reduce_island_bf16" if wire_bf16 else "all_reduce_island"
    cache_key = (
        "bf16",
        name,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        *(
            make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)
            for _ in range(16)
        ),
        *(
            make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)
            for _ in range(16)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        0,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot_bf16.{name}",
            1,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        pack_stride: int,
        slot_bytes: int,
        shard_packs: int,
        total_packs: int,
        grid_x: int,
    ) -> None:
        if len(staging_addresses) != 16 or len(signal_addresses) != 16:
            raise ValueError("two-shot scalar pointer ABI requires sixteen peers")
        raw(
            make_ptr(cutlass.Uint32, payload_address, cute.AddressSpace.gmem, assumed_align=16),
            *(
                make_ptr(cutlass.Uint32, address, cute.AddressSpace.gmem, assumed_align=16)
                for address in staging_addresses
            ),
            *(
                make_ptr(cutlass.Uint32, address, cute.AddressSpace.gmem, assumed_align=4)
                for address in signal_addresses
            ),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            rank,
            pack_stride,
            slot_bytes,
            shard_packs,
            total_packs,
            grid_x,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
