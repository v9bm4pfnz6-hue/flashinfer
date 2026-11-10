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


_ISLAND_OVERLAP_STAGES = 8  # quarter sub-chunks in the software pipeline
_ISLAND_OVERLAP_AREA_UNITS = 10  # 16-byte units per sub-chunk pack per parity


class _TwoShotIslandOverlapAllReduceLaunch(_TwoShotIslandAllReduceLaunch):
    """Software-pipelined variant of the island all-reduce (FP32 partials).

    The owned quarter is split into ``S`` sub-chunks of ``c`` packs. Iteration
    ``t`` (one per-CTA barrier at its end) runs, for the same CTA-local pack
    offsets: A(t) island reduce-scatter push of sub-chunk ``t``; B(t-1) FP32
    island partial of sub-chunk ``t-1`` (order local 0..3, parked in the own
    output as in the serial variant) and its push to the partner; C(t-2)
    final = own + partner (commutative two-term FP32 sum, one BF16 rounding)
    into the owned output quarter and its island all-gather push; D(t-3) copy
    of the received quarters. The island-local legs of one sub-chunk thus
    overlap the cross-island leg of the previous one. Each staging area (RS
    regions, partner partial, AG regions) is written in one iteration and read
    in the next, so two parities (sub-chunk & 1) of ``10 * c`` packs suffice.
    Results are bitwise identical to the serial ``island`` transport.
    """

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
            staging0, staging1, staging2, staging3, staging4, staging5, staging6, staging7,
            staging8, staging9, staging10, staging11, staging12, staging13, staging14, staging15,
        )
        signals = (
            signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7,
            signal8, signal9, signal10, signal11, signal12, signal13, signal14, signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        stages = _ISLAND_OVERLAP_STAGES
        island_base = self._rank & 4
        local = self._rank & 3
        partner = self._rank ^ 4
        next_local = (local + 1) & 3
        # Sub-chunk of the quarter and this CTA's slice of every sub-chunk.
        sub = (shard_packs + Int64(stages - 1)) // Int64(stages)
        chunk = (sub + Int64(gdim) - Int64(1)) // Int64(gdim)
        cta_begin = Int64(bidx) * chunk
        cta_end = cta_begin + chunk
        if cta_end > sub:
            cta_end = sub
        unit = Int64(16)
        area = Int64(_ISLAND_OVERLAP_AREA_UNITS) * sub * unit

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
        own_low = output_address + Int64(local) * shard_packs * unit
        own_high = output_address + Int64(next_local) * shard_packs * unit

        for step in cutlass.range_constexpr(stages + 3):
            # A(step): island reduce-scatter push of sub-chunk `step`.
            if cutlass.const_expr(step < stages):
                parity = step & 1
                first = Int64(step) * sub
                stop = cta_end
                if first + stop > shard_packs:
                    stop = shard_packs - first
                for hop in cutlass.range_constexpr(1, 4):
                    peer_local = (local + hop) & 3
                    destination = (
                        Int64(staging[island_base + peer_local].toint())
                        + staging_slot_offset
                        + Int64(parity) * area
                        + Int64(local) * sub * unit
                    )
                    source = payload_address + (
                        Int64(peer_local) * shard_packs + first
                    ) * unit
                    index = cta_begin + Int64(tidx)
                    while index < stop:
                        words = ld_global_nc_v4_u32(source + index * unit)
                        _st_generic_v4_u32(
                            destination + index * unit,
                            words[0], words[1], words[2], words[3],
                        )
                        index += Int64(self._threads)
            # B(step-1): FP32 island partial, park in own output, push to partner.
            if cutlass.const_expr(step >= 1 and step - 1 < stages):
                chunk_id = step - 1
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                stop = cta_end
                if first + stop > shard_packs:
                    stop = shard_packs - first
                region = self_base + Int64(parity) * area
                remote = partner_base + Int64(parity) * area + Int64(4) * sub * unit
                index = cta_begin + Int64(tidx)
                while index < stop:
                    accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                    for source_local in cutlass.range_constexpr(4):
                        if cutlass.const_expr(source_local == local):
                            words = ld_global_nc_v4_u32(
                                payload_address
                                + (Int64(local) * shard_packs + first + index) * unit
                            )
                        else:
                            words = _ld_generic_v4_u32(
                                region + (Int64(source_local) * sub + index) * unit
                            )
                        self._accumulate_words(accumulator, words, source_local == 0)
                    low0 = f32_as_u32(accumulator[0])
                    low1 = f32_as_u32(accumulator[1])
                    low2 = f32_as_u32(accumulator[2])
                    low3 = f32_as_u32(accumulator[3])
                    high0 = f32_as_u32(accumulator[4])
                    high1 = f32_as_u32(accumulator[5])
                    high2 = f32_as_u32(accumulator[6])
                    high3 = f32_as_u32(accumulator[7])
                    st_global_v4_u32(own_low + (first + index) * unit, low0, low1, low2, low3)
                    st_global_v4_u32(own_high + (first + index) * unit, high0, high1, high2, high3)
                    _st_generic_v4_u32(remote + index * unit, low0, low1, low2, low3)
                    _st_generic_v4_u32(
                        remote + (sub + index) * unit, high0, high1, high2, high3
                    )
                    index += Int64(self._threads)
            # C(step-2): final sum, one rounding, island all-gather push.
            if cutlass.const_expr(step >= 2 and step - 2 < stages):
                chunk_id = step - 2
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                stop = cta_end
                if first + stop > shard_packs:
                    stop = shard_packs - first
                received = self_base + Int64(parity) * area + Int64(4) * sub * unit
                index = cta_begin + Int64(tidx)
                while index < stop:
                    total = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                    mine_low = ld_global_v4_u32(own_low + (first + index) * unit)
                    mine_high = ld_global_v4_u32(own_high + (first + index) * unit)
                    theirs_low = _ld_generic_v4_u32(received + index * unit)
                    theirs_high = _ld_generic_v4_u32(received + (sub + index) * unit)
                    for lane in cutlass.range_constexpr(4):
                        total[lane] = u32_as_f32(mine_low[lane]) + u32_as_f32(
                            theirs_low[lane]
                        )
                        total[lane + 4] = u32_as_f32(mine_high[lane]) + u32_as_f32(
                            theirs_high[lane]
                        )
                    word0 = pack_f32x2_to_bf16x2(total[0], total[1])
                    word1 = pack_f32x2_to_bf16x2(total[2], total[3])
                    word2 = pack_f32x2_to_bf16x2(total[4], total[5])
                    word3 = pack_f32x2_to_bf16x2(total[6], total[7])
                    st_global_v4_u32(own_low + (first + index) * unit, word0, word1, word2, word3)
                    for hop in cutlass.range_constexpr(1, 4):
                        peer_local = (local + hop) & 3
                        destination = (
                            Int64(staging[island_base + peer_local].toint())
                            + staging_slot_offset
                            + Int64(parity) * area
                            + (Int64(6 + local) * sub + index) * unit
                        )
                        _st_generic_v4_u32(destination, word0, word1, word2, word3)
                    index += Int64(self._threads)
            # D(step-3): copy the received quarters of sub-chunk step-3.
            if cutlass.const_expr(step >= 3):
                chunk_id = step - 3
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                stop = cta_end
                if first + stop > shard_packs:
                    stop = shard_packs - first
                for hop in cutlass.range_constexpr(1, 4):
                    source_local = (local + hop) & 3
                    source = (
                        self_base
                        + Int64(parity) * area
                        + Int64(6 + source_local) * sub * unit
                    )
                    destination = output_address + (
                        Int64(source_local) * shard_packs + first
                    ) * unit
                    index = cta_begin + Int64(tidx)
                    while index < stop:
                        words = _ld_generic_v4_u32(source + index * unit)
                        st_global_v4_u32(
                            destination + index * unit,
                            words[0], words[1], words[2], words[3],
                        )
                        index += Int64(self._threads)
            if cutlass.const_expr(step < stages + 2):
                self._barrier(signals, rank)


def island_overlap_fits(shard_packs: int, pack_stride: int) -> bool:
    """Two parities of 10 * ceil(quarter / stages) packs within the slot."""
    sub = -(-int(shard_packs) // _ISLAND_OVERLAP_STAGES)
    return 2 * _ISLAND_OVERLAP_AREA_UNITS * sub <= 8 * int(pack_stride)


# Owner-eighth pipeline: staging units (16-byte packs) per sub-chunk of c packs, h = c / 2.
_EIGHTH_RS = 0  # 2 parities x 4 regions x c   (island reduce-scatter in)
_EIGHTH_PIN = 8  # 2 parities x c               (partner FP32 partial of my eighth)
_EIGHTH_OWN = 10  # 2 parities x c              (own FP32 partial of my eighth, local)
_EIGHTH_FIN = 12  # 2 parities x h              (partner's final BF16 eighth)
_EIGHTH_AG = 13  # 3 slots x 4 regions x c     (island all-gather in)
_EIGHTH_AREA_UNITS = 25  # x c


def island_eighth_sub_packs(shard_packs: int) -> int:
    """Pipeline sub-chunk (packs of the owned quarter), rounded up to even."""
    sub = -(-int(shard_packs) // _ISLAND_OVERLAP_STAGES)
    return sub + (sub & 1)


def island_eighth_fits(shard_packs: int, pack_stride: int) -> bool:
    return _EIGHTH_AREA_UNITS * island_eighth_sub_packs(shard_packs) <= 8 * int(pack_stride)


class _TwoShotIslandEighthAllReduceLaunch(_TwoShotIslandAllReduceLaunch):
    """Owner-eighth, software-pipelined island all-reduce (FP32 partials, W8).

    Like ``island_overlap`` the owned quarter is cut into ``S`` sub-chunks of
    ``c`` (even) packs, but each sub-chunk's quarter is split between the two
    partners: island A (ranks 0-3) owns the first half ``[0, h)`` and island
    B the second half ``[h, c)``. Each partner therefore sends only the FP32
    partial of the half the other owns, the owner finalises it (own + partner,
    commutative two-term FP32 sum, one BF16 rounding) and returns the BF16
    half. Uplink bytes per direction drop from 2S to 1.5S; per-GPU egress is
    1.875S (bandwidth optimum 1.75S). Iteration ``t`` (one per-CTA barrier):

    A(t)   island reduce-scatter push of sub-chunk t;
    B(t-1) FP32 island partial (order 0..3); own half -> local OWN area,
           partner half -> partner's PIN area (cross-island);
    C(t-2) final of the own half -> output, partner's FIN area (cross) and
           the 3 island peers' AG area;
    D(t-3) partner's final half (FIN) -> output and the 3 island peers;
    E(t-4) copy the island peers' quarters (both halves) from AG.

    CTAs walk the half index ``j`` and always handle packs ``j`` and ``j+h``
    together, so every staging/output location of a pack is touched by the
    same CTA index on every rank (the per-CTA barrier contract). Area
    lifetimes: RS/PIN/OWN/FIN are written in one iteration and read in the
    next (2 parities); AG is written in two consecutive iterations and read
    in the third (3 slots). Every final element is rounded once by exactly one
    owner and copied bitwise elsewhere, so all ranks are identical and the
    results equal ``island`` / ``island_overlap``.
    """

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
            staging0, staging1, staging2, staging3, staging4, staging5, staging6, staging7,
            staging8, staging9, staging10, staging11, staging12, staging13, staging14, staging15,
        )
        signals = (
            signal0, signal1, signal2, signal3, signal4, signal5, signal6, signal7,
            signal8, signal9, signal10, signal11, signal12, signal13, signal14, signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        stages = _ISLAND_OVERLAP_STAGES
        island_base = self._rank & 4
        local = self._rank & 3
        partner = self._rank ^ 4
        island_a = self._rank < 4
        unit = Int64(16)
        # Sub-chunk c (even) and half h, both in packs of the owned quarter.
        sub = (shard_packs + Int64(stages - 1)) // Int64(stages)
        sub = sub + (sub & Int64(1))
        half = sub // Int64(2)
        chunk = (half + Int64(gdim) - Int64(1)) // Int64(gdim)
        cta_begin = Int64(bidx) * chunk
        cta_end = cta_begin + chunk
        if cta_end > half:
            cta_end = half
        sub_bytes = sub * unit

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
        own_quarter = output_address + Int64(local) * shard_packs * unit
        # Half offsets of this rank's own and its partner's half of a sub-chunk.
        mine = Int64(0) if island_a else half
        theirs = half if island_a else Int64(0)

        for step in cutlass.range_constexpr(stages + 4):
            # A(step): island reduce-scatter push of sub-chunk `step` (both halves).
            if cutlass.const_expr(step < stages):
                parity = step & 1
                first = Int64(step) * sub
                for hop in cutlass.range_constexpr(1, 4):
                    peer_local = (local + hop) & 3
                    destination = (
                        Int64(staging[island_base + peer_local].toint())
                        + staging_slot_offset
                        + Int64(_EIGHTH_RS + parity * 4 + local) * sub_bytes
                    )
                    source = payload_address + (Int64(peer_local) * shard_packs + first) * unit
                    j = cta_begin + Int64(tidx)
                    while j < cta_end:
                        for side in cutlass.range_constexpr(2):
                            position = j + (half if side == 1 else Int64(0))
                            if first + position < shard_packs:
                                words = ld_global_nc_v4_u32(source + position * unit)
                                _st_generic_v4_u32(
                                    destination + position * unit,
                                    words[0], words[1], words[2], words[3],
                                )
                        j += Int64(self._threads)
            # B(step-1): FP32 island partial; own half -> OWN, partner half -> partner PIN.
            if cutlass.const_expr(step >= 1 and step - 1 < stages):
                chunk_id = step - 1
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                regions = self_base + Int64(_EIGHTH_RS + parity * 4) * sub_bytes
                own_area = self_base + Int64(_EIGHTH_OWN + parity) * sub_bytes
                remote_area = partner_base + Int64(_EIGHTH_PIN + parity) * sub_bytes
                j = cta_begin + Int64(tidx)
                while j < cta_end:
                    for side in cutlass.range_constexpr(2):
                        offset = mine if side == 0 else theirs
                        position = j + offset
                        if first + position < shard_packs:
                            accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                            for source_local in cutlass.range_constexpr(4):
                                if cutlass.const_expr(source_local == local):
                                    words = ld_global_nc_v4_u32(
                                        payload_address
                                        + (Int64(local) * shard_packs + first + position) * unit
                                    )
                                else:
                                    words = _ld_generic_v4_u32(
                                        regions + Int64(source_local) * sub_bytes + position * unit
                                    )
                                self._accumulate_words(accumulator, words, source_local == 0)
                            target = own_area if side == 0 else remote_area
                            _st_generic_v4_u32(
                                target + j * unit,
                                f32_as_u32(accumulator[0]),
                                f32_as_u32(accumulator[1]),
                                f32_as_u32(accumulator[2]),
                                f32_as_u32(accumulator[3]),
                            )
                            _st_generic_v4_u32(
                                target + (half + j) * unit,
                                f32_as_u32(accumulator[4]),
                                f32_as_u32(accumulator[5]),
                                f32_as_u32(accumulator[6]),
                                f32_as_u32(accumulator[7]),
                            )
                    j += Int64(self._threads)
            # C(step-2): final of the own half; output, partner FIN, island AG.
            if cutlass.const_expr(step >= 2 and step - 2 < stages):
                chunk_id = step - 2
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                own_area = self_base + Int64(_EIGHTH_OWN + parity) * sub_bytes
                received = self_base + Int64(_EIGHTH_PIN + parity) * sub_bytes
                remote_final = partner_base + Int64(_EIGHTH_FIN) * sub_bytes + Int64(parity) * half * unit
                j = cta_begin + Int64(tidx)
                while j < cta_end:
                    position = j + mine
                    if first + position < shard_packs:
                        total = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
                        own_low = ld_global_v4_u32(own_area + j * unit)
                        own_high = ld_global_v4_u32(own_area + (half + j) * unit)
                        peer_low = _ld_generic_v4_u32(received + j * unit)
                        peer_high = _ld_generic_v4_u32(received + (half + j) * unit)
                        for lane in cutlass.range_constexpr(4):
                            total[lane] = u32_as_f32(own_low[lane]) + u32_as_f32(peer_low[lane])
                            total[lane + 4] = u32_as_f32(own_high[lane]) + u32_as_f32(
                                peer_high[lane]
                            )
                        word0 = pack_f32x2_to_bf16x2(total[0], total[1])
                        word1 = pack_f32x2_to_bf16x2(total[2], total[3])
                        word2 = pack_f32x2_to_bf16x2(total[4], total[5])
                        word3 = pack_f32x2_to_bf16x2(total[6], total[7])
                        st_global_v4_u32(
                            own_quarter + (first + position) * unit, word0, word1, word2, word3
                        )
                        _st_generic_v4_u32(remote_final + j * unit, word0, word1, word2, word3)
                        for hop in cutlass.range_constexpr(1, 4):
                            peer_local = (local + hop) & 3
                            destination = (
                                Int64(staging[island_base + peer_local].toint())
                                + staging_slot_offset
                                + Int64(_EIGHTH_AG + (chunk_id % 3) * 4 + local) * sub_bytes
                                + position * unit
                            )
                            _st_generic_v4_u32(destination, word0, word1, word2, word3)
                    j += Int64(self._threads)
            # D(step-3): partner's final half -> output and island AG.
            if cutlass.const_expr(step >= 3 and step - 3 < stages):
                chunk_id = step - 3
                parity = chunk_id & 1
                first = Int64(chunk_id) * sub
                final_in = self_base + Int64(_EIGHTH_FIN) * sub_bytes + Int64(parity) * half * unit
                j = cta_begin + Int64(tidx)
                while j < cta_end:
                    position = j + theirs
                    if first + position < shard_packs:
                        words = _ld_generic_v4_u32(final_in + j * unit)
                        st_global_v4_u32(
                            own_quarter + (first + position) * unit,
                            words[0], words[1], words[2], words[3],
                        )
                        for hop in cutlass.range_constexpr(1, 4):
                            peer_local = (local + hop) & 3
                            destination = (
                                Int64(staging[island_base + peer_local].toint())
                                + staging_slot_offset
                                + Int64(_EIGHTH_AG + (chunk_id % 3) * 4 + local) * sub_bytes
                                + position * unit
                            )
                            _st_generic_v4_u32(
                                destination, words[0], words[1], words[2], words[3]
                            )
                    j += Int64(self._threads)
            # E(step-4): copy the island peers' quarters of sub-chunk step-4.
            if cutlass.const_expr(step >= 4):
                chunk_id = step - 4
                first = Int64(chunk_id) * sub
                for hop in cutlass.range_constexpr(1, 4):
                    source_local = (local + hop) & 3
                    source = self_base + Int64(_EIGHTH_AG + (chunk_id % 3) * 4 + source_local) * sub_bytes
                    destination = output_address + (Int64(source_local) * shard_packs + first) * unit
                    j = cta_begin + Int64(tidx)
                    while j < cta_end:
                        for side in cutlass.range_constexpr(2):
                            position = j + (half if side == 1 else Int64(0))
                            if first + position < shard_packs:
                                words = _ld_generic_v4_u32(source + position * unit)
                                st_global_v4_u32(
                                    destination + position * unit,
                                    words[0], words[1], words[2], words[3],
                                )
                        j += Int64(self._threads)
            if cutlass.const_expr(step < stages + 3):
                self._barrier(signals, rank)


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
    overlap: bool = False,
    eighth: bool = False,
) -> Callable[..., None]:
    """Compile the single-launch 2-island hierarchical all-reduce (world 8)."""
    if (overlap or eighth) and wire_bf16:
        raise ValueError("the overlapped island all-reduce carries FP32 partials only")
    if overlap and eighth:
        raise ValueError("choose one island pipeline variant")
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
    launch = (
        _TwoShotIslandEighthAllReduceLaunch
        if eighth
        else _TwoShotIslandOverlapAllReduceLaunch
        if overlap
        else _TwoShotIslandAllReduceLaunch
    )(
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        wire_bf16,
    )
    name = (
        "all_reduce_island_eighth"
        if eighth
        else "all_reduce_island_overlap"
        if overlap
        else ("all_reduce_island_bf16" if wire_bf16 else "all_reduce_island")
    )
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
