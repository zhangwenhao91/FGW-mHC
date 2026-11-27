"""Ascend 950 forward kernels for the TileOPs mHC contract."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

_SIMD_WEIGHT_STRIDE = 64


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def project(x, phi, partial, *, batch: int, width: int, chunk: int):
    splits = width // chunk
    x: T.Tensor[(batch, width), T.bfloat16]
    phi: T.Tensor[(width, 24), T.float32]
    partial: T.Tensor[(batch, splits, 32), T.float32]

    cores = min(batch * splits, 56)
    with T.Kernel(cores) as core:
        x_ub = T.alloc_shared((chunk,), T.bfloat16)
        phi_ub = T.alloc_shared((chunk, 24), T.float32)
        sums = T.alloc_shared((32,), T.float32)
        for token, part in T.Persistent([batch, splits], cores, core, group_size=1):
            start = part * chunk
            T.copy(x[token, start], x_ub)
            T.copy(phi[start, 0], phi_ub)
            with T.SimtVF(threads=32):
                for col in T.Parallel(32):
                    acc = T.alloc_var(T.float32, init=0.0)
                    if col < 24:
                        for k in T.serial(chunk):
                            acc += T.Cast(T.float32, x_ub[k]) * phi_ub[k, col]
                    elif col == 24:
                        for k in T.serial(chunk):
                            value = T.Cast(T.float32, x_ub[k])
                            acc += value * value
                    sums[col] = acc
            T.copy(sums, partial[token, part, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def prepare_cube(x, promoted, squares, *, batch: int, width: int, chunk: int):
    """Promote BF16 activations and publish split-K RMS statistics."""
    splits = width // chunk
    x: T.Tensor[(batch, width), T.bfloat16]
    promoted: T.Tensor[(batch, width), T.float32]
    squares: T.Tensor[(batch, splits, 8), T.float32]

    cores = min(batch * splits, 56)
    with T.Kernel(cores) as core:
        x_ub = T.alloc_shared((chunk,), T.bfloat16)
        promoted_ub = T.alloc_shared((chunk,), T.float32)
        squares_ub = T.alloc_shared((8,), T.float32)
        for token, part in T.Persistent([batch, splits], cores, core, group_size=1):
            start = part * chunk
            T.copy(x[token, start], x_ub)
            with T.SimtVF(threads=32):
                for k in T.Parallel(chunk):
                    promoted_ub[k] = T.Cast(T.float32, x_ub[k])
                for lane in T.Parallel(8):
                    total = T.alloc_var(T.float32, init=0.0)
                    for k in T.serial(chunk // 8):
                        value = T.Cast(T.float32, x_ub[k * 8 + lane])
                        total += value * value
                    squares_ub[lane] = total
            T.copy(promoted_ub, promoted[token, start])
            T.copy(squares_ub, squares[token, part, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def project_cube(x, phi, partial, *, batch: int, width: int, chunk: int):
    """Split-K HF32 projection with FP32 accumulation on Cube."""
    splits = width // chunk
    rows = 16
    cols = 32
    row_tiles = (batch + rows - 1) // rows
    x: T.Tensor[(batch, width), T.float32]
    phi: T.Tensor[(width, 24), T.float32]
    partial: T.Tensor[(splits, row_tiles * rows, cols), T.float32]

    cores = min(row_tiles * splits, 28)
    with T.Kernel(cores) as core:
        a_l1 = T.alloc_l1((rows, chunk), T.float32)
        b_l1 = T.alloc_l1((chunk, cols), T.float32)
        a_l0 = T.alloc_l0a((rows, chunk), T.float32)
        b_l0 = T.alloc_l0b((cols, chunk), T.float32)
        acc = T.alloc_l0c((rows, cols), T.float32)
        T.set_hf32_mode("nearest_even")
        for part, row_tile in T.Persistent([splits, row_tiles], cores, core, group_size=1):
            k = part * chunk
            m = row_tile * rows
            # Bounds-aware loads zero-pad partial M tiles and the last 8 phi columns.
            T.copy(x[m:m + rows, k:k + chunk], a_l1)
            T.copy(phi[k:k + chunk, 0:cols], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0, transpose=True)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, partial[part, m:m + rows, 0:cols])


@T.macro
def _apply_weights(x_ub, weights_ub, res_ub, layer_ub, tile, weight_stride):
    with T.SimdVF():
        for chunk in T.serial(tile // 64):
            col = chunk * 64
            streams = S.alloc_local((4,), T.float32)
            for i in T.unroll(4, explicit=True):
                streams[i] = S.vcvt(S.vld(x_ub[i, col], dist="UNPK_B16"), T.float32, part=0)
            acc = S.alloc_local((1,), T.float32)
            acc[0] = S.vmul(streams[0], S.vld(weights_ub[0], dist="BRC_B32"))
            for i in T.unroll(1, 4, explicit=True):
                S.vmula(acc[0], streams[i], S.vld(weights_ub[i], dist="BRC_B32"))
            S.vsts(layer_ub[col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")
            for row in T.unroll(4, explicit=True):
                acc[0] = S.vmul(streams[0], S.vld(weights_ub[(row + 1) * weight_stride], dist="BRC_B32"))
                for i in T.unroll(1, 4, explicit=True):
                    S.vmula(acc[0], streams[i], S.vld(weights_ub[(row + 1) * weight_stride + i], dist="BRC_B32"))
                S.vsts(res_ub[row, col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")


@T.macro
def _normalize_simd(logits_ub, weights_ub, post_ub, repeat, eps):
    with T.SimdVF():
        four = S.pset(32, "PAT_VL4")
        one = S.vdup(T.float32(1), T.float32, four)
        two = S.vdup(T.float32(2), T.float32, four)
        epsilon = S.vdup(T.float32(eps), T.float32, four)
        pre = S.vld(logits_ub[0, 0])
        post = S.vld(logits_ub[1, 0])
        S.vsts(weights_ub[0], S.vdiv(one, S.vadd(one, S.vexp(S.vneg(pre, four), four), four), four), four)
        S.vsts(post_ub[0], S.vdiv(two, S.vadd(one, S.vexp(S.vneg(post, four), four), four), four), four)

        rows = S.alloc_local((4,), T.float32)
        for row in T.unroll(4, explicit=True):
            logits = S.vld(logits_ub[2 + row, 0])
            row_max = S.vdupv(S.vcmax(logits, four), four)
            rows[row] = S.vexp(S.vsub(logits, row_max, four), four)
        for iteration in T.serial(repeat):
            for row in T.unroll(4, explicit=True):
                row_sum = S.vdupv(S.vcadd(rows[row], four), four)
                rows[row] = S.vdiv(rows[row], S.vadd(row_sum, epsilon, four), four)
            cols = S.vadd(S.vadd(S.vadd(rows[0], rows[1], four), rows[2], four), rows[3], four)
            denom = S.vadd(cols, epsilon, four)
            for row in T.unroll(4, explicit=True):
                rows[row] = S.vdiv(rows[row], denom, four)
        for row in T.unroll(4, explicit=True):
            S.vsts(weights_ub[(row + 1) * _SIMD_WEIGHT_STRIDE], rows[row], four)


@T.macro
def _compute_weights(
    partial_ub, squares_ub, bias_ub, totals, weights_ub, post_ub, logits_ub,
    width, splits, repeat, eps, alpha_pre, alpha_post, alpha_res, cube, fused, simd_weights,
):
    with T.SimtVF(threads=32):
        for q in T.Parallel(32):
            value = T.alloc_var(T.float32, init=0.0)
            if cube and q == 24:
                for split in T.serial(splits):
                    if fused:
                        value += squares_ub[split, 0]
                    else:
                        for lane in T.unroll(8):
                            value += squares_ub[split, lane]
            else:
                for split in T.serial(splits):
                    value += partial_ub[split, q]
            totals[q] = value
    if simd_weights:
        with T.SimtVF(threads=32):
            for q in T.Parallel(24):
                norm = T.sqrt(totals[24] / width) + 0.0001
                scale = T.if_then_else(q < 4, alpha_pre, T.if_then_else(q < 8, alpha_post, alpha_res))
                logits_ub[q // 4, q % 4] = scale * totals[q] / norm + bias_ub[q]
        _normalize_simd(logits_ub, weights_ub, post_ub, repeat, eps)
    else:
        with T.SimtVF(threads=32):
            for lane in T.Parallel(1):
                # Sinkhorn is serial within this lane. Publish only final weights to UB.
                residual = T.alloc_local((16,), T.float32)
                norm = T.sqrt(totals[24] / width) + 0.0001
                for i in T.serial(4):
                    pre = alpha_pre * totals[i] / norm + bias_ub[i]
                    post = alpha_post * totals[4 + i] / norm + bias_ub[4 + i]
                    weights_ub[i] = 1.0 / (1.0 + T.exp(-pre))
                    post_ub[i] = 2.0 / (1.0 + T.exp(-post))
                for i in T.serial(4):
                    row = i * 4
                    for j in T.serial(4):
                        q = 8 + i * 4 + j
                        residual[row + j] = alpha_res * totals[q] / norm + bias_ub[q]
                    row_max = T.alloc_var(T.float32, init=residual[row])
                    for j in T.serial(1, 4):
                        row_max = T.max(row_max, residual[row + j])
                    for j in T.serial(4):
                        residual[row + j] = T.exp(residual[row + j] - row_max)
                for iteration in T.serial(repeat):
                    for i in T.serial(4):
                        row_sum = T.alloc_var(T.float32, init=0.0)
                        for j in T.serial(4):
                            row_sum += residual[i * 4 + j]
                        for j in T.serial(4):
                            residual[i * 4 + j] /= row_sum + eps
                    for j in T.serial(4):
                        col_sum = T.alloc_var(T.float32, init=0.0)
                        for i in T.serial(4):
                            col_sum += residual[i * 4 + j]
                        for i in T.serial(4):
                            residual[i * 4 + j] /= col_sum + eps
                for i in T.serial(16):
                    weights_ub[4 + i] = residual[i]
                for i in T.serial(20, 24):
                    weights_ub[i] = 0.0


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def project_cube_fused(
    x, phi, partial, squares, partial_read, squares_read, bias, weights, h_post, x_res, x_layer, *,
    batch: int, width: int, chunk: int, cores: int, merge: bool,
    tile: int = 64, full_pre: bool = False, simd_weights: bool = False, skip_padding: bool = False,
    repeat: int = 0, eps: float = 0.02,
    alpha_pre: float = 1.0, alpha_post: float = 1.0, alpha_res: float = 1.0,
):
    """Prepare X on paired Vector cores and feed Cube without a GM conversion buffer."""
    splits = width // chunk
    rows = 16
    half = rows // 2
    cols = 32
    # Split larger L1 windows to keep each L0 operand within the existing footprint.
    mad_k = chunk // 2 if chunk > 256 else chunk
    row_tiles = (batch + rows - 1) // rows
    jobs = row_tiles * splits
    x: T.Tensor[(batch, width), T.bfloat16]
    phi: T.Tensor[(width, 24), T.float32]
    partial: T.Tensor[(splits, row_tiles * rows, cols), T.float32]
    squares: T.Tensor[(splits, row_tiles * rows, 8), T.float32]

    partial_read: T.Tensor[(splits, row_tiles * rows, cols), T.float32]
    squares_read: T.Tensor[(splits, row_tiles * rows, 8), T.float32]
    bias: T.Tensor[(24,), T.float32]
    weights: T.Tensor[(batch, 24), T.float32]
    h_post: T.Tensor[(batch, 4), T.float32]

    hidden = width // 4
    x_res: T.Tensor[(batch, 4, hidden), T.bfloat16]
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]

    with T.MixedKernel(cores) as (core, side):
        x_ub = T.alloc_shared((half, chunk), T.bfloat16)
        nz_ub = T.alloc_shared((half + 1, chunk), T.float32)
        T.annotate_layout({nz_ub: make_ascend_compact_nz_layout(nz_ub)})
        squares_ub = T.alloc_shared((half, 8), T.float32)
        a_l1 = T.alloc_l1((rows, chunk), T.float32)
        b_l1 = T.alloc_l1((chunk, cols), T.float32)
        a_l0 = T.alloc_l0a((rows, mad_k), T.float32)
        b_l0 = T.alloc_l0b((cols, mad_k), T.float32)
        acc = T.alloc_l0c((rows, cols), T.float32)

        if merge:
            T.import_source("#define FGW_SYNC(...) ({ __VA_ARGS__; 0; })\n")
            raw_ub = T.alloc_shared((half, cols), T.float32)
            partial_ub = T.alloc_shared((splits, cols), T.float32)
            split_squares_ub = T.alloc_shared((splits, 8), T.float32)
            bias_ub = T.alloc_shared((24,), T.float32)
            totals = T.alloc_shared((32,), T.float32)
            weights_ub = T.alloc_shared((5 * _SIMD_WEIGHT_STRIDE if simd_weights else 24,), T.float32)
            # SIMD reads and writes start on whole register boundaries.
            logits_ub = T.alloc_shared((6, 64), T.float32) if simd_weights else totals
            post_ub = T.alloc_shared((8,), T.float32)

        if full_pre:
            streams_ub = T.alloc_shared((4, tile), T.bfloat16)
            res_ub = T.alloc_shared((4, tile), T.bfloat16)
            layer_ub = T.alloc_shared((tile,), T.bfloat16)
            streams = T.reshape(x, (batch, 4, hidden))

        T.set_hf32_mode("nearest_even")
        for step in T.Pipelined(jobs // cores):
            job = step * cores + core
            part = job // row_tiles
            m = job % row_tiles * rows
            k = part * chunk
            vector_m = m + side * half
            # The tail load zero-pads any rows outside the real batch.
            T.copy(x[vector_m:vector_m + half, k:k + chunk], x_ub)
            T.copy(phi[k:k + chunk, 0:cols], b_l1)
            with T.SimdVF():
                full = S.pset(32)
                first = S.pset(32, "PAT_VL1")
                eight = S.pset(32, "PAT_VL8")
                zero = S.vdup(T.float32(0), T.float32, full)
                nz_stride = T.int32((half + 1) << 16)
                valid_rows = T.min(T.max(batch - vector_m, 0), half) if skip_padding else half
                for row in T.serial(valid_rows):
                    squared = S.alloc_local((1,), T.float32)
                    squared[0] = zero
                    for col_tile in T.unroll(chunk // 64, explicit=True):
                        col = col_tile * 64
                        values = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                        ptr = S.make_ubuf_ptr(T.access_ptr(nz_ub[row, col], "w", 1, 64), "float32")
                        S.vsstb(values, ptr, nz_stride, full)
                        S.vmula(squared[0], values, values, full)
                    total = S.vcadd(squared[0], full)
                    S.vsts(squares_ub[row, 0], S.vsel(total, zero, first), eight)
                if skip_padding:
                    for row_index in T.serial(half - valid_rows):
                        row = valid_rows + row_index
                        for col_tile in T.unroll(chunk // 64, explicit=True):
                            col = col_tile * 64
                            ptr = S.make_ubuf_ptr(T.access_ptr(nz_ub[row, col], "w", 1, 64), "float32")
                            S.vsstb(zero, ptr, nz_stride, full)
                        S.vsts(squares_ub[row, 0], zero, eight)
            # Both Vector halves publish their rows to the same Cube L1 tile.
            T.dual_copy(nz_ub[0:half, 0:chunk], a_l1)
            if not merge or step < jobs // cores - 1:
                T.copy(squares_ub, squares[part, vector_m:vector_m + half, 0:8])
            if chunk == mad_k:
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0, transpose=True)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            else:
                for mad in T.serial(chunk // mad_k):
                    offset = mad * mad_k
                    T.copy(a_l1[:, offset:offset + mad_k], a_l0)
                    T.copy(b_l1[offset:offset + mad_k, :], b_l0, transpose=True)
                    T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=mad == 0)
            if merge:
                # Move the Cube result to Vector so both workspaces are published by MTE3.
                T.dual_copy(acc, raw_ub)
                if step < jobs // cores - 1:
                    T.copy(raw_ub, partial[part, vector_m:vector_m + half, 0:cols])
            else:
                T.copy(acc, partial[part, m:m + rows, 0:cols])

        if merge:
            # Every Vector half participates, including halves with no coefficient work.
            last_job = (jobs // cores - 1) * cores + core
            last_part = last_job // row_tiles
            last_row = last_job % row_tiles * rows + side * half
            with T.PerCoreTask():
                # Anchor publication to the final workspace stores.
                with T.Task():
                    T.copy(raw_ub, partial[last_part, last_row:last_row + half, 0:cols])
                    T.copy(squares_ub, squares[last_part, last_row:last_row + half, 0:8])
                T.ascend_sync_inter_arrive("PIPE_MTE3", 1)
                T.ascend_sync_inter_wait("PIPE_MTE3", 1)
                _drain = T.call_extern("int32", "FGW_SYNC", T.ascend_pipe_barrier("PIPE_MTE3"))
                _ready = T.call_extern("int32", "FGW_SYNC", T.ascend_set_flag("MTE3_MTE2", 7))
            first = core * 2 + side
            coefficient_jobs = T.ceildiv(T.max(batch - first, 0), cores * 2)
            with T.Stage(0):
                with T.Task():
                    T.ascend_wait_flag("MTE3_MTE2", 7)
                    T.copy(bias, bias_ub[0:24])
                for step in T.serial(coefficient_jobs):
                    token = first + step * cores * 2
                    T.copy(partial_read[0:splits, token, 0:cols], partial_ub)
                    T.copy(squares_read[0:splits, token, 0:8], split_squares_ub)
                    _compute_weights(
                        partial_ub, split_squares_ub, bias_ub, totals, weights_ub, post_ub, logits_ub,
                        width, splits, repeat, eps, alpha_pre, alpha_post, alpha_res, True, True, simd_weights,
                    )
                    if full_pre:
                        # Keep each token's weights on the Vector half that computed them.
                        for column_tile in T.serial(hidden // tile):
                            start = column_tile * tile
                            T.copy(streams[token, 0, start], streams_ub)
                            _apply_weights(streams_ub, weights_ub, res_ub, layer_ub, tile, _SIMD_WEIGHT_STRIDE if simd_weights else 4)
                            T.copy(res_ub, x_res[token, 0, start])
                            T.copy(layer_ub, x_layer[token, start])
                    else:
                        T.copy(weights_ub, weights[token, 0])
                    T.copy(post_ub[0:4], h_post[token, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def project_cube_m_only(
    x, phi, bias, h_post, x_res, x_layer, trace, *,
    batch: int, width: int, chunk: int, tile: int, repeat: int, eps: float,
    alpha_pre: float, alpha_post: float, alpha_res: float, debug: bool = False,
):
    """Give one Cube group each M tile and accumulate all K locally."""
    rows = 16
    half = rows // 2
    cols = 32
    splits = width // chunk
    row_tiles = (batch + rows - 1) // rows
    cores = min(row_tiles, 28)
    x: T.Tensor[(batch, width), T.bfloat16]
    phi: T.Tensor[(width, 24), T.float32]
    bias: T.Tensor[(24,), T.float32]
    h_post: T.Tensor[(batch, 4), T.float32]

    trace: T.Tensor[(batch, 32 if debug else 4), T.float32]
    hidden = width // 4
    x_res: T.Tensor[(batch, 4, hidden), T.bfloat16]
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]

    with T.MixedKernel(cores) as (core, side):
        x_ub = T.alloc_shared((half, chunk), T.bfloat16)
        nz_ub = T.alloc_shared((half + 1, chunk), T.float32)
        T.annotate_layout({nz_ub: make_ascend_compact_nz_layout(nz_ub)})
        squares_ub = T.alloc_shared((half, 8), T.float32)
        a_l1 = T.alloc_l1((rows, chunk), T.float32)
        b_l1 = T.alloc_l1((chunk, cols), T.float32)
        a_l0 = T.alloc_l0a((rows, chunk), T.float32)
        b_l0 = T.alloc_l0b((cols, chunk), T.float32)
        acc = T.alloc_l0c((rows, cols), T.float32)

        raw_ub = T.alloc_shared((half, cols), T.float32)
        partial_ub = T.alloc_shared((1, cols), T.float32)
        sum_ub = T.alloc_shared((1, 8), T.float32)
        bias_ub = T.alloc_shared((24,), T.float32)
        totals = T.alloc_shared((32,), T.float32)
        weights_ub = T.alloc_shared((24,), T.float32)
        post_ub = T.alloc_shared((8,), T.float32)
        streams_ub = T.alloc_shared((4, tile), T.bfloat16)
        res_ub = T.alloc_shared((4, tile), T.bfloat16)
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        streams = T.reshape(x, (batch, 4, hidden))

        T.set_hf32_mode("nearest_even")
        for row_tile in T.Persistent([row_tiles], cores, core):
            vector_m = row_tile * rows + side * half
            for part in T.serial(splits):
                k = part * chunk
                # The tail load zero-pads any rows outside the real batch.
                T.copy(x[vector_m:vector_m + half, k:k + chunk], x_ub)
                T.copy(phi[k:k + chunk, 0:cols], b_l1)
                with T.SimdVF():
                    full = S.pset(32)
                    first = S.pset(32, "PAT_VL1")
                    eight = S.pset(32, "PAT_VL8")
                    zero = S.vdup(T.float32(0), T.float32, full)
                    nz_stride = T.int32((half + 1) << 16)
                    valid_rows = T.min(T.max(batch - vector_m, 0), half)
                    for row in T.serial(valid_rows):
                        squared = S.alloc_local((1,), T.float32)
                        squared[0] = zero
                        for col_tile in T.unroll(chunk // 64, explicit=True):
                            col = col_tile * 64
                            values = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                            ptr = S.make_ubuf_ptr(T.access_ptr(nz_ub[row, col], "w", 1, 64), "float32")
                            S.vsstb(values, ptr, nz_stride, full)
                            S.vmula(squared[0], values, values, full)
                        total = S.vcadd(squared[0], full)
                        if part == 0:
                            S.vsts(squares_ub[row, 0], S.vsel(total, zero, first), eight)
                        else:
                            previous = S.vld(squares_ub[row, 0], dist="BRC_B32")
                            accumulated = S.vadd(total, previous, first)
                            S.vsts(squares_ub[row, 0], S.vsel(accumulated, zero, first), eight)
                    if valid_rows < half:
                        for row_index in T.serial(half - valid_rows):
                            row = valid_rows + row_index
                            for col_tile in T.unroll(chunk // 64, explicit=True):
                                col = col_tile * 64
                                ptr = S.make_ubuf_ptr(T.access_ptr(nz_ub[row, col], "w", 1, 64), "float32")
                                S.vsstb(zero, ptr, nz_stride, full)
                            S.vsts(squares_ub[row, 0], zero, eight)
                # Both Vector halves publish their rows to the same Cube L1 tile.
                T.dual_copy(nz_ub[0:half, 0:chunk], a_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0, transpose=True)
                # Keep L0C live through intermediate MADs and release it for the final Fixpipe copy.
                T.gemm(
                    a_l0, b_l0, acc, transpose_B=True, clear_accum=part == 0,
                    unit_flag_ctrl=T.Select(part == splits - 1, 3, 2),
                )
            T.dual_copy(acc, raw_ub, unit_flag_ctrl=3)
            T.copy(bias, bias_ub)
            for row in T.serial(T.min(T.max(batch - vector_m, 0), half)):
                token = vector_m + row
                T.copy(raw_ub[row:row + 1, :], partial_ub)
                T.copy(squares_ub[row:row + 1, :], sum_ub)
                if debug:
                    T.copy(partial_ub[0, 0:24], trace[token, 0:24])
                    T.copy(sum_ub[0, 0:8], trace[token, 24:32])
                _compute_weights(
                    partial_ub, sum_ub, bias_ub, totals, weights_ub, post_ub, totals,
                    width, 1, repeat, eps, alpha_pre, alpha_post, alpha_res, True, True, False,
                )
                for column_tile in T.serial(hidden // tile):
                    offset = column_tile * tile
                    T.copy(streams[token, 0, offset], streams_ub)
                    _apply_weights(streams_ub, weights_ub, res_ub, layer_ub, tile, 4)
                    T.copy(res_ub, x_res[token, 0, offset])
                    T.copy(layer_ub, x_layer[token, offset])
                T.copy(post_ub[0:4], h_post[token, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def mix(
    partial, squares, bias, weights, h_post, *,
    batch: int, width: int, splits: int, repeat: int, eps: float,
    alpha_pre: float, alpha_post: float, alpha_res: float, cube: bool, fused: bool,
):
    shape = (splits, (batch + 15) // 16 * 16, 32) if cube else (batch, splits, 32)
    if fused:
        squares_shape = (splits, shape[1], 8)
    elif cube:
        squares_shape = (batch, splits, 8)
    else:
        squares_shape = shape
    partial: T.Tensor[shape, T.float32]
    squares: T.Tensor[squares_shape, T.float32]
    bias: T.Tensor[(24,), T.float32]
    weights: T.Tensor[(batch, 24), T.float32]
    h_post: T.Tensor[(batch, 4), T.float32]

    cores = min(batch, 56)
    with T.Kernel(cores) as core:
        partial_ub = T.alloc_shared((splits, 32), T.float32)
        squares_ub = T.alloc_shared((splits, 8), T.float32) if cube else partial_ub
        bias_ub = T.alloc_shared((24,), T.float32)
        totals = T.alloc_shared((32,), T.float32)
        weights_ub = T.alloc_shared((24,), T.float32)
        post_ub = T.alloc_shared((8,), T.float32)
        for token in T.Persistent([batch], cores, core, group_size=1):
            if cube:
                T.copy(partial[0:splits, token, 0:32], partial_ub)
                if fused:
                    T.copy(squares[0:splits, token, 0:8], squares_ub)
                else:
                    T.copy(squares[token, 0, 0], squares_ub)
            else:
                T.copy(partial[token, 0, 0], partial_ub)
            T.copy(bias, bias_ub)
            _compute_weights(
                partial_ub, squares_ub, bias_ub, totals, weights_ub, post_ub, totals,
                width, splits, repeat, eps, alpha_pre, alpha_post, alpha_res, cube, fused, False,
            )
            T.copy(weights_ub, weights[token, 0])
            T.copy(post_ub[0:4], h_post[token, 0])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def apply_mix(x, weights, x_res, x_layer, *, batch: int, hidden: int, tile: int):
    tiles = hidden // tile
    x: T.Tensor[(batch, 4, hidden), T.bfloat16]
    weights: T.Tensor[(batch, 24), T.float32]
    x_res: T.Tensor[(batch, 4, hidden), T.bfloat16]
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]

    cores = min(batch * tiles, 56)
    with T.Kernel(cores) as core:
        x_ub = T.alloc_shared((4, tile), T.bfloat16)
        weights_ub = T.alloc_shared((24,), T.float32)
        res_ub = T.alloc_shared((4, tile), T.bfloat16)
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        for token, column_tile in T.Persistent([batch, tiles], cores, core, group_size=1):
            start = column_tile * tile
            T.copy(x[token, 0, start], x_ub)
            T.copy(weights[token, 0], weights_ub)
            _apply_weights(x_ub, weights_ub, res_ub, layer_ub, tile, 4)
            T.copy(res_ub, x_res[token, 0, start])
            T.copy(layer_ub, x_layer[token, start])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def post(x_layer, h_post, x_res, out, *, batch: int, hidden: int, tile: int, cores: int, stages: int, streams: int, total_streams: int):
    tiles = (hidden + tile - 1) // tile
    groups = total_streams // streams
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]
    h_post: T.Tensor[(batch, total_streams), T.float32]
    x_res: T.Tensor[(batch, total_streams, hidden), T.bfloat16]
    out: T.Tensor[(batch, total_streams, hidden), T.bfloat16]

    with T.Kernel(cores) as core:
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        post_ub = T.alloc_shared((8,), T.float32)
        res_ub = T.alloc_shared((streams, tile), T.bfloat16)
        out_ub = T.alloc_shared((streams, tile), T.bfloat16)
        if stages > 0:
            T.annotate_buffer_versions({
                layer_ub: stages, post_ub: stages, res_ub: stages, out_ub: stages,
            })
        for token_group, column_tile in T.Persistent(
            [batch * groups, tiles], cores, core, group_size=1, num_stages=stages,
        ):
            token = token_group // groups
            stream = token_group % groups * streams
            start = column_tile * tile
            count = T.min(tile, hidden - start)
            T.copy(x_layer[token, start:start + count], layer_ub[:count])
            T.copy(h_post[token, stream:stream + streams], post_ub[0:streams])
            T.copy(x_res[token, stream:stream + streams, start:start + count], res_ub[:, :count])
            with T.SimdVF():
                coeff = S.alloc_local((streams,), T.float32)
                for row in T.unroll(streams, explicit=True):
                    coeff[row] = S.vld(post_ub[row], dist="BRC_B32")
                for chunk in T.serial(tile // 64):
                    col = chunk * 64
                    layer = S.vcvt(S.vld(layer_ub[col], dist="UNPK_B16"), T.float32, part=0)
                    for row in T.unroll(streams, explicit=True):
                        residual = S.vcvt(S.vld(res_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                        result = S.vadd(residual, S.vmul(layer, coeff[row]))
                        S.vsts(out_ub[row, col], S.vcvt(result, T.bfloat16), dist="PK_B32")
            T.copy(out_ub[:, :count], out[token, stream:stream + streams, start:start + count], l2_cache_ctrl=0)
