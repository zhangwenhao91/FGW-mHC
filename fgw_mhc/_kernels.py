"""Ascend 950 forward kernels for the TileOPs mHC contract."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


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
def mix(
    partial, bias, weights, h_post, *,
    batch: int, width: int, splits: int, repeat: int, eps: float,
    alpha_pre: float, alpha_post: float, alpha_res: float,
):
    partial: T.Tensor[(batch, splits, 32), T.float32]
    bias: T.Tensor[(24,), T.float32]
    weights: T.Tensor[(batch, 24), T.float32]
    h_post: T.Tensor[(batch, 4), T.float32]

    cores = min(batch, 56)
    with T.Kernel(cores) as core:
        partial_ub = T.alloc_shared((splits, 32), T.float32)
        bias_ub = T.alloc_shared((24,), T.float32)
        totals = T.alloc_shared((32,), T.float32)
        weights_ub = T.alloc_shared((24,), T.float32)
        post_ub = T.alloc_shared((8,), T.float32)
        for token in T.Persistent([batch], cores, core, group_size=1):
            T.copy(partial[token, 0, 0], partial_ub)
            T.copy(bias, bias_ub)
            with T.SimtVF(threads=32):
                for q in T.Parallel(32):
                    value = T.alloc_var(T.float32, init=0.0)
                    for split in T.serial(splits):
                        value += partial_ub[split, q]
                    totals[q] = value
            with T.SimtVF(threads=32):
                for lane in T.Parallel(1):
                    norm = T.sqrt(totals[24] / width) + 0.0001
                    for i in T.serial(4):
                        pre = alpha_pre * totals[i] / norm + bias_ub[i]
                        post = alpha_post * totals[4 + i] / norm + bias_ub[4 + i]
                        weights_ub[i] = 1.0 / (1.0 + T.exp(-pre))
                        post_ub[i] = 2.0 / (1.0 + T.exp(-post))
                    for i in T.serial(4):
                        row = 4 + i * 4
                        for j in T.serial(4):
                            q = 8 + i * 4 + j
                            weights_ub[row + j] = alpha_res * totals[q] / norm + bias_ub[q]
                        row_max = T.alloc_var(T.float32, init=weights_ub[row])
                        for j in T.serial(1, 4):
                            row_max = T.max(row_max, weights_ub[row + j])
                        for j in T.serial(4):
                            weights_ub[row + j] = T.exp(weights_ub[row + j] - row_max)
                    for iteration in T.serial(repeat):
                        for i in T.serial(4):
                            row_sum = T.alloc_var(T.float32, init=0.0)
                            for j in T.serial(4):
                                row_sum += weights_ub[4 + i * 4 + j]
                            for j in T.serial(4):
                                weights_ub[4 + i * 4 + j] /= row_sum + eps
                        for j in T.serial(4):
                            col_sum = T.alloc_var(T.float32, init=0.0)
                            for i in T.serial(4):
                                col_sum += weights_ub[4 + i * 4 + j]
                            for i in T.serial(4):
                                weights_ub[4 + i * 4 + j] /= col_sum + eps
                    for i in T.serial(20, 24):
                        weights_ub[i] = 0.0
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
                        acc[0] = S.vmul(streams[0], S.vld(weights_ub[4 + row * 4], dist="BRC_B32"))
                        for i in T.unroll(1, 4, explicit=True):
                            S.vmula(acc[0], streams[i], S.vld(weights_ub[4 + row * 4 + i], dist="BRC_B32"))
                        S.vsts(res_ub[row, col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")
            T.copy(res_ub, x_res[token, 0, start])
            T.copy(layer_ub, x_layer[token, start])


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def post(x_layer, h_post, x_res, out, *, batch: int, hidden: int, tile: int):
    tiles = hidden // tile
    x_layer: T.Tensor[(batch, hidden), T.bfloat16]
    h_post: T.Tensor[(batch, 4), T.float32]
    x_res: T.Tensor[(batch, 4, hidden), T.bfloat16]
    out: T.Tensor[(batch, 4, hidden), T.bfloat16]

    cores = min(batch * tiles, 56)
    with T.Kernel(cores) as core:
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        post_ub = T.alloc_shared((8,), T.float32)
        res_ub = T.alloc_shared((4, tile), T.bfloat16)
        out_ub = T.alloc_shared((4, tile), T.bfloat16)
        for token, column_tile in T.Persistent([batch, tiles], cores, core, group_size=1):
            start = column_tile * tile
            T.copy(x_layer[token, start], layer_ub)
            T.copy(h_post[token, 0], post_ub[0:4])
            T.copy(x_res[token, 0, start], res_ub)
            with T.SimdVF():
                coeff = S.alloc_local((4,), T.float32)
                for row in T.unroll(4, explicit=True):
                    coeff[row] = S.vld(post_ub[row], dist="BRC_B32")
                for chunk in T.serial(tile // 64):
                    col = chunk * 64
                    layer = S.vcvt(S.vld(layer_ub[col], dist="UNPK_B16"), T.float32, part=0)
                    for row in T.unroll(4, explicit=True):
                        residual = S.vcvt(S.vld(res_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                        result = S.vadd(residual, S.vmul(layer, coeff[row]))
                        S.vsts(out_ub[row, col], S.vcvt(result, T.bfloat16), dist="PK_B32")
            T.copy(out_ub, out[token, 0, start], l2_cache_ctrl=0)
