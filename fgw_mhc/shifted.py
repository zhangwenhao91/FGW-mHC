"""Shifted mHC kernels for precomputed projections and four residual streams."""
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def shifted_pre(residual, projection, squares, scale, bias, previous, layer, next_pre, post, comb, *, hidden: int):
    tokens = T.dynamic("tokens")
    residual: T.Tensor[(tokens, 4, hidden), T.bfloat16]
    projection: T.Tensor[(1, tokens, 24), T.float32]
    squares: T.Tensor[(1, tokens), T.float32]
    scale: T.Tensor[(3,), T.float32]
    bias: T.Tensor[(24,), T.float32]
    previous: T.Tensor[(tokens, 4), T.float32]
    layer: T.Tensor[(tokens, hidden), T.bfloat16]
    next_pre: T.Tensor[(tokens, 4, 1), T.float32]
    post: T.Tensor[(tokens, 4, 1), T.float32]
    comb: T.Tensor[(tokens, 4, 4), T.float32]
    cores = 56
    tile = 512
    eps = 1e-6

    with T.Kernel(cores) as core:
        proj = T.alloc_shared((24,), T.float32)
        sq = T.alloc_shared((8,), T.float32)
        scales = T.alloc_shared((8,), T.float32)
        biases = T.alloc_shared((24,), T.float32)
        prev = T.alloc_shared((8,), T.float32)
        next_ub = T.alloc_shared((8,), T.float32)
        post_ub = T.alloc_shared((8,), T.float32)
        comb_ub = T.alloc_shared((4, 4), T.float32)
        x_ub = T.alloc_shared((4, tile), T.bfloat16)
        layer_ub = T.alloc_shared((tile,), T.bfloat16)
        T.copy(scale, scales[0:3])
        T.copy(bias, biases)
        for token in T.Persistent([tokens], cores, core):
            T.copy(projection[0, token, :], proj)
            T.copy(squares[0, token:token + 1], sq[0:1])
            T.copy(previous[token, :], prev[0:4])
            with T.SimtVF(threads=32):
                for lane in T.Parallel(1):
                    matrix = T.alloc_local((4, 4), T.float32)
                    inv_rms = T.rsqrt(sq[0] / (4 * hidden) + 1e-20)
                    for row in T.serial(4):
                        pre_logit = proj[row] * inv_rms * scales[0] + biases[row]
                        post_logit = proj[row + 4] * inv_rms * scales[1] + biases[row + 4]
                        next_ub[row] = 1.0 / (1.0 + T.exp(-pre_logit)) + eps
                        post_ub[row] = 2.0 / (1.0 + T.exp(-post_logit))
                        for col in T.serial(4):
                            index = 8 + row * 4 + col
                            matrix[row, col] = proj[index] * inv_rms * scales[2] + biases[index]
                        maximum = T.alloc_var(T.float32, init=matrix[row, 0])
                        for col in T.serial(1, 4):
                            maximum = T.max(maximum, matrix[row, col])
                        total = T.alloc_var(T.float32, init=0.0)
                        for col in T.serial(4):
                            matrix[row, col] = T.exp(matrix[row, col] - maximum)
                            total += matrix[row, col]
                        for col in T.serial(4):
                            matrix[row, col] = matrix[row, col] / total + eps

                    for iteration in T.serial(20):
                        if iteration > 0:
                            for row in T.serial(4):
                                row_total = T.alloc_var(T.float32, init=eps)
                                for col in T.serial(4):
                                    row_total += matrix[row, col]
                                for col in T.serial(4):
                                    matrix[row, col] /= row_total
                        for col in T.serial(4):
                            col_total = T.alloc_var(T.float32, init=eps)
                            for row in T.serial(4):
                                col_total += matrix[row, col]
                            for row in T.serial(4):
                                matrix[row, col] /= col_total
                    for row in T.serial(4):
                        for col in T.serial(4):
                            comb_ub[row, col] = matrix[row, col]
            T.copy(next_ub[0:4], next_pre[token, 0:4, 0])
            T.copy(post_ub[0:4], post[token, 0:4, 0])
            T.copy(comb_ub, comb[token, :, :])

            for block in T.serial(hidden // tile):
                start = block * tile
                T.copy(residual[token, 0, start], x_ub)
                with T.SimdVF():
                    coeff = S.alloc_local((4,), T.float32)
                    for row in T.unroll(4, explicit=True):
                        coeff[row] = S.vld(prev[row], dist="BRC_B32")
                    for part in T.serial(tile // 64):
                        col = part * 64
                        acc = S.alloc_local((1,), T.float32)
                        acc[0] = S.vdup(T.float32(0), T.float32)
                        for row in T.unroll(4, explicit=True):
                            values = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
                            S.vmula(acc[0], values, coeff[row])
                        S.vsts(layer_ub[col], S.vcvt(acc[0], T.bfloat16), dist="PK_B32")
                T.copy(layer_ub, layer[token, start])

@tilelang.jit(target="ascend", execution_backend="tvm_ffi")
def shifted_post(layer, residual, post_mix, comb_mix, output, *, hidden: int, h_blk: int):
    tokens = T.dynamic("tokens")
    layer: T.Tensor[(tokens, hidden), T.bfloat16]
    residual: T.Tensor[(tokens, 4, hidden), T.bfloat16]
    post_mix: T.Tensor[(tokens, 4), T.float32]
    comb_mix: T.Tensor[(tokens, 4, 4), T.float32]
    output: T.Tensor[(tokens, 4, hidden), T.bfloat16]
    cores = 56

    with T.Kernel(cores) as core:
        layer_ub = T.alloc_shared((h_blk,), T.bfloat16)
        residual_ub = T.alloc_shared((4, h_blk), T.bfloat16)
        post_ub = T.alloc_shared((8,), T.float32)
        comb_ub = T.alloc_shared((4, 4), T.float32)
        out_ub = T.alloc_shared((4, h_blk), T.bfloat16)
        for token, block in T.Persistent([tokens, hidden // h_blk], cores, core, group_size=1):
            start = block * h_blk
            T.copy(layer[token, start], layer_ub)
            T.copy(residual[token, 0, start], residual_ub)
            T.copy(post_mix[token, :], post_ub[0:4])
            T.copy(comb_mix[token, :, :], comb_ub)
            with T.SimdVF():
                for tile in T.serial(h_blk // 64):
                    col = tile * 64
                    activation = S.vcvt(S.vld(layer_ub[col], dist="UNPK_B16"), T.float32, part=0)
                    inputs = S.alloc_local((4,), T.float32)
                    for source in T.unroll(4, explicit=True):
                        inputs[source] = S.vcvt(
                            S.vld(residual_ub[source, col], dist="UNPK_B16"), T.float32, part=0
                        )
                    value = S.alloc_local((1,), T.float32)
                    for dest in T.unroll(4, explicit=True):
                        value[0] = S.vmul(activation, S.vld(post_ub[dest], dist="BRC_B32"))
                        for source in T.unroll(4, explicit=True):
                            weight = S.vld(comb_ub[source, dest], dist="BRC_B32")
                            S.vmula(value[0], inputs[source], weight)
                        S.vsts(out_ub[dest, col], S.vcvt(value[0], T.bfloat16), dist="PK_B32")
            T.copy(out_ub, output[token, 0, start], l2_cache_ctrl=0)
