"""Forward-only Ascend 950 implementation of TileOPs mHC."""

from functools import lru_cache
from math import gcd, isqrt

import torch

__all__ = ["mhc_pre", "mhc_post"]


def _check_tensor(name, tensor, shape, dtype, device):
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _check_input(tensor):
    if tensor.device.type != "npu" or tensor.ndim != 2:
        raise ValueError("Expected a rank-2 Ascend NPU tensor")
    return tensor.shape


@lru_cache(maxsize=128)
def _compile(stage, **config):
    from . import _kernels

    return getattr(_kernels, stage).compile(**config)


def _run_kernel(name, kernel, *args):
    kernel(*args)


def mhc_pre(
    phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat,
    sinkhorn_eps=0.02, *, projection="cube", pre_tile=None, _launch=_run_kernel,
):
    """Return (x_res, x_layer, h_post) for BF16 inputs and FP32 weights.

    Infer the stream count n from phi's n * (n + 2) columns. Four streams
    with a hidden size divisible by 64 use the requested TileLang projection;
    other shapes use the same equations through PyTorch on the NPU.
    """
    batch, width = _check_input(x)
    if phi.ndim != 2:
        raise ValueError("phi must be rank 2")
    mixes = phi.shape[1]
    streams = isqrt(mixes + 1) - 1
    if streams <= 0 or streams * (streams + 2) != mixes:
        raise ValueError("phi must have n * (n + 2) columns for a positive stream count n")
    if width <= 0 or width % streams:
        raise ValueError("x width must be positive and divisible by the stream count")
    if type(sinkhorn_repeat) is not int or sinkhorn_repeat < 0:
        raise ValueError("sinkhorn_repeat must be a nonnegative integer")
    hidden = width // streams
    device = x.device
    _check_tensor("x", x, (batch, width), torch.bfloat16, device)
    _check_tensor("phi", phi, (width, mixes), torch.float32, device)
    _check_tensor("b", b, (mixes,), torch.float32, device)

    grouped_routes = {"cube_full_g2": 2, "cube_full_g4": 4}
    if projection not in ("cube", "cube_mk", "cube_full", "cube_full_simd_reduce", "cube_full_simd", "cube_full_tail", "cube_full_k", "cube_m_only", "cube_m_only_k512", "cube_m_only_k512_deferred", "cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched", "cube_split", "vector", *grouped_routes):
        raise ValueError(f"Unknown mHC projection {projection!r}")
    if pre_tile is not None and (
        projection != "cube_full" or streams != 4 or hidden % 64 or type(pre_tile) is not int
        or pre_tile <= 0 or pre_tile > 4096 or pre_tile % 64 or hidden % pre_tile
    ):
        raise ValueError("pre_tile requires four-stream cube_full and a 64-aligned divisor of H up to 4096")
    if batch == 0:
        return torch.empty_like(x), x.new_empty((0, hidden)), b.new_empty((0, streams))
    if streams != 4 or hidden % 64:
        from .reference import pre_reference

        return pre_reference(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps)

    if projection in ("cube_m_only", "cube_m_only_k512", "cube_m_only_k512_deferred", "cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched"):
        chunk = gcd(width, 256) if projection == "cube_m_only" else 512
        if width % chunk:
            raise ValueError(f"{projection} requires K divisible by {chunk}")
        tile = hidden if hidden <= 4096 else gcd(hidden, 256)
        kernel = _compile(
            "project_cube_m_only", batch=batch, width=width, chunk=chunk, tile=tile,
            repeat=sinkhorn_repeat, eps=float(sinkhorn_eps),
            alpha_pre=float(alpha_pre), alpha_post=float(alpha_post), alpha_res=float(alpha_res),
            accumulate_squares=projection in ("cube_m_only_k512_deferred", "cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched"),
            k_pipeline_stages=2 if projection in ("cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched") else 1,
            batch_coefficients=projection == "cube_m_only_k512_batched",
        )
        x_res = torch.empty_like(x)
        layer = x.new_empty((batch, hidden))
        post = b.new_empty((batch, 4))
        _launch("pre", kernel, x, phi, b, post, x_res.view(batch, 4, hidden), layer, post)
        return x_res, layer, post

    cube = projection != "vector"
    full_pre = projection in (*grouped_routes, "cube_full", "cube_full_simd_reduce", "cube_full_simd", "cube_full_tail", "cube_full_k")
    merge = full_pre or projection == "cube_mk"
    fused = merge or projection == "cube"
    chunk = gcd(width, 256)
    if projection == "cube_full_k":
        row_tiles = (batch + 15) // 16
        # Select the smallest aligned K block that gives one job per active Cube core.
        for candidate in (256, 320, 384, 448, 512):
            if width % candidate == 0 and row_tiles * (width // candidate) <= 28:
                chunk = candidate
                break
    k_group = grouped_routes.get(projection, 1)
    if width % (chunk * k_group):
        raise ValueError(f"{projection} requires K divisible by {chunk * k_group}")
    splits = width // (chunk * k_group)
    tile = pre_tile or (hidden if full_pre and hidden <= 4096 else gcd(hidden, 256))
    padded_batch = (batch + 15) // 16 * 16
    launch = dict(batch=batch, width=width, chunk=chunk)
    coefficient_config = dict(
        repeat=sinkhorn_repeat,
        eps=float(sinkhorn_eps),
        alpha_pre=float(alpha_pre),
        alpha_post=float(alpha_post),
        alpha_res=float(alpha_res),
    )
    partial_shape = (splits, padded_batch, 32) if cube else (batch, splits, 32)
    partial = torch.empty(partial_shape, device=device, dtype=torch.float32)
    weights = torch.empty((batch, 24), device=device, dtype=torch.float32)
    x_res = torch.empty_like(x)
    x_layer = torch.empty((batch, hidden), device=device, dtype=torch.bfloat16)
    h_post = torch.empty((batch, 4), device=device, dtype=torch.float32)

    if fused:
        jobs = padded_batch // 16 * splits
        cores = next(candidate for candidate in range(min(jobs, 28), 0, -1) if jobs % candidate == 0)
        # The SIMD coefficient schedule currently requires one token at most per AIV.
        simd_weights = projection == "cube_full_simd" and batch <= 2 * cores
        project = _compile(
            "project_cube_fused", **launch, cores=cores, merge=merge,
            **(coefficient_config if merge else {}),
            full_pre=full_pre, tile=tile if full_pre else 64, simd_weights=simd_weights,
            skip_padding=projection == "cube_full_tail", k_group=k_group,
            simd_reduce=projection == "cube_full_simd_reduce",
        )
        squares = torch.empty((splits, padded_batch, 8), device=device, dtype=torch.float32)
        # Read aliases express the workspace publication boundary to the scheduler.
        stage = "pre" if full_pre else "project_and_mix" if merge else "project"
        _launch(
            stage, project, x, phi, partial, squares, partial, squares,
            b, weights, h_post, x_res.view(batch, 4, hidden), x_layer,
        )
    elif cube:
        prepare = _compile("prepare_cube", **launch)
        project = _compile("project_cube", **launch)
        promoted = torch.empty((batch, width), device=device, dtype=torch.float32)
        squares = torch.empty((batch, splits, 8), device=device, dtype=torch.float32)
        _launch("prepare", prepare, x, promoted, squares)
        _launch("project", project, promoted, phi, partial)
    else:
        _launch("project", _compile("project", **launch), x, phi, partial)
        # The Vector specialization reads its RMS statistics from partial.
        squares = partial
    if not merge:
        mixing = _compile(
            "mix", batch=batch, width=width, splits=splits, cube=cube, fused=fused,
            **coefficient_config,
        )
        _launch("mix", mixing, partial, squares, b, weights, h_post)
    if not full_pre:
        apply = _compile("apply_mix", batch=batch, hidden=hidden, tile=tile)
        _launch("apply_mix", apply, x.view(batch, 4, hidden), weights, x_res.view(batch, 4, hidden), x_layer)
    return x_res, x_layer, h_post


def mhc_post(x_layer_out, h_post, x_res, *, pipeline=False, num_vec_cores=56, streams_per_task=4):
    """Return the flattened BF16 residual plus the scaled layer output."""
    batch, hidden = _check_input(x_layer_out)
    device = x_layer_out.device
    _check_tensor("x_layer_out", x_layer_out, (batch, hidden), torch.bfloat16, device)
    if h_post.ndim != 2:
        raise ValueError("h_post must be rank 2")
    streams = h_post.shape[1]
    _check_tensor("h_post", h_post, (batch, streams), torch.float32, device)
    _check_tensor("x_res", x_res, (batch, streams * hidden), torch.bfloat16, device)

    if type(num_vec_cores) is not int or not 1 <= num_vec_cores <= 56:
        raise ValueError("num_vec_cores must be an integer from 1 to 56")
    if type(streams_per_task) is not int or streams_per_task not in (1, 2, 4):
        raise ValueError("streams_per_task must be 1, 2, or 4")
    out = torch.empty_like(x_res)
    if not out.numel():
        return out
    group_width = gcd(streams, streams_per_task)
    token_groups = batch * (streams // group_width)
    tiles_per_group = max(1, num_vec_cores // token_groups)
    alignment = 64 * tiles_per_group
    tile = min(2048, max(256, ((hidden + alignment - 1) // alignment) * 64))
    jobs = token_groups * ((hidden + tile - 1) // tile)
    cores = min(jobs, num_vec_cores)
    stages = 3 if pipeline and jobs > cores else 0
    kernel = _compile(
        "post", batch=batch, hidden=hidden, tile=tile, cores=cores, stages=stages,
        streams=group_width, total_streams=streams,
    )
    kernel(x_layer_out, h_post, x_res.view(batch, streams, hidden), out.view(batch, streams, hidden))
    return out
