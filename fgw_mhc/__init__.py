"""Forward-only Ascend 950 implementation of TileOPs mHC."""

from functools import lru_cache
from math import gcd

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
    batch, width = tensor.shape
    if batch <= 0 or width <= 0 or width % 64:
        raise ValueError("Expected a positive batch and width divisible by 64")
    return batch, width


@lru_cache(maxsize=128)
def _compile(stage, **config):
    from . import _kernels

    return getattr(_kernels, stage).compile(**config)


def _run_kernel(name, kernel, *args):
    kernel(*args)


def mhc_pre(
    phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat,
    sinkhorn_eps=0.02, *, projection="cube", _launch=_run_kernel,
):
    """Return (x_res, x_layer, h_post) for four residual streams.

    Input x is BF16 [B, 4*H]. Phi and bias are FP32 [4*H, 24] and [24].
    H must be divisible by 64. The default mixed Cube projection uses HF32
    multiplication with FP32 accumulation. The split and Vector routes are
    available for performance comparisons.
    """
    batch, width = _check_input(x)
    if width % 256:
        raise ValueError("Expected four streams with hidden size divisible by 64")
    if type(sinkhorn_repeat) is not int or sinkhorn_repeat < 0:
        raise ValueError("sinkhorn_repeat must be a nonnegative integer")
    hidden = width // 4
    device = x.device
    _check_tensor("x", x, (batch, width), torch.bfloat16, device)
    _check_tensor("phi", phi, (width, 24), torch.float32, device)
    _check_tensor("b", b, (24,), torch.float32, device)

    if projection not in ("cube", "cube_mk", "cube_full", "cube_split", "vector"):
        raise ValueError("projection must be 'cube', 'cube_mk', 'cube_full', 'cube_split', or 'vector'")
    cube = projection != "vector"
    full_pre = projection == "cube_full"
    merge = full_pre or projection == "cube_mk"
    fused = merge or projection == "cube"
    chunk = gcd(width, 256)
    splits = width // chunk
    tile = hidden if full_pre and hidden <= 4096 else gcd(hidden, 256)
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
        project = _compile(
            "project_cube_fused", **launch, cores=cores, merge=merge,
            **(coefficient_config if merge else {}),
            **({"full_pre": True, "tile": tile} if full_pre else {}),
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


def mhc_post(x_layer_out, h_post, x_res):
    """Return the flattened BF16 residual plus the scaled layer output."""
    batch, hidden = _check_input(x_layer_out)
    device = x_layer_out.device
    _check_tensor("x_layer_out", x_layer_out, (batch, hidden), torch.bfloat16, device)
    _check_tensor("h_post", h_post, (batch, 4), torch.float32, device)
    _check_tensor("x_res", x_res, (batch, 4 * hidden), torch.bfloat16, device)

    kernel = _compile("post", batch=batch, hidden=hidden, tile=gcd(hidden, 256))
    out = torch.empty_like(x_res)
    kernel(x_layer_out, h_post, x_res.view(batch, 4, hidden), out.view(batch, 4, hidden))
    return out
