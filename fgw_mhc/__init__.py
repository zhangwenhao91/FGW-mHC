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


def mhc_pre(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps=0.02):
    """Return (x_res, x_layer, h_post) for four residual streams.

    Input x is BF16 [B, 4*H]. Phi and bias are FP32 [4*H, 24] and [24].
    H must be divisible by 64.
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

    chunk = gcd(width, 256)
    splits = width // chunk
    tile = gcd(hidden, 256)
    projection = _compile("project", batch=batch, width=width, chunk=chunk)
    mixing = _compile(
        "mix",
        batch=batch,
        width=width,
        splits=splits,
        repeat=sinkhorn_repeat,
        eps=float(sinkhorn_eps),
        alpha_pre=float(alpha_pre),
        alpha_post=float(alpha_post),
        alpha_res=float(alpha_res),
    )
    apply = _compile("apply_mix", batch=batch, hidden=hidden, tile=tile)
    partial = torch.empty((batch, splits, 32), device=device, dtype=torch.float32)
    weights = torch.empty((batch, 24), device=device, dtype=torch.float32)
    x_res = torch.empty_like(x)
    x_layer = torch.empty((batch, hidden), device=device, dtype=torch.bfloat16)
    h_post = torch.empty((batch, 4), device=device, dtype=torch.float32)

    projection(x, phi, partial)
    mixing(partial, b, weights, h_post)
    apply(x.view(batch, 4, hidden), weights, x_res.view(batch, 4, hidden), x_layer)
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
