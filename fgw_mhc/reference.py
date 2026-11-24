"""Independent eager references used for correctness and latency comparisons."""

import torch


def pre_reference(phi, x, bias, alpha_pre, alpha_post, alpha_res, repeat, eps):
    batch, width = x.shape
    streams = x.float().reshape(batch, 4, width // 4)
    flat = streams.reshape(batch, width)
    norm = flat.square().mean(dim=-1, keepdim=True).sqrt() + 0.0001
    projected = (flat @ phi) / norm
    pre, post, residual = projected.split((4, 4, 16), dim=-1)
    b_pre, b_post, b_res = bias.split((4, 4, 16), dim=-1)
    h_pre = (alpha_pre * pre + b_pre).sigmoid()
    h_post = 2 * (alpha_post * post + b_post).sigmoid()
    residual = (alpha_res * residual + b_res).reshape(batch, 4, 4)
    residual = (residual - residual.amax(dim=-1, keepdim=True)).exp()
    for _ in range(repeat):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + eps)
    x_res = torch.bmm(residual, streams).reshape(batch, width).bfloat16()
    x_layer = torch.bmm(h_pre[:, None, :], streams).squeeze(1).bfloat16()
    return x_res, x_layer, h_post


def post_reference(x_layer, h_post, x_res):
    residual = x_res.float().reshape(x_layer.shape[0], 4, x_layer.shape[1])
    scaled = torch.addcmul(residual, x_layer.float()[:, None, :], h_post[:, :, None])
    return scaled.reshape_as(x_res).bfloat16()
