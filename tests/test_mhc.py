import importlib.util

import pytest
import torch

from fgw_mhc import mhc_post, mhc_pre
from fgw_mhc.reference import post_reference, pre_reference


@pytest.fixture(scope="module")
def device():
    pytest.importorskip("torch_npu")
    if importlib.util.find_spec("tilelang") is None or not torch.npu.is_available():
        pytest.skip("Requires Ascend 950 and the TileLang Ascend compiler")
    return torch.device("npu:0")


@pytest.mark.parametrize("batch,hidden,repeat", [(1, 1280, 1), (2, 1920, 1), (4, 2560, 1), (3, 64, 20)])
@pytest.mark.parametrize("projection", ["vector", "cube_split", "cube", "cube_mk", "cube_full", "cube_full_simd", "cube_full_tail"])
def test_pre_post(device, batch, hidden, repeat, projection):
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(batch, 4 * hidden, generator=rng).bfloat16()
    phi = torch.randn(4 * hidden, 24, generator=rng)
    bias = torch.randn(24, generator=rng)
    args = (0.7, -0.3, 0.5, repeat, 0.02)

    expected = pre_reference(phi, x, bias, *args)
    options = {} if projection == "cube" else {"projection": projection}
    actual = mhc_pre(phi.to(device), x.to(device), bias.to(device), *args, **options)
    for output, ref in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), ref, atol=0.01, rtol=0.01)

    layer_out = torch.randn(batch, hidden, generator=rng).bfloat16()
    layer_out_npu = layer_out.to(device)
    result = mhc_post(layer_out_npu, actual[2], actual[0])
    torch.testing.assert_close(
        result.cpu(), post_reference(layer_out, expected[2], expected[0]), atol=0.02, rtol=0.01
    )

    residual = torch.randn(batch, 4 * hidden, generator=rng).bfloat16()
    post_weights = torch.randn(batch, 4, generator=rng)
    result = mhc_post(layer_out_npu, post_weights.to(device), residual.to(device))
    torch.testing.assert_close(
        result.cpu(), post_reference(layer_out, post_weights, residual), atol=0.01, rtol=0.01
    )
