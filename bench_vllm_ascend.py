"""Compare native FGW-mHC and vLLM AscendC Pre + Post pipelines on one NPU."""

import argparse
import csv
import math
from pathlib import Path

from bench_mhc import CASES, time_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("vllm-ascend-results.csv"))
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        for batch, hidden in CASES:
            print(f"B={batch} n=4 hidden={hidden}")
        return
    if min(args.warmup, args.repeat, args.sinkhorn_iters) < 1:
        parser.error("warmup, repeat and sinkhorn iterations must be positive")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        vllm_pre = torch.ops._C_ascend.npu_hc_pre_v2
        vllm_post = torch.ops._C_ascend.npu_hc_post
    except (ImportError, OSError, AttributeError) as exc:
        parser.exit(1, f"Could not load vLLM Ascend A5 HcPre/HcPost custom ops: {exc}\n")

    eps = 1e-6
    rng = torch.Generator().manual_seed(42)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow((
            "batch", "hidden", "sinkhorn_iters", "tilelang_pair_us", "vllm_ascendc_pair_us",
            "vllm_over_tilelang_speedup", "layer_max_abs_diff", "h_post_max_abs_diff",
            "native_pair_output_max_abs_diff",
        ))
        output.flush()
        for batch, hidden in CASES:
            width = 4 * hidden
            x = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            phi = (torch.randn(width, 24, generator=rng) / math.sqrt(width)).to(device)
            bias = torch.randn(24, generator=rng).to(device)
            layer_out = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
            vllm_x = x.view(1, batch, 4, hidden)
            vllm_phi = phi.transpose(0, 1).contiguous()
            vllm_layer_out = layer_out.view(1, batch, hidden)
            scale = torch.ones(3, dtype=torch.float32, device=device)
            pre_args = (phi, x, bias, 1.0, 1.0, 1.0, args.sinkhorn_iters, eps)
            vllm_args = (vllm_x, vllm_phi, scale, bias, 4, args.sinkhorn_iters, eps, eps)

            def tilelang_pair():
                residual, layer_in, post = mhc_pre(*pre_args)
                return layer_in, post, mhc_post(layer_out, post, residual)

            def vllm_pair():
                layer_in, post, comb = vllm_pre(*vllm_args)
                return layer_in, post, vllm_post(vllm_layer_out, vllm_x, post, comb)

            actual = tilelang_pair()
            residual_ref, layer_ref, post_ref = pre_reference(*pre_args)
            expected = (layer_ref, post_ref, post_reference(layer_out, post_ref, residual_ref))
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            try:
                baseline = vllm_pair()
                torch.npu.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(f"vLLM AscendC failed for B={batch}, hidden={hidden}: {exc}") from exc

            diffs = []
            for fg, vv in zip(actual, baseline, strict=True):
                if not torch.isfinite(vv).all().item():
                    raise RuntimeError(f"Non-finite vLLM AscendC output for B={batch}, hidden={hidden}")
                diffs.append((fg.float() - vv.reshape_as(fg).float()).abs().max().item())

            tilelang_us = time_us(tilelang_pair, torch, args.warmup, args.repeat)
            vllm_us = time_us(vllm_pair, torch, args.warmup, args.repeat)
            if min(tilelang_us, vllm_us) <= 0:
                raise RuntimeError(f"Invalid event time for B={batch}, hidden={hidden}")
            speedup = vllm_us / tilelang_us
            writer.writerow((
                batch, hidden, args.sinkhorn_iters, f"{tilelang_us:.3f}", f"{vllm_us:.3f}",
                f"{speedup:.3f}", *(f"{diff:.6g}" for diff in diffs),
            ))
            output.flush()
            print(f"B={batch} hidden={hidden}: TileLang {tilelang_us:.3f} us, vLLM {vllm_us:.3f} us, {speedup:.3f}x", flush=True)


if __name__ == "__main__":
    main()
