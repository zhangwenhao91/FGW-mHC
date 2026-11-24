"""Measure TileLang and matching CANN or AscendC implementations on Ascend."""

import argparse
import csv
import importlib
import sys
from pathlib import Path
from statistics import median


CASES = ((1, 1280), (2, 1920), (4, 2560))


def time_us(fn, torch, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        del result
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--baseline-module", help="Importable module exporting same-contract AscendC mhc_pre and mhc_post")
    parser.add_argument("--output", type=Path, help="Write CSV to a file")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        for batch, hidden in CASES:
            print(f"B={batch} n=4 hidden={hidden}")
        return
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be positive")

    import torch
    import torch_npu  # noqa: F401

    if args.output is None:
        run(args, sys.stdout, torch)
    else:
        with args.output.open("w", encoding="utf-8", newline="") as output:
            run(args, output, torch)


def run(args, output, torch):
    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    if args.baseline_module:
        module = importlib.import_module(args.baseline_module)
        pre_baseline = module.mhc_pre
        post_baseline = module.mhc_post
        pre_label = post_label = args.baseline_module
    else:
        pre_baseline = pre_reference
        post_baseline = post_reference
        pre_label = "cann_eager_composite"
        post_label = "cann_addcmul_with_casts"

    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("stage", "batch", "hidden", "baseline", "tilelang_us", "baseline_us", "speedup"))
    output.flush()
    rng = torch.Generator().manual_seed(42)
    for batch, hidden in CASES:
        x = torch.randn(batch, 4 * hidden, generator=rng).to(device, torch.bfloat16)
        phi = torch.randn(4 * hidden, 24, generator=rng).to(device)
        bias = torch.randn(24, generator=rng).to(device)
        layer = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
        h_post = torch.randn(batch, 4, generator=rng).to(device)
        pre_args = (phi, x, bias, 1.0, 1.0, 1.0, 1, 0.02)
        post_args = (layer, h_post, x)
        for stage, op, baseline, ref, label, inputs in (
            ("pre", mhc_pre, pre_baseline, pre_reference, pre_label, pre_args),
            ("post", mhc_post, post_baseline, post_reference, post_label, post_args),
        ):
            expected = ref(*inputs)
            torch.testing.assert_close(op(*inputs), expected, atol=0.01, rtol=0.01)
            if baseline is not ref:
                torch.testing.assert_close(baseline(*inputs), expected, atol=0.01, rtol=0.01)
            tilelang_us = time_us(lambda: op(*inputs), torch, args.warmup, args.repeat)
            baseline_us = time_us(lambda: baseline(*inputs), torch, args.warmup, args.repeat)
            if tilelang_us <= 0 or baseline_us <= 0:
                raise RuntimeError(f"Invalid NPU event timing for {stage}, B={batch}, H={hidden}")
            writer.writerow((stage, batch, hidden, label, f"{tilelang_us:.3f}", f"{baseline_us:.3f}", f"{baseline_us / tilelang_us:.3f}"))
            output.flush()


if __name__ == "__main__":
    main()
