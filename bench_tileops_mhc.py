"""Benchmark FGW-mHC Pre and Post separately against TileOPs Ascend."""
import argparse
import csv
from functools import partial
import math
from pathlib import Path
from statistics import median
import sys

from bench_mhc import CASES, time_us


def compare(fgw, upstream, reference, torch, args):
    for op in (fgw, upstream):
        torch.testing.assert_close(op(), reference, atol=0.01, rtol=0.01)
    torch.npu.synchronize()

    samples = {"fgw": [], "upstream": []}
    candidates = (("fgw", fgw), ("upstream", upstream))
    for index in range(args.rounds):
        for name, op in (candidates if index % 2 == 0 else candidates[::-1]):
            elapsed = time_us(op, torch, args.warmup, args.repeat)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid {name} event time: {elapsed}")
            samples[name].append(elapsed)
    return median(samples["fgw"]), median(samples["upstream"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops-repo", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--projection", default="cube_full",
                        choices=("cube", "cube_mk", "cube_full", "cube_full_simd", "cube_full_tail", "cube_full_k"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sinkhorn-iters", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--output", type=Path, default=Path("tileops-mhc-results.csv"))
    args = parser.parse_args()
    if min(args.warmup, args.repeat, args.rounds, *args.sinkhorn_iters) < 1:
        parser.error("warmup, repeat, rounds and Sinkhorn iteration counts must be positive")
    source = args.tileops_repo.resolve() / "src"
    for stage in ("pre", "post"):
        if not (source / f"tileops/_kernels/ascend/sequence_modeling/mhc_{stage}.py").is_file():
            parser.error(f"Missing Ascend mHC {stage} kernel under {source}")

    sys.path.insert(0, str(source))
    import torch
    import torch_npu  # noqa: F401
    # A source checkout has no installed backend entry-point metadata.
    import tileops.ascend  # noqa: F401
    from tileops.sequence_modeling import MHCPostFwdOp, MHCPreFwdOp

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(42)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("stage", "batch", "hidden", "sinkhorn_iters", "sinkhorn_eps",
                         "fgw_variant", "fgw_us", "tileops_ascend_us", "speedup"))

        def record(stage, batch, hidden, iters, eps, variant, ours, theirs, ref):
            fgw_us, tileops_us = compare(ours, theirs, ref, torch, args)
            speedup = tileops_us / fgw_us
            writer.writerow((stage, batch, hidden, iters, eps, variant,
                             f"{fgw_us:.3f}", f"{tileops_us:.3f}", f"{speedup:.3f}"))
            output.flush()
            print(f"{stage} B={batch} H={hidden} iters={iters}: "
                  f"FGW {fgw_us:.3f} us, TileOPs {tileops_us:.3f} us, {speedup:.3f}x", flush=True)

        for batch, hidden in CASES:
            width = 4 * hidden
            x = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            phi = (torch.randn(width, 24, generator=rng) / math.sqrt(width)).to(device)
            bias = torch.randn(24, generator=rng).to(device)
            pre_inputs = (phi, x, bias)
            for repeat in args.sinkhorn_iters:
                eps = 0.02
                params = (1.0, 1.0, 1.0, repeat, eps)
                pre = MHCPreFwdOp(*params, target="ascend")
                record("pre", batch, hidden, repeat, eps, args.projection,
                       partial(mhc_pre, *pre_inputs, *params, projection=args.projection),
                       partial(pre, *pre_inputs), pre_reference(*pre_inputs, *params))

            layer = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
            weights = torch.randn(batch, 4, generator=rng).to(device)
            residual = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            post_inputs = (layer, weights, residual)
            post = MHCPostFwdOp(target="ascend")
            record("post", batch, hidden, "", "", "default",
                   partial(mhc_post, *post_inputs), partial(post, *post_inputs),
                   post_reference(*post_inputs))


if __name__ == "__main__":
    main()
