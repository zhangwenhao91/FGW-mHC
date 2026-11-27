"""Time the Pre launches and Post on one NPU with identical inputs."""

import argparse
from collections import defaultdict
import csv
import math
from statistics import median

from bench_mhc import CASES, time_us


def stage_times(run, torch, repeat):
    samples = defaultdict(list)
    for _ in range(repeat):
        events = []

        def timed_launch(name, fn, *args):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            result = fn(*args)
            end.record()
            events.append((name, start, end))
            return result

        result = run(timed_launch)
        events[-1][2].synchronize()
        for name, start, end in events:
            duration = start.elapsed_time(end) * 1000
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError(f"Invalid NPU event time for {name}: {duration}")
            samples[name].append(duration)
        del result
    return {name: median(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-stage-results.csv")
    args = parser.parse_args()
    if min(args.warmup, args.repeat) < 1:
        parser.error("--warmup and --repeat must be positive")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(42)
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("batch", "hidden", "projection", "stage", "median_us"))
        for batch, hidden in CASES:
            width = 4 * hidden
            x = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            phi = (torch.randn(width, 24, generator=rng) / math.sqrt(width)).to(device)
            bias = torch.randn(24, generator=rng).to(device)
            layer_out = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
            inputs = (phi, x, bias, 1.0, 1.0, 1.0, 20, 1e-6)
            expected = pre_reference(*inputs)

            for projection in ("cube", "cube_mk"):
                def pair(launch=None):
                    options = {} if launch is None else {"_launch": launch}
                    residual, layer_in, weight = mhc_pre(*inputs, projection=projection, **options)
                    post_inputs = (layer_out, weight, residual)
                    out = mhc_post(*post_inputs) if launch is None else launch("post", mhc_post, *post_inputs)
                    return layer_in, weight, out

                torch.testing.assert_close(
                    mhc_pre(*inputs, projection=projection), expected, atol=0.01, rtol=0.01
                )
                total = time_us(pair, torch, args.warmup, args.repeat)
                stages = stage_times(pair, torch, args.repeat)
                if projection == "cube":
                    stages["project_plus_mix"] = stages["project"] + stages["mix"]
                stages["pair"] = total
                for stage, elapsed in stages.items():
                    writer.writerow((batch, hidden, projection, stage, f"{elapsed:.3f}"))
                output.flush()
                print(f"B={batch} H={hidden} {projection}: " +
                      ", ".join(f"{stage}={elapsed:.3f} us" for stage, elapsed in stages.items()),
                      flush=True)


if __name__ == "__main__":
    main()
