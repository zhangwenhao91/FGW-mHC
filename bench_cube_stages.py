"""Time the Pre launches and Post on one NPU with identical inputs."""

import argparse
from collections import defaultdict
from itertools import permutations
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


def replay_stage_times(run, torch, warmup, repeat, launches_per_event):
    launches = []

    def capture(name, kernel, *inputs):
        launches.append((name, kernel, inputs))
        return kernel(*inputs)

    run(capture)
    torch.npu.synchronize()
    stages = {}
    for name, kernel, inputs in launches:
        def replay():
            for _ in range(launches_per_event):
                kernel(*inputs)

        stages[name] = time_us(replay, torch, warmup, repeat) / launches_per_event
    return stages


def ablate_m_only(writer, torch, args, inputs, expected, batch, hidden):
    from fgw_mhc import _compile

    phi, x, bias = inputs[:3]
    width = 4 * hidden
    config = dict(batch=batch, width=width, chunk=512 if args.ablate_k512 else math.gcd(width, 256), tile=hidden,
                  repeat=1, eps=0.02, alpha_pre=1.0, alpha_post=1.0, alpha_res=1.0)
    residual = x.new_empty((batch, 4, hidden))
    layer = x.new_empty((batch, hidden))
    post = bias.new_empty((batch, 4))
    weights = bias.new_empty((batch, 24))
    projection = bias.new_empty((batch, 32))
    if args.projection_k_sweep or args.projection_buffer_sweep:
        flat = x.float()
        expected_projection = flat @ phi
        expected_squares = flat.square().sum(-1)

        def prepare(chunk, stages):
            kernel = _compile(
                "project_cube_m_only", **(config | {"chunk": chunk}),
                projection_only=True, k_pipeline_stages=stages,
            )

            def run():
                kernel(x, phi, bias, post, residual, layer, projection)

            run()
            torch.testing.assert_close(projection[:, :24], expected_projection, atol=0.01, rtol=0.01)
            torch.testing.assert_close(projection[:, 24], expected_squares, atol=0.01, rtol=0.01)
            return run

        if args.projection_buffer_sweep:
            variants = tuple((f"projection_k512_stages{stages}", prepare(512, stages)) for stages in (1, 2))
        else:
            variants = tuple((f"projection_k{chunk}", prepare(chunk, 1)) for chunk in (256, 512))
    else:
        full = _compile("project_cube_m_only", **config)
        project = _compile("project_cube_m_only", **config, projection_only=True)
        ablated = _compile("project_cube_m_only", **config, coefficients_only=True)

        def run_full():
            full(x, phi, bias, post, residual, layer, post)

        def run_ablated():
            ablated(x, phi, bias, post, residual, layer, weights)

        def run_project():
            project(x, phi, bias, post, residual, layer, projection)

        run_full()
        torch.testing.assert_close((residual.flatten(1), layer, post), expected, atol=0.01, rtol=0.01)
        run_ablated()
        streams = x.float().view(batch, 4, hidden)
        rebuilt_residual = torch.bmm(weights[:, 4:20].reshape(batch, 4, 4), streams).flatten(1).bfloat16()
        rebuilt_layer = torch.bmm(weights[:, None, :4], streams).squeeze(1).bfloat16()
        torch.testing.assert_close((rebuilt_residual, rebuilt_layer, post), expected, atol=0.01, rtol=0.01)

        run_project()
        flat = streams.flatten(1)
        torch.testing.assert_close(projection[:, :24], flat @ phi, atol=0.01, rtol=0.01)
        torch.testing.assert_close(projection[:, 24], flat.square().sum(-1), atol=0.01, rtol=0.01)

        variants = (("full_pre", run_full), ("coefficients_only", run_ablated), ("projection_only", run_project))
    samples = defaultdict(list)
    count = args.launches_per_event
    orders = tuple(permutations(variants))
    for order in orders * (2 if len(variants) == 2 else 1):
        for name, run in order:
            def submit():
                for _ in range(count):
                    run()

            duration = time_us(submit, torch, args.warmup, args.repeat) / count
            samples[name].append(duration)
            writer.writerow((batch, hidden, "cube_m_only_k512" if args.ablate_k512 or args.projection_buffer_sweep else "cube_m_only", name, f"{duration:.3f}", count, 1, 0.02))
    for name, values in samples.items():
        print(f"N={batch} H={hidden} {name}: median {median(values):.3f} us, "
              f"range {min(values):.3f}–{max(values):.3f} us", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--simd-reduce", action="store_true", help="Compare baseline and SIMD partial reduction")
    parser.add_argument("--ablate-m-only", action="store_true", help="Compare full Pre, coefficients, and projection at N=1024 and 4096")
    parser.add_argument("--projection-buffer-sweep", action="store_true", help="Compare one and two K pipeline stages at N=1024, K window 512")
    parser.add_argument("--projection-k-sweep", action="store_true", help="Compare projection-only K windows at N=1024")
    parser.add_argument("--ablate-k512", action="store_true", help="Use K512 for the M-only Pre ablation")
    parser.add_argument("--batch", type=int, choices=[batch for batch, _ in CASES])
    parser.add_argument("--tileops-spec", action="store_true", help="Use 1 Sinkhorn iteration and epsilon 0.02")
    parser.add_argument("--launches-per-event", type=int, default=1,
                        help="Replay each prepared stage this many times per timing window")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-stage-results.csv")
    args = parser.parse_args()
    if args.projection_buffer_sweep and (not args.ablate_m_only or args.projection_k_sweep):
        parser.error("--projection-buffer-sweep requires --ablate-m-only without --projection-k-sweep")
    if args.ablate_k512 and (not args.ablate_m_only or args.projection_k_sweep):
        parser.error("--ablate-k512 requires --ablate-m-only without --projection-k-sweep")
    if args.projection_k_sweep and not args.ablate_m_only:
        parser.error("--projection-k-sweep requires --ablate-m-only")
    if args.ablate_m_only and (args.batch is not None or args.simd_reduce):
        parser.error("--ablate-m-only uses its own cases and cannot be combined with --simd-reduce")
    if min(args.warmup, args.repeat, args.launches_per_event) < 1:
        parser.error("warmup, repeat and launches per event must be positive")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(42)
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("batch", "hidden", "projection", "stage", "median_us", "launches_per_event", "sinkhorn_iters", "sinkhorn_eps"))
        cases = ((1024, 2560), (4096, 2560)) if args.ablate_m_only else CASES
        if args.projection_k_sweep or args.projection_buffer_sweep:
            cases = cases[:1]
        for batch, hidden in cases:
            width = 4 * hidden
            x = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            phi = (torch.randn(width, 24, generator=rng) / math.sqrt(width)).to(device)
            bias = torch.randn(24, generator=rng).to(device)
            layer_out = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
            if args.batch is not None and batch != args.batch:
                continue
            iters, eps = (1, 0.02) if args.tileops_spec or args.ablate_m_only else (20, 1e-6)
            inputs = (phi, x, bias, 1.0, 1.0, 1.0, iters, eps)
            expected = None if args.projection_k_sweep or args.projection_buffer_sweep else pre_reference(*inputs)
            if args.ablate_m_only:
                ablate_m_only(writer, torch, args, inputs, expected, batch, hidden)
                output.flush()
                continue

            routes = ("cube_full", "cube_full_simd_reduce") if args.simd_reduce else ("cube", "cube_mk", "cube_full")
            for projection in routes:
                def pair(launch=None):
                    options = {} if launch is None else {"_launch": launch}
                    residual, layer_in, weight = mhc_pre(*inputs, projection=projection, **options)
                    post_inputs = (layer_out, weight, residual)
                    out = mhc_post(*post_inputs) if launch is None else launch("post", mhc_post, *post_inputs)
                    return layer_in, weight, out

                torch.testing.assert_close(
                    mhc_pre(*inputs, projection=projection), expected, atol=0.01, rtol=0.01
                )
                pair_count = args.launches_per_event if args.simd_reduce else 1

                def submit_pairs():
                    for _ in range(pair_count):
                        pair()

                total = time_us(submit_pairs if pair_count > 1 else pair, torch, args.warmup, args.repeat) / pair_count
                if args.launches_per_event == 1:
                    stages = stage_times(pair, torch, args.repeat)
                else:
                    stages = replay_stage_times(pair, torch, args.warmup, args.repeat, args.launches_per_event)
                if projection == "cube":
                    stages["project_plus_mix"] = stages["project"] + stages["mix"]
                stages["pair"] = total
                for stage, elapsed in stages.items():
                    count = pair_count if stage == "pair" else args.launches_per_event
                    writer.writerow((batch, hidden, projection, stage, f"{elapsed:.3f}", count, iters, eps))
                output.flush()
                print(f"B={batch} H={hidden} {projection}: " +
                      ", ".join(f"{stage}={elapsed:.3f} us" for stage, elapsed in stages.items()),
                      flush=True)


if __name__ == "__main__":
    main()
