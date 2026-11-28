"""Compare native FGW-mHC and vLLM AscendC Pre + Post pipelines on one NPU."""

import argparse
import csv
from contextlib import nullcontext
import math
import os
import sys
from importlib.util import find_spec
from pathlib import Path
from statistics import median

from bench_mhc import CASES, time_us


LARGE_TOKEN_CASES = ((256, 2560), (1024, 2560), (4096, 2560))


def merge_results(tilelang_path, vllm_path, output_path, cases=CASES):
    tables = []
    projections = {}
    for path, backend in ((tilelang_path, "tilelang"), (vllm_path, "vllm")):
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        stages = {row.get("stage", "pair") for row in rows}
        if stages not in ({"pair"}, {"pre", "post"}):
            raise ValueError(f"{path} must contain either pair timings or separate pre and post timings")
        if len(rows) != len(cases) * len(stages) or any(row["backend"] != backend for row in rows):
            raise ValueError(f"{path} must contain all requested cases for {backend}")
        table = {}
        for row in rows:
            stage = row.get("stage", "pair")
            latency = float(row["pair_us"] if stage == "pair" else row["stage_us"])
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid median latency in {path}: {row}")
            key = tuple(int(row[field]) for field in ("batch", "hidden", "sinkhorn_iters", "warmup", "repeat"))
            key = (*key, float(row.get("sinkhorn_eps", 1e-6)), int(row.get("pairs_per_event", 1)), stage)
            table[key] = latency
            if backend == "tilelang":
                route = row.get("projection", "vector")
                projections[key] = "cube_split" if route == "cube" else route
        if sorted((key[:2], key[-1]) for key in table) != sorted((shape, stage) for shape in cases for stage in stages):
            raise ValueError(f"{path} does not contain the requested benchmark shapes")
        tables.append(table)

    tilelang, vllm = tables
    if tilelang.keys() != vllm.keys():
        raise ValueError("Benchmark stages, shapes, Sinkhorn parameters, warmup and repetitions must match")
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        split = any(key[-1] != "pair" for key in tilelang)
        show_count = any(key[-2] != 1 for key in tilelang)
        count_columns = (("calls_per_event" if split else "pairs_per_event"),) if show_count else ()
        stage_columns = ("stage",) if split else ()
        metric = "stage" if split else "pair"
        writer.writerow(("batch", "hidden", "sinkhorn_iters", "warmup", "repeat", "sinkhorn_eps", *count_columns, *stage_columns,
                         "tilelang_projection", f"tilelang_{metric}_us", f"vllm_ascendc_{metric}_us",
                         "vllm_over_tilelang_speedup"))
        for key, latency in tilelang.items():
            fields = (*key[:-2], *((key[-2],) if show_count else ()), *((key[-1],) if split else ()))
            writer.writerow((*fields, projections[key], f"{latency:.3f}", f"{vllm[key]:.3f}", f"{vllm[key] / latency:.3f}"))


def prepare_vllm_opp():
    """Expose packaged OPP metadata before torch_npu initializes CANN."""
    spec = find_spec("vllm_ascend")
    if spec is None or spec.origin is None:
        raise ImportError("vllm_ascend is not installed")
    vendors = Path(spec.origin).parent / "_cann_ops_custom" / "vendors"
    paths = sorted(str(path) for path in vendors.glob("*") if path.is_dir())
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
        dict.fromkeys(path for path in (*paths, *existing) if path)
    )


def summarize_profile(directory, backend, steps):
    """Report device durations and gaps for consecutive native Pre/Post launches."""
    reports = list(directory.rglob("kernel_details.csv"))
    if not reports:
        print(f"No kernel_details.csv under {directory}", flush=True)
        return
    report = max(reports, key=lambda path: path.stat().st_mtime_ns)
    events = []
    with report.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = {name.strip().lower().replace(" ", ""): name for name in reader.fieldnames or ()}
        required = ("name", "starttime(us)", "duration(us)")
        if any(name not in columns for name in required):
            print(f"{report}: unexpected profiler columns {reader.fieldnames}", flush=True)
            return
        name_col, start_col, duration_col = (columns[name] for name in required)
        for row in reader:
            name = row[name_col].lower()
            if (any(kernel in name for kernel in ("project_cube_fused", "project_cube_m_only")) if backend == "tilelang" else "hcpre" in name):
                stage = "pre"
            elif ("post" in name if backend == "tilelang" else "hcpost" in name):
                stage = "post"
            else:
                continue
            events.append((float(row[start_col]), float(row[duration_col]), stage, row[name_col]))
    events.sort(key=lambda event: event[0])
    if [event[2] for event in events] != ["pre", "post"] * steps:
        print(f"{report}: expected {steps} alternating Pre/Post pairs, matched "
              f"{[(event[2], event[3]) for event in events]}", flush=True)
        return
    pairs = list(zip(events[::2], events[1::2]))
    totals = (
        median(pre[1] for pre, post in pairs),
        median(post[1] for pre, post in pairs),
        median(post[0] - (pre[0] + pre[1]) for pre, post in pairs),
        median(post[0] + post[1] - pre[0] for pre, post in pairs),
    )
    between_pairs = [
        next_pre[0] - (post[0] + post[1])
        for (_, post), (next_pre, _) in zip(pairs, pairs[1:])
    ]
    gap_suffix = f", post_to_next_pre_gap={median(between_pairs):.3f} us" if between_pairs else ""
    print(f"{backend} device medians across {steps} profiled pairs: pre={totals[0]:.3f} us, "
          f"post={totals[1]:.3f} us, pre_to_post_gap={totals[2]:.3f} us, "
          f"pair_span={totals[3]:.3f} us{gap_suffix}", flush=True)


def profile_pair(pair, torch, profiler, directory, warmup, steps):
    for _ in range(warmup):
        pair()
    torch.npu.synchronize()

    directory.mkdir(parents=True, exist_ok=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=steps, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as capture:
        for step in range(steps):
            with torch.profiler.record_function(f"mhc_pair_{step:02d}"):
                pair(trace=True)
            if step == steps - 1:
                torch.npu.synchronize()
            capture.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--projection", choices=("vector", "cube_split", "cube", "cube_mk", "cube_full", "cube_full_simd", "cube_full_tail", "cube_full_k", "cube_m_only", "cube_m_only_k512", "cube_m_only_k512_deferred", "cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched", "tileops_tuned"), default="cube")
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("vllm-ascend-results.csv"))
    parser.add_argument("--sweep-pairs", type=int, nargs="+", metavar="PAIRS",
                        help="Test whole-pair latency with several pair counts per NPU event")
    parser.add_argument("--batch", type=int, choices=[batch for batch, _ in CASES],
                        help="Limit the event batch-size sweep to one workload")
    parser.add_argument("--split-stages", action="store_true", help="Measure each backend’s native Pre and Post separately")
    parser.add_argument("--profile-dir", type=Path, help="Capture consecutive pair launches with the NPU profiler")
    parser.add_argument("--profile-batch", type=int)
    parser.add_argument("--profile-steps", type=int, default=12)
    parser.add_argument("--large-tokens", action="store_true", help="Measure 256, 1024, and 4096 tokens at H=2560")
    parser.add_argument("--n4096-only", action="store_true", help="Limit K512 cases to N=4096")
    parser.add_argument("--full-h-double-post", action="store_true", help="Use the full H tile with double buffering for TileLang Post")
    parser.add_argument("--large-k512-cases", action="store_true", help="Use the 1024 and 4096 token K512 workloads")
    parser.add_argument("--pairs-per-event", type=int, default=1)
    parser.add_argument("--tiling-only", action="store_true", help="Launch vLLM Pre once per case for CANN logging")
    parser.add_argument("--list-cases", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--backend", choices=("both", "tilelang", "vllm"), default="both")
    mode.add_argument("--merge", type=Path, nargs=2, metavar=("TILELANG_CSV", "VLLM_CSV"))
    parser.add_argument("--tileops-spec", action="store_true", help="Use TileOPs mHC manifest Sinkhorn settings")
    args = parser.parse_args()
    if args.tiling_only and (args.backend != "vllm" or args.merge or args.profile_dir
                             or args.split_stages or args.sweep_pairs):
        parser.error("--tiling-only requires --backend vllm without timing modes")
    if args.n4096_only and not args.large_k512_cases:
        parser.error("--n4096-only requires --large-k512-cases")
    if args.full_h_double_post and (not args.n4096_only or args.backend != "tilelang"):
        parser.error("--full-h-double-post requires --n4096-only and --backend tilelang")
    if args.large_k512_cases:
        args.large_tokens = True
        cases = LARGE_TOKEN_CASES[-1:] if args.n4096_only else LARGE_TOKEN_CASES[1:]
    else:
        cases = LARGE_TOKEN_CASES if args.large_tokens else CASES
    if args.pairs_per_event < 1 or (args.pairs_per_event > 1 and (
        args.merge or args.backend == "both" or args.sweep_pairs
    )):
        parser.error("--pairs-per-event must be positive and requires single-backend timing when greater than 1")
    if args.profile_batch is not None and (not args.profile_dir or args.profile_batch not in dict(cases)):
        parser.error("--profile-batch requires --profile-dir and a batch in the selected benchmark cases")
    if args.large_tokens and args.batch is not None:
        parser.error("--large-tokens cannot be used with the small-case batch filters")
    if args.tileops_spec:
        if args.merge or args.backend == "both":
            parser.error("--tileops-spec requires a single backend")
        args.sinkhorn_iters = 1
    if args.profile_dir and (args.merge or args.split_stages or args.backend == "both"):
        parser.error("--profile-dir requires one backend and continuous pair execution")
    if args.sweep_pairs is not None and (
        args.merge or args.split_stages or args.profile_dir or args.backend == "both"
    ):
        parser.error("--sweep-pairs requires a single backend and unprofiled pair execution")
    if args.sweep_pairs is not None and min(args.sweep_pairs) < 1:
        parser.error("pair counts must be positive")
    if args.batch is not None and args.sweep_pairs is None:
        parser.error("--batch requires --sweep-pairs")
    if args.merge:
        merge_results(*args.merge, args.output, cases=cases)
        return
    if args.split_stages and args.backend == "both":
        parser.error("--split-stages requires --backend tilelang or --backend vllm")
    if args.list_cases:
        for batch, hidden in cases:
            print(f"B={batch} n=4 hidden={hidden}")
        return
    if min(args.warmup, args.repeat, args.sinkhorn_iters, args.profile_steps) < 1:
        parser.error("warmup, repeat, sinkhorn iterations and profile steps must be positive")

    if args.backend != "tilelang":
        try:
            prepare_vllm_opp()
        except ImportError as exc:
            parser.exit(1, f"Could not locate vLLM Ascend custom ops: {exc}\n")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    if args.backend != "tilelang":
        try:
            from vllm_ascend.utils import bootstrap_custom_op_env

            bootstrap_custom_op_env()
            import vllm_ascend.vllm_ascend_C  # noqa: F401

            vllm_pre = torch.ops._C_ascend.npu_hc_pre_v2
            vllm_post = torch.ops._C_ascend.npu_hc_post
        except (ImportError, OSError, AttributeError) as exc:
            parser.exit(1, f"Could not load vLLM Ascend A5 HcPre/HcPost custom ops: {exc}\n")

    torch.npu.set_device(device)
    eps = 0.02 if args.tileops_spec else 1e-6
    rng = torch.Generator().manual_seed(42)
    output_context = nullcontext(sys.stdout) if args.profile_dir or args.tiling_only else args.output.open("w", encoding="utf-8", newline="")
    with output_context as output:
        writer = csv.writer(output)
        if args.profile_dir or args.tiling_only:
            pass
        elif args.sweep_pairs:
            writer.writerow(("backend", "batch", "hidden", "sinkhorn_iters", "projection",
                             "pairs_per_event", "order", "pair_us"))
        elif args.split_stages:
            writer.writerow(("backend", "batch", "hidden", "sinkhorn_iters", "warmup", "repeat", "projection", "stage", "stage_us", "sinkhorn_eps", "pairs_per_event"))
        elif args.backend != "both":
            writer.writerow(("backend", "batch", "hidden", "sinkhorn_iters", "warmup", "repeat", "projection", "pair_us", "sinkhorn_eps", "pairs_per_event"))
        else:
            writer.writerow((
                "batch", "hidden", "sinkhorn_iters", "tilelang_projection", "tilelang_pair_us", "vllm_ascendc_pair_us",
                "vllm_over_tilelang_speedup", "layer_max_abs_diff", "h_post_max_abs_diff",
                "native_pair_output_max_abs_diff",
            ))
        output.flush()
        for batch, hidden in cases:
            route = args.projection
            if route == "tileops_tuned":
                route = "cube_m_only" if (batch, hidden) in ((1, 1280), (2, 1920)) else "cube_full"
            projection_label = "cube_fused" if route == "cube" else route
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
            vllm_args = (vllm_x, vllm_phi, scale, bias, 4, args.sinkhorn_iters, 1e-6, eps)

            if args.tiling_only:
                print(f"BEGIN vLLM HcPre N={batch} H={hidden}", flush=True)
                vllm_pre(*vllm_args)
                torch.npu.synchronize()
                print(f"END vLLM HcPre N={batch} H={hidden}", flush=True)
                continue

            post_options = {"post_tile": hidden, "pipeline": True, "pipeline_depth": 2} if args.full_h_double_post else {}
            def tilelang_pair(trace=False):
                if not trace:
                    residual, layer_in, post = mhc_pre(*pre_args, projection=route)
                    return layer_in, post, mhc_post(layer_out, post, residual, **post_options)
                with torch.profiler.record_function("fgw_pre"):
                    residual, layer_in, post = mhc_pre(*pre_args, projection=route)
                with torch.profiler.record_function("fgw_post"):
                    result = mhc_post(layer_out, post, residual, **post_options)
                return layer_in, post, result

            def vllm_pair(trace=False):
                if not trace:
                    layer_in, post, comb = vllm_pre(*vllm_args)
                    return layer_in, post, vllm_post(vllm_layer_out, vllm_x, post, comb)
                with torch.profiler.record_function("vllm_pre"):
                    layer_in, post, comb = vllm_pre(*vllm_args)
                with torch.profiler.record_function("vllm_post"):
                    result = vllm_post(vllm_layer_out, vllm_x, post, comb)
                return layer_in, post, result

            if args.sweep_pairs and args.batch is not None and batch != args.batch:
                continue
            if args.profile_dir and args.profile_batch is not None and batch != args.profile_batch:
                continue

            if args.split_stages:
                if args.backend == "tilelang":
                    def pre_op():
                        return mhc_pre(*pre_args, projection=route)

                    residual, layer_in, post = pre_op()
                    expected = pre_reference(*pre_args)
                    torch.testing.assert_close((residual, layer_in, post), expected, atol=0.01, rtol=0.01)

                    def post_op():
                        return mhc_post(layer_out, post, residual, **post_options)

                    torch.testing.assert_close(
                        post_op(), post_reference(layer_out, post, residual), atol=0.01, rtol=0.01,
                    )
                else:
                    def pre_op():
                        return vllm_pre(*vllm_args)

                    layer_in, post, comb = pre_op()

                    def post_op():
                        return vllm_post(vllm_layer_out, vllm_x, post, comb)

                    if any(not torch.isfinite(tensor).all().item() for tensor in (layer_in, post, comb, post_op())):
                        raise RuntimeError(f"Non-finite vLLM AscendC output for B={batch}, hidden={hidden}")

                # The Post inputs stay resident and are prepared before the timed trials.
                for stage, op in (("pre", pre_op), ("post", post_op)):
                    def submit_stage():
                        for _ in range(args.pairs_per_event - 1):
                            op()
                        return op()

                    submit = op if args.pairs_per_event == 1 else submit_stage
                    latency = time_us(submit, torch, args.warmup, args.repeat) / args.pairs_per_event
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError(f"Invalid {stage} event time for B={batch}, hidden={hidden}")
                    projection = projection_label if args.backend == "tilelang" else ""
                    writer.writerow((args.backend, batch, hidden, args.sinkhorn_iters,
                                     args.warmup, args.repeat, projection, stage, f"{latency:.6f}", eps, args.pairs_per_event))
                    output.flush()
                    print(f"{args.backend} {stage} B={batch} hidden={hidden}: {latency:.3f} us", flush=True)
                continue

            if args.backend != "vllm":
                actual = tilelang_pair()
                residual_ref, layer_ref, post_ref = pre_reference(*pre_args)
                expected = (layer_ref, post_ref, post_reference(layer_out, post_ref, residual_ref))
                if args.large_tokens:
                    residual, layer, post = mhc_pre(*pre_args, projection=route)
                    for name, observed, target in (
                        ("pre residual", residual, residual_ref),
                        ("pre layer", layer, layer_ref),
                        ("pre post weights", post, post_ref),
                        ("post with identical inputs", actual[2], post_reference(layer_out, post, residual)),
                        ("pair output", actual[2], expected[2]),
                    ):
                        # The composed BF16 output includes both stages' rounding errors.
                        atol = 0.02 if name == "pair output" else 0.01
                        try:
                            torch.testing.assert_close(observed, target, atol=atol, rtol=0.01)
                        except AssertionError as exc:
                            raise AssertionError(
                                f"{route} N={batch} H={hidden} {name}: {exc}"
                            ) from exc
                    pair_error = (actual[2].float() - expected[2].float()).abs().max().item()
                    print(f"{route} N={batch} H={hidden} pair max abs error: {pair_error:.6g}", flush=True)
                else:
                    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            if args.backend != "tilelang":
                try:
                    baseline = vllm_pair()
                    torch.npu.synchronize()
                except RuntimeError as exc:
                    raise RuntimeError(f"vLLM AscendC failed for B={batch}, hidden={hidden}: {exc}") from exc
                if any(not torch.isfinite(tensor).all().item() for tensor in baseline):
                    raise RuntimeError(f"Non-finite vLLM AscendC output for B={batch}, hidden={hidden}")

            if args.sweep_pairs:
                pair = tilelang_pair if args.backend == "tilelang" else vllm_pair
                projection = projection_label if args.backend == "tilelang" else ""
                sizes = tuple(dict.fromkeys(args.sweep_pairs))
                for order, counts in (("forward", sizes), ("reverse", sizes[::-1])):
                    for count in counts:
                        def submit():
                            result = None
                            for _ in range(count):
                                result = pair()
                            return result

                        elapsed = time_us(pair if count == 1 else submit, torch, args.warmup, args.repeat)
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise RuntimeError(f"Invalid event time for {count} pairs: {elapsed}")
                        per_pair = elapsed / count
                        writer.writerow((args.backend, batch, hidden, args.sinkhorn_iters, projection,
                                         count, order, f"{per_pair:.6f}"))
                        output.flush()
                        print(f"{args.backend} B={batch} H={hidden} {count} pairs/event {order}: "
                              f"{per_pair:.3f} us/pair", flush=True)
                continue

            if args.profile_dir:
                pair = tilelang_pair if args.backend == "tilelang" else vllm_pair
                def submit_profile_bracket():
                    for _ in range(args.pairs_per_event - 1):
                        pair()
                    return pair()

                bracket = pair if args.pairs_per_event == 1 else submit_profile_bracket
                directory = args.profile_dir / args.backend / f"b{batch}-h{hidden}"
                directory.mkdir(parents=True, exist_ok=True)
                timings = directory / "unprofiled-pair.csv"
                with timings.open("w", encoding="utf-8", newline="") as summary:
                    timing_writer = csv.writer(summary)
                    timing_writer.writerow(("backend", "batch", "hidden", "sinkhorn_iters",
                                            "warmup", "repeat", "phase", "pair_us", "pairs_per_event"))
                    for phase in ("before", "after"):
                        if phase == "after":
                            profile_pair(pair, torch, torch_npu.profiler, directory, args.warmup, args.profile_steps)
                        latency = time_us(bracket, torch, args.warmup, args.repeat) / args.pairs_per_event
                        if not math.isfinite(latency) or latency <= 0:
                            raise RuntimeError(f"Invalid unprofiled pair time for B={batch}, H={hidden}: {latency}")
                        timing_writer.writerow((args.backend, batch, hidden, args.sinkhorn_iters,
                                                args.warmup, args.repeat, phase, f"{latency:.6f}", args.pairs_per_event))
                        summary.flush()
                        print(f"{args.backend} B={batch} H={hidden} {phase} trace: {latency:.3f} us", flush=True)
                summarize_profile(directory, args.backend, args.profile_steps)
                print(f"Trace for {args.backend} B={batch} H={hidden}: {directory}", flush=True)
                continue

            if args.backend != "both":
                pair = tilelang_pair if args.backend == "tilelang" else vllm_pair
                def submit_pairs():
                    for _ in range(args.pairs_per_event - 1):
                        pair()
                    return pair()

                submit = pair if args.pairs_per_event == 1 else submit_pairs
                latency = time_us(submit, torch, args.warmup, args.repeat) / args.pairs_per_event
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError(f"Invalid event time for B={batch}, hidden={hidden}")
                writer.writerow((args.backend, batch, hidden, args.sinkhorn_iters,
                                 args.warmup, args.repeat, projection_label if args.backend == "tilelang" else "", f"{latency:.6f}", eps, args.pairs_per_event))
                output.flush()
                print(f"{args.backend} B={batch} hidden={hidden}: {latency:.3f} us", flush=True)
                continue

            diffs = []
            for fg, vv in zip(actual, baseline, strict=True):
                diffs.append((fg.float() - vv.reshape_as(fg).float()).abs().max().item())

            tilelang_us = time_us(tilelang_pair, torch, args.warmup, args.repeat)
            vllm_us = time_us(vllm_pair, torch, args.warmup, args.repeat)
            if min(tilelang_us, vllm_us) <= 0:
                raise RuntimeError(f"Invalid event time for B={batch}, hidden={hidden}")
            speedup = vllm_us / tilelang_us
            writer.writerow((
                batch, hidden, args.sinkhorn_iters, projection_label, f"{tilelang_us:.3f}", f"{vllm_us:.3f}",
                f"{speedup:.3f}", *(f"{diff:.6g}" for diff in diffs),
            ))
            output.flush()
            print(f"B={batch} hidden={hidden}: TileLang {tilelang_us:.3f} us, vLLM {vllm_us:.3f} us, {speedup:.3f}x", flush=True)


if __name__ == "__main__":
    main()
