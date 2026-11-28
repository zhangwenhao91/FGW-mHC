"""Time the Pre launches and Post on one NPU with identical inputs."""

import argparse
from collections import defaultdict
from itertools import permutations
from pathlib import Path
import csv
from decimal import Decimal
from difflib import unified_diff
import math
from statistics import median
import sys

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


def profile_projection(run, torch, profiler, directory, warmup, launches, *, warmup_run=None):
    warmup_run = run if warmup_run is None else warmup_run
    for _ in range(warmup):
        warmup_run()
    torch.npu.synchronize()
    directory.mkdir(parents=True, exist_ok=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as capture:
        for _ in range(launches):
            run()
        torch.npu.synchronize()
        capture.step()

    reports = sorted(path for path in directory.rglob("*.csv")
                     if path.name.startswith(("kernel_details", "op_summary", "ai_core")))
    if not reports:
        print(f"No pipe metrics CSV recognized under {directory}", flush=True)
    for path in reports:
        if path.name == "kernel_details.csv":
            with path.open(newline="", encoding="utf-8-sig") as source:
                rows = [row for row in csv.DictReader(source) if "project_cube_m_only" in row.get("Name", "")]
            metrics = ("Duration(us)", "aic_mac_time(us)", "aic_mte1_time(us)", "aic_mte2_time(us)",
                       "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)")
            summaries = []
            for metric in metrics:
                values = [float(row[metric]) for row in rows if row.get(metric, "").strip() not in ("", "N/A", "-")]
                if values:
                    summaries.append(f"{metric}={median(values):.3f}")
            if warmup_run is not run and len(rows) > 1:
                rows.sort(key=lambda row: float(row["Start Time(us)"]))
                for metric in ("Duration(us)", "aiv_mte2_time(us)"):
                    values = [row.get(metric, "").strip() for row in rows]
                    if all(value not in ("", "N/A", "-") for value in values):
                        print(f"{directory.name} Pre {metric}: first={float(values[0]):.3f}, "
                              f"subsequent_median={median(map(float, values[1:])):.3f}", flush=True)
            if summaries:
                print(f"{directory.name} Pre device medians ({len(rows)} rows): " + ", ".join(summaries), flush=True)
        print(f"Pipe profiler report: {path}", flush=True)
        with path.open(encoding="utf-8-sig", errors="replace") as source:
            for _ in range(2):
                line = source.readline()
                if line:
                    print(line.rstrip(), flush=True)


def summarize_post_cache(directory):
    summaries = {}
    for policy, context in (("normal_fv", "continuous_pair"), ("notalloc_ci", "continuous_pair_notalloc_ci")):
        reports = sorted((directory / context).rglob("kernel_details.csv"))
        if len(reports) != 1:
            raise ValueError(f"Expected one kernel_details.csv for {context}, found {len(reports)}")

        events = []
        with reports[0].open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                name = row["Name"].lower()
                if "project_cube_m_only" in name:
                    stage = "pre"
                elif "post" in name:
                    stage = "post"
                else:
                    continue
                events.append((Decimal(row["Start Time(us)"].strip()), Decimal(row["Duration(us)"].strip()), stage))
        events.sort(key=lambda event: event[0])
        pairs = list(zip(events[::2], events[1::2]))
        if len(pairs) < 2 or [event[2] for event in events] != ["pre", "post"] * len(pairs):
            raise ValueError(f"Expected at least two alternating Pre/Post pairs in {reports[0]}")

        def metrics(samples):
            return (
                median(pre[1] for pre, post in samples),
                median(post[1] for pre, post in samples),
                median(post[0] - (pre[0] + pre[1]) for pre, post in samples),
                median(post[0] + post[1] - pre[0] for pre, post in samples),
            )

        summaries[policy] = (metrics(pairs[:1]), metrics(pairs[1:]))
        print(f"{policy}: {reports[0]} ({len(pairs)} pairs)")

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("sample", "metric", "normal_fv_us", "notalloc_ci_us", "notalloc_minus_normal_us"))
    for sample, normal, notalloc in zip(("first", "subsequent_median"), summaries["normal_fv"], summaries["notalloc_ci"]):
        for name, first, second in zip(("pre", "post", "pre_to_post_gap", "pair_span"), normal, notalloc):
            writer.writerow((sample, name, f"{first:.3f}", f"{second:.3f}", f"{second - first:+.3f}"))


def profile_pre_context(writer, torch, args, inputs, layer_out, expected, batch, hidden):
    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference
    from torch_npu import profiler

    def pre():
        return mhc_pre(*inputs, projection="cube_m_only_k512_batched")

    def run_post(residual, post, cache):
        return mhc_post(
            layer_out, post, residual, post_tile=hidden, pipeline=True,
            pipeline_depth=2, output_cache=cache,
        )

    def pair(cache, run_pre=pre):
        residual, _, post = run_pre()
        return run_post(residual, post, cache)

    residual, layer, post = pre()
    torch.testing.assert_close((residual, layer, post), expected, atol=0.01, rtol=0.01)
    expected_post = post_reference(layer_out, post, residual)
    caches = ("normal_fv", "notalloc_ci") if args.compare_post_cache else ("normal_fv",)
    for cache in caches:
        torch.testing.assert_close(run_post(residual, post, cache), expected_post, atol=0.01, rtol=0.01)

    variants = (("standalone_pre", pre), ("continuous_pair", lambda: pair("normal_fv")))
    if args.compare_post_cache:
        variants += (("continuous_pair_notalloc_ci", lambda: pair("notalloc_ci")),)
    if args.compare_pre_input_reuse:
        phi, x, bias, *parameters = inputs
        alternating_inputs = (inputs, (phi, x.clone(), bias, *parameters))
        next_input = 0

        def alternating_pre():
            nonlocal next_input
            result = mhc_pre(*alternating_inputs[next_input], projection="cube_m_only_k512_batched")
            next_input ^= 1
            return result

        alternating_pre()
        torch.testing.assert_close(alternating_pre(), expected, atol=0.01, rtol=0.01)
        variants += (("alternating_input_pre", alternating_pre),)
    cached_variants = ()
    if args.compare_ub_residual:
        def cached_pre():
            return mhc_pre(*inputs, projection="cube_m_only_k512_cached_input")

        torch.testing.assert_close(cached_pre(), expected, atol=0.01, rtol=0.01)
        torch.testing.assert_close(
            pair("normal_fv", cached_pre), expected_post, atol=0.02, rtol=0.01,
        )
        cached_variants = (
            ("standalone_pre_ub_residual", cached_pre),
            ("continuous_pair_ub_residual", lambda: pair("normal_fv", cached_pre)),
        )
        variants += cached_variants
    cached_names = {name for name, _ in cached_variants}
    count = args.launches_per_event
    for phase in ("before", "after"):
        if phase == "after":
            for name, run in variants:
                profile_projection(
                    run, torch, profiler, args.profile_pre_context / name, args.warmup,
                    max(8, count), warmup_run=pre if args.compare_post_cache else None,
                )
        for name, run in (variants if phase == "before" else variants[::-1]):
            def submit():
                for _ in range(count - 1):
                    run()
                return run()

            elapsed = time_us(submit, torch, args.warmup, args.repeat) / count
            route = "cube_m_only_k512_cached_input" if name in cached_names else "cube_m_only_k512_batched"
            writer.writerow((batch, hidden, route, f"{name}_{phase}", f"{elapsed:.3f}", count, 1, 0.02))
            print(f"N={batch} H={hidden} {name} {phase} profile: {elapsed:.3f} us", flush=True)


def ablate_m_only(writer, torch, args, inputs, expected, batch, hidden):
    from fgw_mhc import _compile

    phi, x, bias = inputs[:3]
    width = 4 * hidden
    config = dict(batch=batch, width=width, chunk=512 if args.ablate_k512 else math.gcd(width, 256), tile=hidden,
                  repeat=1, eps=0.02, alpha_pre=1.0, alpha_post=1.0, alpha_res=1.0)
    if args.ablate_deferred_s2:
        config.update(accumulate_squares=True, k_pipeline_stages=2)
    residual = x.new_empty((batch, 4, hidden))
    layer = x.new_empty((batch, hidden))
    post = bias.new_empty((batch, 4))
    weights = bias.new_empty((batch, 24))
    projection = bias.new_empty((batch, 32))
    if args.projection_k_sweep or args.projection_buffer_sweep or args.profile_projection or args.projection_m_tile_sweep or args.projection_mk_compare:
        flat = x.float()
        expected_projection = flat @ phi
        expected_squares = flat.square().sum(-1)

        sources = {}

        def prepare(chunk, stages, a_ping_pong=False, compute_squares=True, accumulate_squares=False, rows=16, mad_k=0, separate_nz=False, block_nz=False, weight_nk=False, l0_ping_pong=False):
            kernel = _compile(
                "project_cube_m_only", **(config | {"chunk": chunk}),
                projection_only=True, k_pipeline_stages=stages, a_ping_pong=a_ping_pong,
                compute_squares=compute_squares, accumulate_squares=accumulate_squares, rows=rows, mad_k_override=mad_k, separate_nz=separate_nz, block_nz=block_nz, weight_nk=weight_nk, l0_ping_pong=l0_ping_pong,
            )

            if args.dump_projection_code is not None:
                source = kernel.get_kernel_source()
                if not source:
                    raise RuntimeError(f"TileLang returned no device source for {stages} pipeline stages")
                suffix = "_a_ping_pong" if a_ping_pong else "_deferred" if accumulate_squares else ""
                if l0_ping_pong:
                    suffix += "_l0_ping_pong"
                path = args.dump_projection_code / f"projection_k{chunk}_stages{stages}{suffix}.cpp"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
                sources[stages, a_ping_pong] = (path, source)
                print(f"Generated device source: {path}", flush=True)

            weight = phi
            if weight_nk:
                weight = phi.new_zeros((32, width))
                weight[:24].copy_(phi.transpose(0, 1))

            def run():
                kernel(x, weight, bias, post, residual, layer, projection)

            run()
            try:
                torch.npu.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Projection launch failed for N={batch}, H={hidden}, K={chunk}, "
                    f"stages={stages}, block_nz={block_nz}"
                ) from exc
            torch.testing.assert_close(projection[:, :24].cpu(), expected_projection.cpu(), atol=0.01, rtol=0.01)
            if compute_squares:
                torch.testing.assert_close(
                    projection[:, 24].cpu(), expected_squares.cpu(), atol=0.01,
                    rtol=0.001 if args.deferred_buffer_sweep else 0.01,
                )
            return run

        if args.projection_mk_compare:
            baseline = prepare(512, 2, accumulate_squares=True)
            split_k = 1536
            shards = (width + split_k - 1) // split_k
            partials = bias.new_empty((shards * batch, 32))
            kernel = _compile(
                "project_cube_m_only", **(config | {"chunk": 128}),
                rows=256, mad_k_override=32, split_k=split_k, projection_only=True,
                accumulate_squares=True, k_pipeline_stages=2,
            )
            partial_view = partials.view(shards, batch, 32)
            reduced = bias.new_empty((batch, 32))

            def run_partials():
                kernel(x, phi, bias, post, residual, layer, partials)

            def reduce_partials():
                torch.sum(partial_view, dim=0, out=reduced)

            def run_mk():
                run_partials()
                reduce_partials()

            run_mk()
            for shard in range(shards):
                start = shard * split_k
                stop = min(start + split_k, width)
                torch.testing.assert_close(
                    partial_view[shard, :, :24], flat[:, start:stop] @ phi[start:stop], atol=0.01, rtol=0.01,
                )
                torch.testing.assert_close(
                    partial_view[shard, :, 24], flat[:, start:stop].square().sum(-1), atol=0.01, rtol=0.001,
                )
            torch.testing.assert_close(reduced[:, :24], expected_projection, atol=0.01, rtol=0.01)
            torch.testing.assert_close(reduced[:, 24], expected_squares, atol=0.01, rtol=0.001)
            variants = (
                ("projection_m16_k512", baseline),
                ("projection_mk_partials", run_partials),
                ("projection_mk_with_torch_reduce", run_mk),
            )
        elif args.projection_m_tile_sweep:
            variants = (
                ("projection_m16_k512", prepare(512, 2, accumulate_squares=True)),
                ("projection_m256_k128_mad64", prepare(128, 2, accumulate_squares=True, rows=256)),
                ("projection_m256_k128_mad32", prepare(128, 2, accumulate_squares=True, rows=256, mad_k=32)),
            )
        elif args.profile_projection:
            if args.compare_l0_ping_pong:
                variants = (
                    ("projection_l0_single", prepare(512, 2, accumulate_squares=True)),
                    ("projection_l0_ping_pong", prepare(512, 2, accumulate_squares=True, l0_ping_pong=True)),
                )
            elif args.compare_weight_layout:
                variants = (
                    ("projection_weight_kn", prepare(512, 2, accumulate_squares=True)),
                    ("projection_weight_nk", prepare(512, 2, accumulate_squares=True, weight_nk=True)),
                )
            elif args.compare_block_nz:
                variants = (
                    ("projection_fused_nz", prepare(512, 2, accumulate_squares=True)),
                    ("projection_block_nz", prepare(512, 2, accumulate_squares=True, separate_nz=True, block_nz=True)),
                )
            elif args.compare_separate_nz:
                variants = (
                    ("projection_fused_nz", prepare(512, 2, accumulate_squares=True)),
                    ("projection_separate_nz", prepare(512, 2, accumulate_squares=True, separate_nz=True)),
                )
            elif args.deferred_buffer_sweep:
                variants = tuple(
                    (f"projection_k512_deferred_stages{stages}", prepare(512, stages, accumulate_squares=True))
                    for stages in (1, 2)
                )
            else:
                variants = (("projection_k512_stages1", prepare(512, 1)),)
            if args.accumulate_squares:
                variants += (("projection_k512_deferred_squares", prepare(512, 1, accumulate_squares=True)),)
            if args.compare_squares:
                variants += (("projection_k512_no_squares", prepare(512, 1, compute_squares=False)),)
        elif args.projection_buffer_sweep:
            variants = tuple((f"projection_k512_stages{stages}", prepare(512, stages)) for stages in (1, 2))
            if args.projection_a_ping_pong:
                variants += (("projection_k512_stages2_a_ping_pong", prepare(512, 2, True)),)
            if sources:
                first_path, first = sources[2, False] if args.projection_a_ping_pong else sources[1, False]
                second_path, second = sources[2, args.projection_a_ping_pong]
                diff = "".join(unified_diff(
                    first.splitlines(keepends=True), second.splitlines(keepends=True),
                    fromfile=first_path.name, tofile=second_path.name,
                ))
                diff_name = "a-ping-pong.diff" if args.projection_a_ping_pong else "pipeline.diff"
                diff_path = args.dump_projection_code / diff_name
                diff_path.write_text(diff, encoding="utf-8")
                print(f"Generated device sources {'differ' if diff else 'are identical'}. Diff: {diff_path}", flush=True)
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

        if args.batch_coefficients:
            expected_weights = weights[:, :20].clone()
            batched = _compile("project_cube_m_only", **config, coefficients_only=True, batch_coefficients=True)

            def run_batched():
                batched(x, phi, bias, post, residual, layer, weights)

            run_batched()
            torch.testing.assert_close(weights[:, :20], expected_weights, atol=0.001, rtol=0.001)
            torch.testing.assert_close(post, expected[2], atol=0.01, rtol=0.01)
            variants = (("coefficients_only", run_ablated), ("batched_coefficients", run_batched))
        else:
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
            route = "cube_m_only"
            if args.ablate_deferred_s2:
                route = "cube_m_only_k512_deferred_s2"
            elif args.ablate_k512 or args.projection_buffer_sweep or args.profile_projection:
                route = "cube_m_only_k512"
            writer.writerow((batch, hidden, route, name, f"{duration:.3f}", count, 1, 0.02))
    for name, values in samples.items():
        print(f"N={batch} H={hidden} {name}: median {median(values):.3f} us, "
              f"range {min(values):.3f}–{max(values):.3f} us", flush=True)
    if args.profile_projection:
        from torch_npu import profiler

        for name, run in variants:
            profile_projection(run, torch, profiler, args.profile_projection / name,
                               args.warmup, max(8, count))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--simd-reduce", action="store_true", help="Compare baseline and SIMD partial reduction")
    parser.add_argument("--ablate-m-only", action="store_true", help="Compare full Pre, coefficients, and projection at N=1024 and 4096")
    parser.add_argument("--projection-a-ping-pong", action="store_true", help="Add an explicit two-slot activation L1 variant")
    parser.add_argument("--dump-projection-code", type=Path, help="Save the device sources and diff for the buffer sweep")
    parser.add_argument("--projection-buffer-sweep", action="store_true", help="Compare one and two K pipeline stages at N=1024, K window 512")
    parser.add_argument("--projection-mk-compare", action="store_true", help="Compare vLLM-style 4x7 M/K projection at N=1024")
    parser.add_argument("--projection-m-tile-sweep", action="store_true", help="Compare M16/K512 and M256/K128 at N=4096")
    parser.add_argument("--projection-k-sweep", action="store_true", help="Compare projection-only K windows at N=1024")
    parser.add_argument("--batch-coefficients", action="store_true", help="Compare per-token and batched SIMT coefficient generation")
    parser.add_argument("--ablate-deferred-s2", action="store_true", help="Ablate the two-stage deferred K512 Pre")
    parser.add_argument("--ablate-k512", action="store_true", help="Use K512 for the M-only Pre ablation")
    parser.add_argument("--accumulate-squares", action="store_true", help="Compare per-window and deferred square reductions")
    parser.add_argument("--compare-squares", action="store_true", help="Also profile projection with squared sum disabled")
    parser.add_argument("--deferred-buffer-sweep", action="store_true", help="Profile one and two K stages with deferred square reduction")
    parser.add_argument("--compare-l0-ping-pong", action="store_true", help="Compare one and two L0 operand slots")
    parser.add_argument("--compare-weight-layout", action="store_true", help="Compare KxN and prepacked NxK projection weights")
    parser.add_argument("--compare-block-nz", action="store_true", help="Compare fused NZ and vLLM-style block gather")
    parser.add_argument("--compare-separate-nz", action="store_true", help="Compare fused and separate AIV NZ conversion")
    parser.add_argument("--summarize-post-cache", type=Path, help="Read existing Post cache profiling CSVs without an NPU")
    parser.add_argument("--compare-ub-residual", action="store_true", help="Profile the baseline and UB-resident residual Pre in both contexts")
    parser.add_argument("--compare-pre-input-reuse", action="store_true", help="Profile independent Pre while alternating two equal residual buffers")
    parser.add_argument("--compare-post-cache", action="store_true", help="Compare Post L2 store policies in the Pre context profile")
    parser.add_argument("--profile-pre-context", type=Path, help="Compare isolated and paired Pre pipeline utilization at N=4096")
    parser.add_argument("--profile-projection", type=Path, help="Capture K512 projection pipe utilization at N=1024")
    parser.add_argument("--batch", type=int, choices=[batch for batch, _ in CASES])
    parser.add_argument("--tileops-spec", action="store_true", help="Use 1 Sinkhorn iteration and epsilon 0.02")
    parser.add_argument("--launches-per-event", type=int, default=1,
                        help="Replay each prepared stage this many times per timing window")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-stage-results.csv")
    args = parser.parse_args()
    if args.summarize_post_cache:
        try:
            summarize_post_cache(args.summarize_post_cache)
        except (OSError, ValueError, KeyError, ArithmeticError) as exc:
            parser.exit(1, f"Could not summarize Post cache profile: {exc}\n")
        return
    if args.compare_ub_residual and (
        not args.profile_pre_context or args.compare_pre_input_reuse or args.compare_post_cache
    ):
        parser.error("--compare-ub-residual requires --profile-pre-context without other context comparisons")
    if args.compare_pre_input_reuse and (not args.profile_pre_context or args.compare_post_cache):
        parser.error("--compare-pre-input-reuse requires --profile-pre-context without --compare-post-cache")
    if args.compare_post_cache and not args.profile_pre_context:
        parser.error("--compare-post-cache requires --profile-pre-context")
    if args.profile_pre_context and (args.batch is not None or args.ablate_m_only or args.simd_reduce):
        parser.error("--profile-pre-context selects its own N=4096 workload")
    if args.accumulate_squares and not args.profile_projection:
        parser.error("--accumulate-squares requires --profile-projection")
    if args.compare_squares and not args.profile_projection:
        parser.error("--compare-squares requires --profile-projection")
    if args.compare_l0_ping_pong and (
        not args.profile_projection or args.compare_weight_layout or args.compare_block_nz
        or args.compare_separate_nz or args.accumulate_squares or args.compare_squares or args.deferred_buffer_sweep
    ):
        parser.error("--compare-l0-ping-pong requires --profile-projection without other projection comparisons")
    if args.compare_weight_layout and (
        not args.profile_projection or args.compare_block_nz or args.compare_separate_nz
        or args.accumulate_squares or args.compare_squares or args.deferred_buffer_sweep
    ):
        parser.error("--compare-weight-layout requires --profile-projection without other projection comparisons")
    if args.compare_block_nz and args.compare_separate_nz:
        parser.error("--compare-block-nz cannot be combined with --compare-separate-nz")
    if (args.compare_separate_nz or args.compare_block_nz) and (
        not args.profile_projection or args.accumulate_squares or args.compare_squares or args.deferred_buffer_sweep
    ):
        parser.error("NZ comparisons require --profile-projection without other projection comparisons")
    if args.profile_projection and (
        not args.ablate_m_only or args.projection_k_sweep or args.projection_buffer_sweep
    ):
        parser.error("--profile-projection requires --ablate-m-only without a projection sweep")
    if args.projection_a_ping_pong and not args.projection_buffer_sweep:
        parser.error("--projection-a-ping-pong requires --projection-buffer-sweep")
    if args.dump_projection_code is not None and not (args.projection_buffer_sweep or args.deferred_buffer_sweep or args.compare_l0_ping_pong):
        parser.error("--dump-projection-code requires a buffer comparison")
    if args.deferred_buffer_sweep and (
        not args.profile_projection or args.accumulate_squares or args.compare_squares
    ):
        parser.error("--deferred-buffer-sweep requires --profile-projection without other square comparisons")
    if args.projection_buffer_sweep and (not args.ablate_m_only or args.projection_k_sweep):
        parser.error("--projection-buffer-sweep requires --ablate-m-only without --projection-k-sweep")
    if args.batch_coefficients and not args.ablate_deferred_s2:
        parser.error("--batch-coefficients requires --ablate-deferred-s2")
    if args.ablate_deferred_s2 and (
        not args.ablate_k512 or args.profile_projection or args.projection_buffer_sweep or args.projection_k_sweep
    ):
        parser.error("--ablate-deferred-s2 requires --ablate-k512 without a projection profile or sweep")
    if args.ablate_k512 and (not args.ablate_m_only or args.projection_k_sweep):
        parser.error("--ablate-k512 requires --ablate-m-only without --projection-k-sweep")
    if args.projection_mk_compare and (
        not args.ablate_m_only or args.projection_k_sweep or args.projection_buffer_sweep
        or args.profile_projection or args.ablate_k512 or args.projection_m_tile_sweep
    ):
        parser.error("--projection-mk-compare requires --ablate-m-only without other ablation options")
    if args.projection_m_tile_sweep and (
        not args.ablate_m_only or args.projection_k_sweep or args.projection_buffer_sweep
        or args.profile_projection or args.ablate_k512
    ):
        parser.error("--projection-m-tile-sweep requires --ablate-m-only without other ablation options")
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
        if args.projection_k_sweep or args.projection_buffer_sweep or args.profile_projection or args.projection_mk_compare:
            cases = cases[:1]
        if args.projection_m_tile_sweep:
            cases = cases[-1:]
        if args.profile_pre_context:
            cases = ((4096, 2560),)
        for batch, hidden in cases:
            width = 4 * hidden
            x = torch.randn(batch, width, generator=rng).to(device, torch.bfloat16)
            phi = (torch.randn(width, 24, generator=rng) / math.sqrt(width)).to(device)
            bias = torch.randn(24, generator=rng).to(device)
            layer_out = torch.randn(batch, hidden, generator=rng).to(device, torch.bfloat16)
            if args.batch is not None and batch != args.batch:
                continue
            iters, eps = (1, 0.02) if args.tileops_spec or args.ablate_m_only or args.profile_pre_context else (20, 1e-6)
            inputs = (phi, x, bias, 1.0, 1.0, 1.0, iters, eps)
            expected = None if args.projection_k_sweep or args.projection_buffer_sweep or args.profile_projection or args.projection_m_tile_sweep or args.projection_mk_compare else pre_reference(*inputs)
            if args.profile_pre_context:
                profile_pre_context(writer, torch, args, inputs, layer_out, expected, batch, hidden)
                output.flush()
                continue
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
