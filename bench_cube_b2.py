"""Retest Cube pair latency with all execution orders on one NPU."""
import argparse
import csv
from itertools import permutations
from pathlib import Path
from statistics import median

from bench_mhc import CASES, time_us

ROUTES = ("cube", "cube_mk", "cube_full")


def verify_windowed_cache(torch, inputs, layer, source_path, route):
    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    phi, residual, bias, *params = inputs
    hidden = layer.shape[1]
    cases = (inputs, (phi, -residual, bias, *params))
    references = []
    for case in cases:
        x_res, x_layer, h_post = pre_reference(*case)
        references.append((x_res, x_layer, h_post, post_reference(layer, h_post, x_res)))

    def dump_launch(name, kernel, *tensors):
        source = kernel.get_kernel_source()
        if not source:
            raise RuntimeError("TileLang returned no generated device source")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        kernel(*tensors)

    mhc_pre(*inputs, projection=route, _launch=dump_launch)
    torch.npu.synchronize()
    print(f"Generated device source: {source_path}", flush=True)

    names = ("x_res", "x_layer", "h_post", "pair")
    launches = 8
    for burst in range(4):
        pending = []
        for index in range(launches):
            case_id = (burst + index) % len(cases)
            x_res, x_layer, h_post = mhc_pre(*cases[case_id], projection=route)
            out = mhc_post(layer, h_post, x_res, post_tile=hidden, pipeline=True, pipeline_depth=2)
            pending.append((case_id, (x_res, x_layer, h_post, out)))
        torch.npu.synchronize()
        for index, (case_id, actual) in enumerate(pending):
            for name, value, reference in zip(names, actual, references[case_id]):
                try:
                    torch.testing.assert_close(value, reference, atol=0.02 if name == "pair" else 0.01, rtol=0.01)
                except AssertionError as exc:
                    raise AssertionError(f"burst={burst + 1}, pair={index + 1}, input={case_id}, {name}: {exc}") from exc
        pending.clear()
        print(f"Alternating-input burst {burst + 1}: {launches} pairs passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--large-post-tiles", action="store_true", help="Compare full-H tiles and double buffering for large-token Post")
    mode.add_argument("--b4-post-tiles", action="store_true", help="Compare Post task layouts in the B=4 TileOPs pair")
    mode.add_argument("--post-streams", action="store_true", help="Time Post only with 1, 2, or 4 streams per task")
    mode.add_argument("--post-pipeline", action="store_true", help="Compare default, capped, and triple-buffered Post")
    mode.add_argument("--pre-h-tile", action="store_true", help="Compare cube_full Pre output H tiles at B=4")
    mode.add_argument("--grouped-k", action="store_true", help="Compare 1, 2, and 4 K blocks per Cube group")
    mode.add_argument("--m-only-k-window", action="store_true", help="Compare full M-only Pre at K windows 256 and 512")
    mode.add_argument("--m-only", action="store_true", help="Compare cube_full with M-only Pre")
    mode.add_argument("--split-k", action="store_true", help="Compare cube_full with one split-K job per core")
    mode.add_argument("--tail", action="store_true", help="Compare cube_full with valid-row preparation")
    mode.add_argument("--simd-reduce", action="store_true", help="Compare SIMD partial reduction at 32 pairs per event")
    mode.add_argument("--simd", action="store_true", help="Compare cube_full with SIMD Sinkhorn only")
    parser.add_argument("--debug-m-only", action="store_true", help="Print M-only projection and squared sum before timing")
    parser.add_argument("--double-window-stage", action="store_true", help="Compare single- and double-slot guarded window staging")
    parser.add_argument("--guarded-window-cache", action="store_true", help="Compare windowed caching with per-window MTE2 reuse guards")
    parser.add_argument("--profile-simd-pairs", type=Path, help="Capture interleaved SIMD reduction pairs after timing")
    parser.add_argument("--verify-windowed-cache", type=Path, help="Dump device source and stress-test the windowed cache path")
    parser.add_argument("--cache-pre-residual", action="store_true", help="Compare retaining Pre input residual in UB")
    parser.add_argument("--l0-ping-pong", action="store_true", help="Compare L0 operand buffering in batched K512 Pre")
    parser.add_argument("--batch-coefficients", action="store_true", help="Compare per-token and batched coefficients in full K512 Pre")
    parser.add_argument("--deferred-pipeline", action="store_true", help="Compare one and two K stages in full deferred K512 Pre")
    parser.add_argument("--deferred-squares", action="store_true", help="Compare K512 with per-window and deferred square reductions")
    parser.add_argument("--post-pair", action="store_true", help="Time whole pairs in the large Post tile comparison")
    parser.add_argument("--pre-only", action="store_true", help="Time Pre without Post")
    parser.add_argument("--post-cores", type=int, choices=range(1, 57), metavar="1..56", default=8)
    parser.add_argument("--batch", type=int, choices=(*[batch for batch, _ in CASES], 1024, 4096), default=2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cycles", type=int, default=2, help="Repeats of all route orders")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-b2-orders.csv")
    args = parser.parse_args()
    if min(args.cycles, args.warmup, args.repeat) < 1:
        parser.error("cycles, warmup and repeat must be positive")

    if args.post_pair and (not args.large_post_tiles or args.pre_only):
        parser.error("--post-pair requires --large-post-tiles without --pre-only")
    if args.debug_m_only and not args.m_only:
        parser.error("--debug-m-only requires --m-only")
    if args.double_window_stage and not args.guarded_window_cache:
        parser.error("--double-window-stage requires --guarded-window-cache")
    if args.guarded_window_cache and not args.cache_pre_residual:
        parser.error("--guarded-window-cache requires --cache-pre-residual")
    if args.verify_windowed_cache and (not args.cache_pre_residual or args.pre_only):
        parser.error("--verify-windowed-cache requires --cache-pre-residual without --pre-only")
    if args.cache_pre_residual and (
        not args.m_only_k_window or args.l0_ping_pong or args.batch_coefficients
        or args.deferred_pipeline or args.deferred_squares
    ):
        parser.error("--cache-pre-residual requires --m-only-k-window without other comparisons")
    if args.l0_ping_pong and (
        not args.m_only_k_window or args.batch_coefficients or args.deferred_pipeline or args.deferred_squares
    ):
        parser.error("--l0-ping-pong requires --m-only-k-window without other comparisons")
    if args.batch_coefficients and (not args.m_only_k_window or args.deferred_pipeline or args.deferred_squares):
        parser.error("--batch-coefficients requires --m-only-k-window without other deferred comparisons")
    if args.deferred_pipeline and (not args.m_only_k_window or args.deferred_squares):
        parser.error("--deferred-pipeline requires --m-only-k-window without --deferred-squares")
    if args.deferred_squares and not args.m_only_k_window:
        parser.error("--deferred-squares requires --m-only-k-window")
    cases = ((1024, 2560), (4096, 2560)) if args.m_only_k_window or args.large_post_tiles else CASES
    if args.batch not in dict(cases):
        parser.error("--m-only-k-window and --large-post-tiles use batches 1024 and 4096, and other modes use 1, 2, or 4")
    if args.b4_post_tiles and args.batch != 4:
        parser.error("--b4-post-tiles requires --batch 4")
    if args.profile_simd_pairs and not args.simd_reduce:
        parser.error("--profile-simd-pairs requires --simd-reduce")
    if args.simd_reduce and args.batch != 4:
        parser.error("--simd-reduce requires --batch 4")
    if args.pre_h_tile and args.batch != 4:
        parser.error("--pre-h-tile requires --batch 4")
    if args.pre_only and not (args.m_only or args.grouped_k or args.pre_h_tile or args.m_only_k_window):
        parser.error("--pre-only requires --m-only, --m-only-k-window, --grouped-k, or --pre-h-tile")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(42)
    # Advance the generator exactly as the earlier stages benchmark does.
    for batch, hidden in cases:
        width = 4 * hidden
        x = torch.randn(batch, width, generator=rng)
        phi = torch.randn(width, 24, generator=rng) / width**0.5
        bias = torch.randn(24, generator=rng)
        layer = torch.randn(batch, hidden, generator=rng)
        if batch == args.batch:
            break
    x = x.to(device, torch.bfloat16)
    phi = phi.to(device)
    bias = bias.to(device)
    layer = layer.to(device, torch.bfloat16)
    sinkhorn = (1, 0.02) if args.m_only or args.grouped_k or args.pre_h_tile or args.m_only_k_window or args.large_post_tiles or args.simd_reduce or args.b4_post_tiles else (20, 1e-6)
    inputs = (phi, x, bias, 1.0, 1.0, 1.0, *sinkhorn)

    if args.verify_windowed_cache:
        route = "cube_m_only_k512_guarded_cache" if args.guarded_window_cache else "cube_m_only_k512_windowed_cache"
        if args.double_window_stage:
            route = "cube_m_only_k512_double_stage"
        verify_windowed_cache(torch, inputs, layer, args.verify_windowed_cache, route)
        return

    if args.debug_m_only:
        from math import gcd
        from fgw_mhc import _compile

        kernel = _compile(
            "project_cube_m_only", batch=batch, width=width, chunk=gcd(width, 256), tile=hidden,
            repeat=1, eps=0.02, alpha_pre=1.0, alpha_post=1.0, alpha_res=1.0, debug=True,
        )
        trace = bias.new_empty((batch, 32))
        kernel(x, phi, bias, bias.new_empty((batch, 4)),
               torch.empty_like(x).view(batch, 4, hidden), torch.empty_like(layer), trace)
        flat = x.float()
        expected_projection = flat @ phi
        expected_squares = flat.square().sum(-1)
        for name, actual, reference in (
            ("projection", trace[:, :24], expected_projection),
            ("squares", trace[:, 24], expected_squares),
        ):
            print(f"{name} actual={actual.tolist()} expected={reference.tolist()}", flush=True)

    if args.large_post_tiles:
        if not args.post_pair:
            residual, _, weights = mhc_pre(*inputs, projection="cube_m_only_k512_batched")
            post_inputs = (layer, weights, residual)
        post_options = {
            "default": {},
            "full_h": {"post_tile": hidden},
            "full_h_double": {"post_tile": hidden, "pipeline": True, "pipeline_depth": 2},
        }
    elif args.post_streams:
        weights = torch.randn(batch, 4, generator=rng).to(device)
        post_inputs = (layer, weights, x)

    if args.b4_post_tiles:
        post_options = {
            "default": {},
            "two_streams_h512": {"streams_per_task": 2, "post_tile": 512},
            "four_streams_h320": {"streams_per_task": 4, "post_tile": 320},
        }

    post_only = args.post_streams or (args.large_post_tiles and not args.post_pair)

    def run(route):
        if args.post_pair:
            residual, layer_in, weights = mhc_pre(*inputs, projection="cube_m_only_k512_batched")
            return residual, layer_in, weights, mhc_post(layer, weights, residual, **post_options[route])
        if args.large_post_tiles:
            return mhc_post(*post_inputs, **post_options[route])
        if args.post_streams:
            return mhc_post(*post_inputs, streams_per_task=int(route))
        projection = "cube_full" if args.post_pipeline or args.pre_h_tile or args.b4_post_tiles else route
        pre_tile = int(route) if args.pre_h_tile else None
        residual, layer_in, post = mhc_pre(*inputs, projection=projection, pre_tile=pre_tile)
        if args.pre_only:
            return residual, layer_in, post
        options = post_options[route] if args.b4_post_tiles else {}
        if args.cache_pre_residual:
            options = {"post_tile": hidden, "pipeline": True, "pipeline_depth": 2}
        if args.post_pipeline and route != "post_default":
            options = {"num_vec_cores": args.post_cores, "pipeline": route == "post_triple"}
        return residual, layer_in, post, mhc_post(layer, post, residual, **options)

    if post_only:
        expected = post_reference(*post_inputs)
    else:
        residual, layer_in, post = pre_reference(*inputs)
        expected = (residual, layer_in, post)
        if not args.pre_only:
            expected += (post_reference(layer, post, residual),)
    routes = ROUTES
    if args.large_post_tiles:
        routes = ("default", "full_h_double") if args.post_pair else tuple(post_options)
    elif args.b4_post_tiles:
        routes = tuple(post_options)
    elif args.post_streams:
        routes = ("1", "2", "4")
    elif args.post_pipeline:
        routes = ("post_default", "post_capped", "post_triple")
    elif args.simd_reduce:
        routes = ("cube_full", "cube_full_simd_reduce")
    elif args.simd:
        routes = ("cube_full", "cube_full_simd")
    elif args.pre_h_tile:
        routes = ("2560", "1280", "640")
    elif args.grouped_k:
        routes = ("cube_full", "cube_full_g2", "cube_full_g4")
    elif args.double_window_stage:
        routes = ("cube_m_only_k512_batched", "cube_m_only_k512_guarded_cache", "cube_m_only_k512_double_stage")
    elif args.guarded_window_cache:
        routes = ("cube_m_only_k512_batched", "cube_m_only_k512_windowed_cache", "cube_m_only_k512_guarded_cache")
    elif args.cache_pre_residual:
        routes = (
            "cube_m_only_k512_batched", "cube_m_only_k512_cached_input", "cube_m_only_k512_windowed_cache",
        )
    elif args.l0_ping_pong:
        routes = ("cube_m_only_k512_batched", "cube_m_only_k512_batched_l0")
    elif args.batch_coefficients:
        routes = ("cube_m_only_k512_deferred_s2", "cube_m_only_k512_batched")
    elif args.deferred_pipeline:
        routes = ("cube_m_only_k512_deferred", "cube_m_only_k512_deferred_s2")
    elif args.deferred_squares:
        routes = ("cube_m_only_k512", "cube_m_only_k512_deferred")
    elif args.m_only_k_window:
        routes = ("cube_m_only", "cube_m_only_k512")
    elif args.m_only:
        routes = ("cube_full", "cube_m_only")
    elif args.split_k:
        routes = ("cube_full", "cube_full_k")
    elif args.tail:
        routes = ("cube_full", "cube_full_tail")
    for route in routes:
        actual = run(route)
        names = ("post",) if post_only else ("x_res", "x_layer", "h_post", "pair")
        for name, value, reference in zip(
            names, (actual,) if post_only else actual,
            (expected,) if post_only else expected,
        ):
            try:
                atol = 0.02 if (args.m_only_k_window or args.post_pair) and name == "pair" else 0.01
                torch.testing.assert_close(value, reference, atol=atol, rtol=0.01)
            except AssertionError as exc:
                raise AssertionError(f"{route}, B={batch}, H={hidden}, {name}: {exc}") from exc
    torch.npu.synchronize()

    pairs_per_event = 32 if args.simd_reduce or args.b4_post_tiles else 8 if args.m_only_k_window or args.large_post_tiles else 1
    if pairs_per_event > 1:
        for _ in range(args.warmup):
            for route in routes:
                run(route)
        torch.npu.synchronize()

    medians = {route: [] for route in routes}
    positions = {route: [[] for _ in routes] for route in routes}
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        variant, metric = ("streams_per_task", "post_us") if args.post_streams else ("projection", "pair_us")
        if args.pre_h_tile:
            variant = "pre_h_tile"
        if args.large_post_tiles:
            variant, metric = "post_config", "pair_us" if args.post_pair else "post_us"
        if args.b4_post_tiles:
            variant = "post_config"
        if args.pre_only:
            metric = "pre_us"
        if pairs_per_event > 1:
            metric = f"{metric}_at_{pairs_per_event}_per_event" if args.m_only_k_window or args.large_post_tiles else "us_per_pair_at_32_pairs_per_event"
        writer.writerow(("batch", "hidden", "cycle", "order", "position", variant, metric))
        for cycle in range(1, args.cycles + 1):
            for order in permutations(routes):
                order_label = "/".join(order)
                for position, route in enumerate(order, start=1):
                    def submit():
                        for _ in range(pairs_per_event - 1):
                            run(route)
                        return run(route)

                    measure = submit if pairs_per_event > 1 else lambda: run(route)
                    latency = time_us(measure, torch, args.warmup, args.repeat) / pairs_per_event
                    medians[route].append(latency)
                    positions[route][position - 1].append(latency)
                    writer.writerow((batch, hidden, cycle, order_label, position, route, f"{latency:.3f}"))
                    output.flush()

    for route, timings in medians.items():
        by_position = ", ".join(f"{median(values):.3f}" for values in positions[route])
        print(f"{route}: median {median(timings):.3f} us, range {min(timings):.3f}–{max(timings):.3f} us; "
              f"position medians [{by_position}] us")

    if args.profile_simd_pairs:
        from bench_vllm_ascend import profile_pair, summarize_profile

        order = (routes + routes[::-1]) * 8
        directory = args.profile_simd_pairs
        profile_pair(lambda: run(order[0]), torch, torch_npu.profiler, directory, 0, len(order),
                     profile_step=lambda step: run(order[step]))
        summarize_profile(directory, "tilelang", len(order), routes=order)



if __name__ == "__main__":
    main()
