"""Retest Cube pair latency with all execution orders on one NPU."""
import argparse
import csv
from itertools import permutations
from statistics import median

from bench_mhc import CASES, time_us

ROUTES = ("cube", "cube_mk", "cube_full")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--post-streams", action="store_true", help="Time Post only with 1, 2, or 4 streams per task")
    mode.add_argument("--post-pipeline", action="store_true", help="Compare default, capped, and triple-buffered Post")
    mode.add_argument("--grouped-k", action="store_true", help="Compare 1, 2, and 4 K blocks per Cube group")
    mode.add_argument("--m-only", action="store_true", help="Compare cube_full with M-only Pre")
    mode.add_argument("--split-k", action="store_true", help="Compare cube_full with one split-K job per core")
    mode.add_argument("--tail", action="store_true", help="Compare cube_full with valid-row preparation")
    mode.add_argument("--simd", action="store_true", help="Compare cube_full with SIMD Sinkhorn only")
    parser.add_argument("--debug-m-only", action="store_true", help="Print M-only projection and squared sum before timing")
    parser.add_argument("--pre-only", action="store_true", help="Time Pre without Post")
    parser.add_argument("--post-cores", type=int, choices=range(1, 57), metavar="1..56", default=8)
    parser.add_argument("--batch", type=int, choices=tuple(batch for batch, _ in CASES), default=2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cycles", type=int, default=2, help="Repeats of all route orders")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-b2-orders.csv")
    args = parser.parse_args()
    if min(args.cycles, args.warmup, args.repeat) < 1:
        parser.error("cycles, warmup and repeat must be positive")

    if args.debug_m_only and not args.m_only:
        parser.error("--debug-m-only requires --m-only")
    if args.pre_only and not (args.m_only or args.grouped_k):
        parser.error("--pre-only requires --m-only or --grouped-k")

    import torch
    import torch_npu  # noqa: F401

    from fgw_mhc import mhc_post, mhc_pre
    from fgw_mhc.reference import post_reference, pre_reference

    device = torch.device(args.device)
    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(42)
    # Advance the generator exactly as the earlier stages benchmark does.
    for batch, hidden in CASES:
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
    inputs = (phi, x, bias, 1.0, 1.0, 1.0, *((1, 0.02) if args.m_only or args.grouped_k else (20, 1e-6)))

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

    if args.post_streams:
        weights = torch.randn(batch, 4, generator=rng).to(device)
        post_inputs = (layer, weights, x)

    def run(route):
        if args.post_streams:
            return mhc_post(*post_inputs, streams_per_task=int(route))
        projection = "cube_full" if args.post_pipeline else route
        residual, layer_in, post = mhc_pre(*inputs, projection=projection)
        if args.pre_only:
            return residual, layer_in, post
        options = {}
        if args.post_pipeline and route != "post_default":
            options = {"num_vec_cores": args.post_cores, "pipeline": route == "post_triple"}
        return residual, layer_in, post, mhc_post(layer, post, residual, **options)

    if args.post_streams:
        expected = post_reference(*post_inputs)
    else:
        residual, layer_in, post = pre_reference(*inputs)
        expected = (residual, layer_in, post)
        if not args.pre_only:
            expected += (post_reference(layer, post, residual),)
    routes = ROUTES
    if args.post_streams:
        routes = ("1", "2", "4")
    elif args.post_pipeline:
        routes = ("post_default", "post_capped", "post_triple")
    elif args.simd:
        routes = ("cube_full", "cube_full_simd")
    elif args.grouped_k:
        routes = ("cube_full", "cube_full_g2", "cube_full_g4")
    elif args.m_only:
        routes = ("cube_full", "cube_m_only")
    elif args.split_k:
        routes = ("cube_full", "cube_full_k")
    elif args.tail:
        routes = ("cube_full", "cube_full_tail")
    for route in routes:
        actual = run(route)
        names = ("post",) if args.post_streams else ("x_res", "x_layer", "h_post", "pair")
        for name, value, reference in zip(
            names, (actual,) if args.post_streams else actual,
            (expected,) if args.post_streams else expected,
        ):
            try:
                torch.testing.assert_close(value, reference, atol=0.01, rtol=0.01)
            except AssertionError as exc:
                raise AssertionError(f"{route}, B={batch}, H={hidden}, {name}: {exc}") from exc
    torch.npu.synchronize()

    medians = {route: [] for route in routes}
    positions = {route: [[] for _ in routes] for route in routes}
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        variant, metric = ("streams_per_task", "post_us") if args.post_streams else ("projection", "pair_us")
        if args.pre_only:
            metric = "pre_us"
        writer.writerow(("batch", "hidden", "cycle", "order", "position", variant, metric))
        for cycle in range(1, args.cycles + 1):
            for order in permutations(routes):
                order_label = "/".join(order)
                for position, route in enumerate(order, start=1):
                    latency = time_us(lambda: run(route), torch, args.warmup, args.repeat)
                    medians[route].append(latency)
                    positions[route][position - 1].append(latency)
                    writer.writerow((batch, hidden, cycle, order_label, position, route, f"{latency:.3f}"))
                    output.flush()

    for route, timings in medians.items():
        by_position = ", ".join(f"{median(values):.3f}" for values in positions[route])
        print(f"{route}: median {median(timings):.3f} us, range {min(timings):.3f}–{max(timings):.3f} us; "
              f"position medians [{by_position}] us")


if __name__ == "__main__":
    main()
