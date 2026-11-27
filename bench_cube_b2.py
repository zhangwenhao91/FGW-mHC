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
    mode.add_argument("--tail", action="store_true", help="Compare cube_full with valid-row preparation")
    mode.add_argument("--simd", action="store_true", help="Compare cube_full with SIMD Sinkhorn only")
    parser.add_argument("--batch", type=int, choices=tuple(batch for batch, _ in CASES), default=2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cycles", type=int, default=2, help="Repeats of all route orders")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", default="cube-b2-orders.csv")
    args = parser.parse_args()
    if min(args.cycles, args.warmup, args.repeat) < 1:
        parser.error("cycles, warmup and repeat must be positive")

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
    inputs = (phi, x, bias, 1.0, 1.0, 1.0, 20, 1e-6)

    def run(route):
        residual, layer_in, post = mhc_pre(*inputs, projection=route)
        return residual, layer_in, post, mhc_post(layer, post, residual)

    residual, layer_in, post = pre_reference(*inputs)
    expected = (residual, layer_in, post, post_reference(layer, post, residual))
    routes = ROUTES
    if args.simd:
        routes = ("cube_full", "cube_full_simd")
    elif args.tail:
        routes = ("cube_full", "cube_full_tail")
    for route in routes:
        torch.testing.assert_close(run(route), expected, atol=0.01, rtol=0.01)
    torch.npu.synchronize()

    medians = {route: [] for route in routes}
    positions = {route: [[] for _ in routes] for route in routes}
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("batch", "hidden", "cycle", "order", "position", "projection", "pair_us"))
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
