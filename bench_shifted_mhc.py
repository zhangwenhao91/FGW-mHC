"""Time standalone shifted mHC Pre and Post separately."""
import argparse
import csv
from pathlib import Path
from math import isfinite

from bench_mhc import time_us
from bench_vllm_ascend import prepare_vllm_opp

TILES = {4096: 4096, 5120: 2560, 7168: 3584}


def pre_ref(x, projection, squares, scale, bias, previous):
    import torch

    norm = (squares[0, :, None] / (4 * x.shape[-1]) + 1e-20).rsqrt()
    pre, post, comb = (projection[0] * norm).split((4, 4, 16), dim=-1)
    pre = (pre * scale[0] + bias[:4]).sigmoid().unsqueeze(-1) + 1e-6
    post = (2 * (post * scale[1] + bias[4:8]).sigmoid()).unsqueeze(-1)
    comb = (comb * scale[2] + bias[8:]).reshape(-1, 4, 4).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return (x.float() * previous.unsqueeze(-1)).sum(1).bfloat16(), pre, post, comb


def merge_results(fgw_path, vllm_path, output_path):
    tables = []
    for path, backend in ((fgw_path, "tilelang"), (vllm_path, "vllm")):
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if not rows or any(row["backend"] != backend for row in rows):
            raise ValueError(f"{path}: expected {backend} timings")
        table = {}
        for row in rows:
            key = tuple(row[field] for field in ("stage", "scope", "tokens", "hidden", "warmup", "repeat"))
            latency = float(row["latency_us"])
            if key in table or not isfinite(latency) or latency <= 0:
                raise ValueError(f"{path}: duplicate or invalid sample {row}")
            table[key] = latency
        tables.append(table)
    fgw, vllm = tables
    if not vllm or set(vllm) != set(fgw):
        raise ValueError("FGW and vLLM stages, scopes, shapes and timing settings must match")
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("stage", "scope", "tokens", "hidden", "warmup", "repeat",
                         "fgw_us", "vllm_ascendc_us", "speedup"))
        for key, elapsed in fgw.items():
            baseline = vllm[key]
            writer.writerow((*key, f"{elapsed:.3f}", f"{baseline:.3f}", f"{baseline / elapsed:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--backend", choices=("tilelang", "vllm"), default="tilelang")
    mode.add_argument("--merge", type=Path, nargs=2, metavar=("FGW_CSV", "VLLM_CSV"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 512, 2048, 8192])
    parser.add_argument("--hidden", type=int, nargs="+", choices=tuple(TILES), default=list(TILES))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("shifted-mhc.csv"))
    args = parser.parse_args()
    if args.merge:
        merge_results(*args.merge, args.output)
        return
    if args.backend == "vllm":
        prepare_vllm_opp()
    if min(args.warmup, args.repeat, *args.tokens) < 1:
        parser.error("tokens, warmup and repeat must be positive")
    import torch
    import torch_npu  # noqa: F401

    if args.backend == "tilelang":
        from fgw_mhc.shifted import shifted_post, shifted_pre
    else:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        vllm_pre = torch.ops._C_ascend.npu_hc_pre_v3
        vllm_post = torch.ops._C_ascend.npu_hc_post

    device = torch.device(args.device)
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(42)

    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("backend", "stage", "scope", "tokens", "hidden", "warmup", "repeat", "latency_us"))

        def measure(stage, op, reference):
            torch.testing.assert_close(op(), reference(), atol=0.01, rtol=0.01)
            latency = time_us(op, torch, args.warmup, args.repeat)
            if not isfinite(latency) or latency <= 0:
                raise RuntimeError(f"Invalid event time: {latency}")
            scope = "projection_included" if stage == "pre" else "full_residual_mix"
            writer.writerow((args.backend, stage, scope, tokens, hidden, args.warmup, args.repeat, f"{latency:.6f}"))
            output.flush()
            print(f"{args.backend} {stage} N={tokens} H={hidden}: {latency:.3f} us", flush=True)

        for hidden in args.hidden:
            if args.backend == "tilelang":
                pre_kernel = shifted_pre.compile(hidden=hidden)
                post_kernel = shifted_post.compile(hidden=hidden, h_blk=TILES[hidden])
            for tokens in args.tokens:
                def rand(*shape):
                    return torch.randn(shape, generator=generator).to(device)

                x = rand(tokens, 4, hidden).bfloat16()
                layer_out = rand(tokens, hidden).bfloat16()
                post_mix = rand(tokens, 4)
                # Binary-exact, asymmetric coefficients exercise the orientation on both stacks.
                comb_mix = rand(tokens, 4, 4).sign() * 0.25

                pre_rng = torch.Generator().manual_seed(42 + tokens + hidden)

                def pre_rand(*shape):
                    return torch.randn(shape, generator=pre_rng).to(device)

                hc_fn = pre_rand(24, 4 * hidden) / (4 * hidden)**0.5
                scale, bias = pre_rand(3), pre_rand(24)
                previous = pre_rand(tokens, 4).sigmoid()

                def project():
                    flat = x.float().flatten(1)
                    return (flat @ hc_fn.T).unsqueeze(0), flat.square().sum(-1).unsqueeze(0)

                def reference_pre():
                    return pre_ref(x, *project(), scale, bias, previous)

                if args.backend == "tilelang":
                    def run_pre():
                        projection, squares = project()
                        layer = x.new_empty((tokens, hidden))
                        next_pre = scale.new_empty((tokens, 4, 1))
                        post = torch.empty_like(next_pre)
                        comb = scale.new_empty((tokens, 4, 4))
                        pre_kernel(x, projection, squares, scale, bias, previous, layer, next_pre, post, comb)
                        return layer, next_pre, post, comb

                    def run_post():
                        out = torch.empty_like(x)
                        post_kernel(layer_out, x, post_mix, comb_mix, out)
                        return out
                else:
                    vllm_pre_inputs = (x.unsqueeze(0), hc_fn, scale, bias, previous.unsqueeze(0),
                                       4, 20, 1e-20, 1e-6)
                    vllm_post_inputs = tuple(t.unsqueeze(0) for t in (layer_out, x, post_mix, comb_mix))

                    def run_pre():
                        layer, post, comb, next_pre = vllm_pre(*vllm_pre_inputs)
                        return (layer.reshape(tokens, hidden), next_pre.reshape(tokens, 4, 1),
                                post.reshape(tokens, 4, 1), comb.reshape(tokens, 4, 4))

                    def run_post():
                        return vllm_post(*vllm_post_inputs).squeeze(0)

                measure("pre", run_pre, reference_pre)

                def post_ref():
                    return (layer_out.float().unsqueeze(1) * post_mix.unsqueeze(-1)
                            + torch.bmm(comb_mix.transpose(1, 2), x.float())).bfloat16()

                measure("post", run_post, post_ref)


if __name__ == "__main__":
    main()
