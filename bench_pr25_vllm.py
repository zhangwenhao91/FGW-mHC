"""Benchmark the shifted mHC in TileLang Ascend PR #25 against vLLM AscendC."""
import argparse
import csv
from math import isfinite, sqrt
from pathlib import Path
import subprocess
import sys

from bench_mhc import time_us
from bench_shifted_mhc import pre_ref
from bench_vllm_ascend import prepare_vllm_opp

PR_HEAD = "053ddb470aefcbc6972d994739983734eeb4eaf8"
SETTINGS = ("stage", "tokens", "hidden", "warmup", "repeat", "calls_per_event")


def merge(pr_path, vllm_path, output):
    tables = []
    for path, backend in ((pr_path, "pr25"), (vllm_path, "vllm")):
        table = {}
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = tuple(row[field] for field in SETTINGS)
                latency = float(row["latency_us"])
                if row["backend"] != backend or key in table or not isfinite(latency) or latency <= 0:
                    raise ValueError(f"{path}: invalid or duplicate timing {row}")
                table[key] = latency
        tables.append(table)
    pr, vllm = tables
    if not pr or pr.keys() != vllm.keys():
        raise ValueError("PR #25 and vLLM timing settings and cases must match")
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow((*SETTINGS, "pr25_us", "vllm_ascendc_us", "pr25_speedup"))
        for key, latency in pr.items():
            writer.writerow((*key, f"{latency:.3f}", f"{vllm[key]:.3f}", f"{vllm[key] / latency:.3f}"))


def load_ops(backend, pr_repo, torch):
    if backend == "pr25":
        head = subprocess.check_output(["git", "-C", str(pr_repo), "rev-parse", "HEAD"], text=True).strip()
        if head != PR_HEAD:
            raise ValueError(f"{pr_repo}: need PR #25 commit {PR_HEAD}, found {head}")
        sys.path.insert(0, str(pr_repo.resolve() / "examples" / "ascend"))
        from mhc.mhc_pre import mhc_pre
        from mhc.mhc_post import mhc_post
        return mhc_pre, mhc_post

    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    native_pre = torch.ops._C_ascend.npu_hc_pre_v3
    native_post = torch.ops._C_ascend.npu_hc_post

    def pre(x, previous, phi, scale, bias):
        layer, post, comb, next_pre = native_pre(
            x.unsqueeze(0), phi, scale, bias, previous.unsqueeze(0),
            hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6,
        )
        return (layer.view(x.shape[0], x.shape[-1]), next_pre.view(-1, 4, 1),
                post.view(-1, 4, 1), comb.view(-1, 4, 4))

    def post(layer, x, post_mix, comb_mix):
        return native_post(layer.unsqueeze(0), x.unsqueeze(0), post_mix.view(1, -1, 4),
                           comb_mix.unsqueeze(0)).squeeze(0)

    return pre, post


def benchmark_case(args, torch, device, pre, post, tokens, hidden):
    rng = torch.Generator().manual_seed(42 + tokens + hidden)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, generator=rng)

    x = rand(tokens, 4, hidden, dtype=torch.bfloat16)
    layer_out = rand(tokens, hidden, dtype=torch.bfloat16)
    previous = rand(tokens, 4).sigmoid().add_(1e-6)
    phi = rand(24, 4 * hidden).div_(sqrt(4 * hidden))
    scale = rand(3).mul_(0.1).add_(1)
    bias = rand(24).mul_(0.1)
    pre_inputs = tuple(t.to(device) for t in (x, previous, phi, scale, bias))
    x_dev = pre_inputs[0]
    layer_dev = layer_out.to(device)

    def run_pre():
        return pre(*pre_inputs)

    actual_pre = run_pre()
    # Bound the FP32 reference workspace while checking every token.
    ref_post = torch.empty((tokens, 4, 1))
    ref_comb = torch.empty((tokens, 4, 4))
    for start in range(0, tokens, 128):
        section = slice(start, start + 128)
        flat = x[section].float().flatten(1)
        expected = pre_ref(x[section], (flat @ phi.T).unsqueeze(0),
                           flat.square().sum(-1).unsqueeze(0), scale, bias, previous[section])
        for index, (actual, reference) in enumerate(zip(actual_pre, expected, strict=True)):
            torch.testing.assert_close(actual[section].cpu(), reference,
                                       atol=0.03125 if index == 0 else 0.005, rtol=0.01)
        ref_post[section], ref_comb[section] = expected[2:]
    del actual_pre
    post_inputs = ref_post.to(device), ref_comb.to(device)

    def run_post():
        return post(layer_dev, x_dev, *post_inputs)

    def run_pair():
        layer, next_pre, post_mix, comb = run_pre()
        return layer, next_pre, post(layer_dev, x_dev, post_mix, comb)

    for op, pair in ((run_post, False), (run_pair, True)):
        result = op()
        actual = result[-1] if pair else result
        for start in range(0, tokens, 128):
            section = slice(start, start + 128)
            expected = (layer_out[section].float().unsqueeze(1) * ref_post[section]
                        + torch.bmm(ref_comb[section].transpose(1, 2), x[section].float())).bfloat16()
            torch.testing.assert_close(actual[section].cpu(), expected, atol=0.03125, rtol=0.01)
        del actual, result

    for stage, op in (("pre", run_pre), ("post", run_post), ("pair", run_pair)):
        def submit():
            for _ in range(args.calls_per_event):
                result = op()
            return result

        elapsed = time_us(submit, torch, args.warmup, args.repeat) / args.calls_per_event
        print(f"{args.backend} {stage} N={tokens} H={hidden}: {elapsed:.3f} us", flush=True)
        yield args.backend, stage, tokens, hidden, args.warmup, args.repeat, args.calls_per_event, f"{elapsed:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backend", choices=("pr25", "vllm"))
    mode.add_argument("--merge", type=Path, nargs=2)
    parser.add_argument("--pr-repo", type=Path, default=Path("../tilelang-asc-pr25"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 128, 1024, 4096])
    parser.add_argument("--hidden", type=int, nargs="+", choices=(4096, 5120, 7168), default=[4096, 5120, 7168])
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--calls-per-event", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.merge:
        merge(*args.merge, args.output)
        return
    if min(*args.tokens, args.calls_per_event, args.warmup, args.repeat) < 1:
        parser.error("tokens and timing parameters must be positive")
    if args.backend == "vllm":
        prepare_vllm_opp()
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)
    pre, post = load_ops(args.backend, args.pr_repo, torch)
    with args.output.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("backend", *SETTINGS, "latency_us"))
        with torch.inference_mode():
            for hidden in dict.fromkeys(args.hidden):
                for tokens in dict.fromkeys(args.tokens):
                    try:
                        for row in benchmark_case(args, torch, device, pre, post, tokens, hidden):
                            writer.writerow(row)
                            out.flush()
                    except Exception as exc:
                        raise RuntimeError(f"{args.backend} N={tokens} H={hidden} failed") from exc


if __name__ == "__main__":
    main()
