#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

card="${1:-4}"
candidate="${2:-../TileOPs-mhc-clean}"
baseline="${3:-../TileOPs-mhc-baseline}"
for repo in "$candidate" "$baseline"; do
    git -C "$repo" rev-parse --short HEAD
    test -f "$repo/src/tileops/ascend.py"
done

prefix="manifest-entrypoints-card$card-$(date +%Y%m%d-%H%M%S)-$$"
FGW_MHC_TILEOPS_REPO= FGW_MHC_MANIFEST_LOCAL_ONLY=0 FGW_MHC_MANIFEST_PROJECTION=tileops_tuned FGW_MHC_CPP_PRE_ALLOCATOR=0 \
    FGW_MHC_TIMING_PREFIX="$prefix-fgw" bash run_tileops_spec_vllm.sh "$card" validate-manifest
FGW_MHC_TILEOPS_REPO="$candidate" FGW_MHC_MANIFEST_LOCAL_ONLY=1 FGW_MHC_MANIFEST_PROJECTION=cube_full \
    FGW_MHC_TIMING_PREFIX="$prefix-candidate" bash run_tileops_spec_vllm.sh "$card" validate-manifest
FGW_MHC_TILEOPS_REPO="$baseline" FGW_MHC_MANIFEST_LOCAL_ONLY=1 FGW_MHC_MANIFEST_PROJECTION=cube_full \
    FGW_MHC_TIMING_PREFIX="$prefix-baseline" bash run_tileops_spec_vllm.sh "$card" validate-manifest

python - "$prefix" <<'PY'
import csv
from statistics import mean, median
import sys

prefix = sys.argv[1]
shapes = ((1, 1280), (2, 1920), (4, 2560))
counts = (1, 8, 32)
orders = ("forward", "reverse")
keys = [(b, h, n) for b, h in shapes for n in counts]

def load(path):
    with open(path, newline="") as source:
        rows = {(int(r["batch"]), int(r["hidden"]), int(r["pairs_per_event"]), r["order"]):
                float(r["pair_us"]) for r in csv.DictReader(source)}
    if set(rows) != {(*key, order) for key in keys for order in orders}:
        raise ValueError(f"Incomplete or unexpected workload timings: {path}")
    return {key: mean(rows[(*key, order)] for order in orders) for key in keys}

with open(f"{prefix}-results.csv", "w", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(("comparison", "batch", "hidden", "pairs_per_event", "candidate_us",
                     "baseline_us", "speedup_median", "speedup_min", "speedup_max", "wins", "rounds"))
    for label, our_tag, base_tag, base_backend in (
        ("fgw_vs_vllm", "fgw", "fgw", "vllm"),
        ("tileops_candidate_vs_original", "candidate", "baseline", "tilelang"),
    ):
        trials = [(load(f"{prefix}-{our_tag}-round{i}-tilelang.csv"),
                   load(f"{prefix}-{base_tag}-round{i}-{base_backend}.csv")) for i in range(1, 5)]
        for key in keys:
            ours = [a[key] for a, _ in trials]
            baselines = [b[key] for _, b in trials]
            ratios = [b / a for a, b in zip(ours, baselines, strict=True)]
            writer.writerow((label, *key, f"{median(ours):.3f}", f"{median(baselines):.3f}",
                             f"{median(ratios):.3f}", f"{min(ratios):.3f}", f"{max(ratios):.3f}",
                             sum(r > 1 for r in ratios), len(ratios)))
PY
cat "$prefix-results.csv"
