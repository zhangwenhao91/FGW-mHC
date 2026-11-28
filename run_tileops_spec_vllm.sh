#!/usr/bin/env bash
# Run both native stages and the full pair with the TileOPs MHC manifest settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

card="${1:-4}"
case "${2:-all}" in
    all) modes=(stages pair); projection=cube_full; label=tileops-spec ;;
    tuned-pair) modes=(pair); projection=tileops_tuned; label=tileops-spec-tuned ;;
    profile-b4) modes=(profile); projection=cube_full; label=tileops-spec ;;
    sweep-b4) modes=(sweep); projection=cube_full; label=tileops-spec ;;
    validate-b4) modes=(1 2 3 4); projection=cube_full; label="tileops-spec-validate-b4-card$card-$(date +%Y%m%d-%H%M%S)-$$" ;;
    allocator-ab-b4) modes=(sweep); projection=cube_full; label=tileops-spec-allocator-ab-b4 ;;
    allocate-b4) modes=(sweep); projection=cube_full; label=tileops-spec-allocate-b4 ;;
    direct-b4) modes=(sweep); projection=cube_full; label=tileops-spec-direct-b4 ;;
    compare-b4) modes=(sweep); projection=cube_full; label=tileops-spec-compare-b4 ;;
    large-pair) modes=(large-full large-m-only); projection=cube_full; label=tileops-spec ;;
    large-stages) modes=(large-stages); projection=cube_m_only; label=tileops-spec ;;
    large-k512-pair) modes=(large-k512-pair); projection=cube_m_only_k512; label=tileops-spec ;;
    large-k512-stages) modes=(large-k512-stages); projection=cube_m_only_k512; label=tileops-spec ;;
    large-k512-deferred-pair) modes=(large-k512-deferred-pair); projection=cube_m_only_k512_deferred; label=tileops-spec ;;
    large-k512-s2) modes=(large-k512-s2-pair large-k512-s2-stages); projection=cube_m_only_k512_deferred_s2; label=tileops-spec ;;
    large-k512-batched) modes=(large-k512-batched-pair large-k512-batched-stages); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    profile-n4096) modes=(large-k512-profile); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    tiling-large) modes=(tiling); label=tileops-spec ;;
    large-k512-post-double) modes=(large-k512-post-double); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    profile-n4096-post-double) modes=(large-k512-post-double-profile); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    *) echo "Usage: $0 [card] [all|tuned-pair|profile-b4|sweep-b4|compare-b4|direct-b4|allocate-b4|allocator-ab-b4|validate-b4|large-pair|large-stages|large-k512-pair|large-k512-stages|large-k512-deferred-pair|large-k512-s2|large-k512-batched|profile-n4096|tiling-large|large-k512-post-double|profile-n4096-post-double]" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$card"
devices=(--device /dev/davinci_manager)
for node in /dev/davinci[0-9]* /dev/hisi_hdc /dev/ummu /dev/uburma; do
    if [[ -c "$node" ]]; then
        devices+=(--device "$node")
    fi
done

validation_csvs=()
for mode in "${modes[@]}"; do
    if [[ "${2:-all}" == validate-b4 ]]; then
        round="$mode"
        mode=sweep
    fi
    stage_args=()
    if [[ "$mode" == stages || "$mode" == large-*-stages || "$mode" == large-stages ]]; then
        stage_args=(--split-stages)
    fi
    prefix="$label-$mode-card$card"
    if [[ "${2:-all}" == validate-b4 ]]; then
        prefix="$label-round$round"
        validation_csvs+=("$prefix-tilelang.csv" "$prefix-vllm.csv")
    fi
    common=(--tileops-spec --device npu:0 "${stage_args[@]}")
    if [[ "$mode" == sweep ]]; then
        common+=(--batch 4 --sweep-pairs 1 8 32)
    fi
    merge_args=()
    baseline="$prefix-vllm.csv"
    if [[ "$mode" == large-k512-* ]]; then
        common+=(--large-k512-cases --pairs-per-event 8)
        merge_args+=(--large-k512-cases)
    elif [[ "$mode" == large-* ]]; then
        common+=(--large-tokens)
        merge_args+=(--large-tokens)
        if [[ "$mode" == large-m-only ]]; then
            baseline="$label-large-full-card$card-vllm.csv"
            projection=cube_m_only
        fi
    fi
    tilelang_args=()
    if [[ "${2:-all}" == allocator-ab-b4 ]]; then
        tilelang_args+=(--allocator-ab-b4)
    fi
    if [[ "${2:-all}" == validate-b4 ]]; then
        tilelang_args+=(--verify-b4-stream)
    fi
    if [[ "${2:-all}" == compare-b4 || "${2:-all}" == direct-b4 || "${2:-all}" == allocate-b4 ]]; then
        tilelang_args+=(--compare-fresh-b4)
        if [[ "${2:-all}" == direct-b4 || "${2:-all}" == allocate-b4 ]]; then
            tilelang_args+=(--compare-direct-b4)
        fi
        if [[ "${2:-all}" == allocate-b4 ]]; then
            tilelang_args+=(--compare-allocate-b4)
        fi
    fi
    if [[ "$mode" == large-k512-post-double || "$mode" == large-k512-post-double-profile ]]; then
        common+=(--n4096-only)
        merge_args+=(--n4096-only)
        tilelang_args+=(--full-h-double-post)
    fi
    tilelang_output=(--output "$prefix-tilelang.csv")
    vllm_output=(--output "$prefix-vllm.csv")
    if [[ "$mode" == profile || "$mode" == large-k512-*-profile || "$mode" == large-k512-profile ]]; then
        profile_batch=4
        if [[ "$mode" == profile ]]; then
            common+=(--pairs-per-event 32 --profile-steps 32)
        fi
        if [[ "$mode" == large-k512* ]]; then
            profile_batch=4096
        fi
        profile_dir="$prefix-b$profile_batch-$(date +%Y%m%d-%H%M%S)-$$"
        common+=(--profile-batch "$profile_batch" --profile-dir "$profile_dir")
        tilelang_output=()
        vllm_output=()
    fi
    run_tilelang() {
        python bench_vllm_ascend.py --backend tilelang --projection "$projection" \
            "${common[@]}" "${tilelang_args[@]}" "${tilelang_output[@]}"
    }
    vllm_first=false
    if [[ "${2:-all}" == validate-b4 ]] && (( round % 2 == 0 )); then
        vllm_first=true
    fi
    if [[ "$mode" != tiling && "$vllm_first" == false ]]; then
        run_tilelang
    fi
    if [[ "$mode" != large-m-only ]]; then
        cann_env=()
        vllm_args=("${common[@]}" "${vllm_output[@]}")
        if [[ "$mode" == tiling ]]; then
            cann_env=(-e ASCEND_GLOBAL_LOG_LEVEL=1 -e ASCEND_SLOG_PRINT_TO_STDOUT=1)
            vllm_args=("${common[@]}" --large-k512-cases --tiling-only)
        fi

        run_vllm() {
            docker run --rm --pull=never --shm-size=1g "${devices[@]}" \
                "${cann_env[@]}" -e ASCEND_RT_VISIBLE_DEVICES \
                -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
                -v /usr/local/dcmi:/usr/local/dcmi:ro \
                -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" \
                -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
                -v "$PWD:/fgw-mhc" -w /fgw-mhc --entrypoint python \
                quay.io/ascend/vllm-ascend:v0.30.0-a5 bench_vllm_ascend.py \
                --backend vllm "${vllm_args[@]}"
        }

        if [[ "$mode" == tiling ]]; then
            log="$prefix-$(date +%Y%m%d-%H%M%S).log"
            if ! run_vllm >"$log" 2>&1; then
                tail -n 60 "$log" >&2
                exit 1
            fi
            grep -E 'BEGIN vLLM HcPre|END vLLM HcPre|HcPre arch35 tiling decision' "$log" || true
            if ! grep -q 'HcPre arch35 tiling decision' "$log"; then
                echo "The runtime did not emit the HcPre tiling decision to stdout."
            fi
            echo "Full CANN log: $log"
            continue
        fi
        run_vllm
    fi
    if [[ "$vllm_first" == true ]]; then
        run_tilelang
    fi
    if [[ "$mode" == profile || "$mode" == large-k512-*-profile || "$mode" == large-k512-profile ]]; then
        cat "$profile_dir"/{tilelang,vllm}/"b$profile_batch-h2560"/unprofiled-pair.csv
        if [[ "$mode" == profile ]]; then
            cat "$profile_dir"/{tilelang,vllm}/"b$profile_batch-h2560"/device-pair.csv
        fi
        continue
    fi
    if [[ "$mode" == sweep ]]; then
        cat "$prefix-tilelang.csv" "$prefix-vllm.csv"
        if [[ "${2:-all}" == allocator-ab-b4 ]]; then
            python - "$prefix-tilelang.csv" "$prefix-vllm.csv" <<'PY'
import csv
from statistics import mean, median
import sys

def read(path):
    with open(path, newline="") as source:
        return {(int(row["pairs_per_event"]), row["order"], row["projection"]): float(row["pair_us"])
                for row in csv.DictReader(source)}

fgw, vllm = map(read, sys.argv[1:])
for count in (1, 8, 32):
    orders = ("forward", "reverse")
    baseline = mean(vllm[count, order, ""] for order in orders)
    python_us = [fgw[count, f"{cycle}-{order}", "python_pre"] for cycle in range(1, 5) for order in orders]
    cpp_us = [fgw[count, f"{cycle}-{order}", "cpp_pre"] for cycle in range(1, 5) for order in orders]
    deltas = [a - b for a, b in zip(python_us, cpp_us, strict=True)]
    wins = sum(delta > 0 for delta in deltas)
    print(f"{count} pairs/event: Python {median(python_us):.3f} us, C++ {median(cpp_us):.3f} us, "
          f"paired saving {median(deltas):+.3f} us, C++ wins {wins}/{len(deltas)}, "
          f"vLLM {baseline:.3f} us")
PY
        fi
        if [[ "${2:-all}" == compare-b4 || "${2:-all}" == direct-b4 || "${2:-all}" == allocate-b4 ]]; then
            python - "$prefix-tilelang.csv" "$prefix-vllm.csv" <<'PY'
import csv
from statistics import mean
import sys

def times(path):
    with open(path, newline="") as source:
        return {(row["projection"], row["order"]): float(row["pair_us"])
                for row in csv.DictReader(source) if row["pairs_per_event"] == "32"}

fgw, vllm = map(times, sys.argv[1:])
orders = ("forward", "reverse")
baseline = mean(vllm["", order] for order in orders)
for name in dict.fromkeys(name for name, _ in fgw):
    latency = mean(fgw[name, order] for order in orders)
    print(f"{name}: {latency:.3f} us/pair, vLLM {baseline:.3f} us/pair, {baseline / latency:.3f}x")
if ("cube_full_fresh_direct", "forward") in fgw:
    saved = mean(fgw["cube_full_fresh", order] - fgw["cube_full_fresh_direct", order] for order in orders)
    print(f"Direct launcher delta: {saved:+.3f} us/pair saved at 32 pairs/event")
if ("cube_full_fresh_direct_cpp", "forward") in fgw:
    saved = mean(fgw["cube_full_fresh_direct", order] - fgw["cube_full_fresh_direct_cpp", order] for order in orders)
    print(f"C++ allocator delta: {saved:+.3f} us/pair saved at 32 pairs/event")
PY
        fi
        continue
    fi
    python bench_vllm_ascend.py --merge "$prefix-tilelang.csv" "$baseline" \
        "${merge_args[@]}" --output "$prefix-results.csv"
    cat "$prefix-results.csv"
done

if [[ "${2:-all}" == validate-b4 ]]; then
    python - "${validation_csvs[@]}" <<'PY'
import csv
from statistics import mean, median
import sys

def latencies(path):
    with open(path, newline="") as source:
        return {(int(row["pairs_per_event"]), row["order"]): float(row["pair_us"])
                for row in csv.DictReader(source)}

orders = ("forward", "reverse")
ratios = {n: [] for n in (1, 8, 32)}
paths = iter(sys.argv[1:])
for round_id, (fgw_path, vllm_path) in enumerate(zip(paths, paths, strict=True), start=1):
    fgw, vllm = latencies(fgw_path), latencies(vllm_path)
    for count, samples in ratios.items():
        ours = mean(fgw[count, order] for order in orders)
        baseline = mean(vllm[count, order] for order in orders)
        samples.append(baseline / ours)
        print(f"Round {round_id}, {count} pairs/event: FGW {ours:.3f} us, "
              f"vLLM {baseline:.3f} us, {samples[-1]:.3f}x")

for count, samples in ratios.items():
    wins = sum(value > 1 for value in samples)
    print(f"{count} pairs/event: median {median(samples):.3f}x, "
          f"range {min(samples):.3f} to {max(samples):.3f}x, wins {wins}/{len(samples)}")
PY
fi
