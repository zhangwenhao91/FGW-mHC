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
    large-pair) modes=(large-full large-m-only); projection=cube_full; label=tileops-spec ;;
    large-stages) modes=(large-stages); projection=cube_m_only; label=tileops-spec ;;
    large-k512-pair) modes=(large-k512-pair); projection=cube_m_only_k512; label=tileops-spec ;;
    large-k512-stages) modes=(large-k512-stages); projection=cube_m_only_k512; label=tileops-spec ;;
    large-k512-deferred-pair) modes=(large-k512-deferred-pair); projection=cube_m_only_k512_deferred; label=tileops-spec ;;
    large-k512-s2) modes=(large-k512-s2-pair large-k512-s2-stages); projection=cube_m_only_k512_deferred_s2; label=tileops-spec ;;
    *) echo "Usage: $0 [card] [all|tuned-pair|profile-b4|sweep-b4|large-pair|large-stages|large-k512-pair|large-k512-stages|large-k512-deferred-pair|large-k512-s2]" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$card"
devices=(--device /dev/davinci_manager)
for node in /dev/davinci[0-9]* /dev/hisi_hdc /dev/ummu /dev/uburma; do
    if [[ -c "$node" ]]; then
        devices+=(--device "$node")
    fi
done

for mode in "${modes[@]}"; do
    stage_args=()
    if [[ "$mode" == stages || "$mode" == large-*-stages || "$mode" == large-stages ]]; then
        stage_args=(--split-stages)
    fi
    prefix="$label-$mode-card$card"
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
    tilelang_output=(--output "$prefix-tilelang.csv")
    vllm_output=(--output "$prefix-vllm.csv")
    if [[ "$mode" == profile ]]; then
        profile_dir="$prefix-b4-$(date +%Y%m%d-%H%M%S)-$$"
        common+=(--profile-batch 4 --profile-dir "$profile_dir")
        tilelang_output=()
        vllm_output=()
    fi
    python bench_vllm_ascend.py --backend tilelang --projection "$projection" \
        "${common[@]}" "${tilelang_output[@]}"
    if [[ "$mode" != large-m-only ]]; then
        docker run --rm --pull=never --shm-size=1g "${devices[@]}" \
            -e ASCEND_RT_VISIBLE_DEVICES \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
            -v /usr/local/dcmi:/usr/local/dcmi:ro \
            -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" \
            -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
            -v "$PWD:/fgw-mhc" -w /fgw-mhc --entrypoint python \
            quay.io/ascend/vllm-ascend:v0.30.0-a5 bench_vllm_ascend.py \
            --backend vllm "${common[@]}" "${vllm_output[@]}"
    fi
    if [[ "$mode" == profile ]]; then
        cat "$profile_dir"/{tilelang,vllm}/b4-h2560/unprofiled-pair.csv
        continue
    fi
    if [[ "$mode" == sweep ]]; then
        cat "$prefix-tilelang.csv" "$prefix-vllm.csv"
        continue
    fi
    python bench_vllm_ascend.py --merge "$prefix-tilelang.csv" "$baseline" \
        "${merge_args[@]}" --output "$prefix-results.csv"
    cat "$prefix-results.csv"
done
