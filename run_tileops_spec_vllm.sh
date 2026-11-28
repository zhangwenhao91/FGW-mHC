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
    large-k512-batched) modes=(large-k512-batched-pair large-k512-batched-stages); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    profile-n4096) modes=(large-k512-profile); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    tiling-large) modes=(tiling); label=tileops-spec ;;
    large-k512-post-double) modes=(large-k512-post-double); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    profile-n4096-post-double) modes=(large-k512-post-double-profile); projection=cube_m_only_k512_batched; label=tileops-spec ;;
    *) echo "Usage: $0 [card] [all|tuned-pair|profile-b4|sweep-b4|large-pair|large-stages|large-k512-pair|large-k512-stages|large-k512-deferred-pair|large-k512-s2|large-k512-batched|profile-n4096|tiling-large|large-k512-post-double|profile-n4096-post-double]" >&2; exit 2 ;;
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
    tilelang_args=()
    if [[ "$mode" == large-k512-post-double || "$mode" == large-k512-post-double-profile ]]; then
        common+=(--n4096-only)
        merge_args+=(--n4096-only)
        tilelang_args+=(--full-h-double-post)
    fi
    tilelang_output=(--output "$prefix-tilelang.csv")
    vllm_output=(--output "$prefix-vllm.csv")
    if [[ "$mode" == profile || "$mode" == large-k512-*-profile || "$mode" == large-k512-profile ]]; then
        profile_batch=4
        if [[ "$mode" == large-k512* ]]; then
            profile_batch=4096
        fi
        profile_dir="$prefix-b$profile_batch-$(date +%Y%m%d-%H%M%S)-$$"
        common+=(--profile-batch "$profile_batch" --profile-dir "$profile_dir")
        tilelang_output=()
        vllm_output=()
    fi
    if [[ "$mode" != tiling ]]; then
        python bench_vllm_ascend.py --backend tilelang --projection "$projection" \
            "${common[@]}" "${tilelang_args[@]}" "${tilelang_output[@]}"
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
    if [[ "$mode" == profile || "$mode" == large-k512-*-profile || "$mode" == large-k512-profile ]]; then
        cat "$profile_dir"/{tilelang,vllm}/"b$profile_batch-h2560"/unprofiled-pair.csv
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
