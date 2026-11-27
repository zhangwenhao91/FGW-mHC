#!/usr/bin/env bash
# Run both native stages and the full pair with the TileOPs MHC manifest settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

card="${1:-4}"
case "${2:-all}" in
    all) modes=(stages pair); projection=cube_full; label=tileops-spec ;;
    tuned-pair) modes=(pair); projection=tileops_tuned; label=tileops-spec-tuned ;;
    profile-b4) modes=(profile); projection=cube_full; label=tileops-spec ;;
    *) echo "Usage: $0 [card] [all|tuned-pair|profile-b4]" >&2; exit 2 ;;
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
    if [[ "$mode" == stages ]]; then
        stage_args=(--split-stages)
    fi
    prefix="$label-$mode-card$card"
    common=(--tileops-spec --device npu:0 "${stage_args[@]}")
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
    docker run --rm --pull=never --shm-size=1g "${devices[@]}" \
        -e ASCEND_RT_VISIBLE_DEVICES \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "$PWD:/fgw-mhc" -w /fgw-mhc --entrypoint python \
        quay.io/ascend/vllm-ascend:v0.30.0-a5 bench_vllm_ascend.py \
        --backend vllm "${common[@]}" "${vllm_output[@]}"
    if [[ "$mode" == profile ]]; then
        cat "$profile_dir"/{tilelang,vllm}/b4-h2560/unprofiled-pair.csv
        continue
    fi
    python bench_vllm_ascend.py --merge "$prefix-tilelang.csv" "$prefix-vllm.csv" \
        --output "$prefix-results.csv"
    cat "$prefix-results.csv"
done
