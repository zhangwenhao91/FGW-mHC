#!/usr/bin/env bash
# Run both native stages and the full pair with the TileOPs MHC manifest settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

card="${1:-4}"
case "${2:-all}" in
    all) modes=(stages pair); projection=cube_full; label=tileops-spec ;;
    tuned-pair) modes=(pair); projection=tileops_tuned; label=tileops-spec-tuned ;;
    *) echo "Usage: $0 [card] [all|tuned-pair]" >&2; exit 2 ;;
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
    python bench_vllm_ascend.py --backend tilelang --projection "$projection" \
        "${common[@]}" --output "$prefix-tilelang.csv"
    docker run --rm --pull=never --shm-size=1g "${devices[@]}" \
        -e ASCEND_RT_VISIBLE_DEVICES \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "$PWD:/fgw-mhc" -w /fgw-mhc --entrypoint python \
        quay.io/ascend/vllm-ascend:v0.30.0-a5 bench_vllm_ascend.py \
        --backend vllm "${common[@]}" --output "$prefix-vllm.csv"
    python bench_vllm_ascend.py --merge "$prefix-tilelang.csv" "$prefix-vllm.csv" \
        --output "$prefix-results.csv"
    cat "$prefix-results.csv"
done
