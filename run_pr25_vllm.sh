#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

card="${1:-4}"
if [[ $# -gt 0 ]]; then shift; fi
repo="${PR25_REPO:-../tilelang-asc-pr25}"
head=053ddb470aefcbc6972d994739983734eeb4eaf8
if [[ ! -e "$repo/.git" ]]; then
    git init "$repo"
    git -C "$repo" fetch --depth 1 https://github.com/yangsichan/tilelang-asc.git refs/pull/25/head
    git -C "$repo" checkout --detach "$head"
fi

export ASCEND_RT_VISIBLE_DEVICES="$card"
devices=(--device /dev/davinci_manager)
for node in /dev/davinci[0-9]* /dev/hisi_hdc /dev/ummu /dev/uburma; do
    if [[ -c "$node" ]]; then devices+=(--device "$node"); fi
done

prefix="pr25-vllm-card$card"
common=(--device npu:0 "$@")
python bench_pr25_vllm.py --backend pr25 --pr-repo "$repo" "${common[@]}" --output "$prefix-pr25.csv"
docker run --rm --pull=never --shm-size=1g "${devices[@]}" -e ASCEND_RT_VISIBLE_DEVICES \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$PWD:/fgw-mhc" -w /fgw-mhc --entrypoint python \
    quay.io/ascend/vllm-ascend:v0.30.0-a5 bench_pr25_vllm.py \
    --backend vllm "${common[@]}" --output "$prefix-vllm.csv"
python bench_pr25_vllm.py --merge "$prefix-pr25.csv" "$prefix-vllm.csv" --output "$prefix-results.csv"
cat "$prefix-results.csv"
