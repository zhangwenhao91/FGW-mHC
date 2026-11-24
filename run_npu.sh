#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_cmd="${PYTHON:-python3}"

case "${1:-}" in
    --list-cases|-h|--help)
        exec "$python_cmd" bench_mhc.py "$@"
        ;;
esac

"$python_cmd" -c 'import torch, torch_npu, tilelang.ascend.language; assert torch.npu.is_available(), "No usable Ascend NPU"'
"$python_cmd" -m pytest -q tests >&2
exec "$python_cmd" bench_mhc.py "$@"
