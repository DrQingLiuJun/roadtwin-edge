#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ! -f "$1" || -e "$2" ]]; then echo "Usage: bash tools/build_engine.sh model.onnx new-model.engine" >&2; exit 2; fi
trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
temporary="${2}.building"
trap 'rm -f -- "$temporary"' EXIT
"$trtexec" --onnx="$1" --saveEngine="$temporary" --fp16 --memPoolSize=workspace:512 --builderOptimizationLevel=0 --skipInference
"$trtexec" --loadEngine="$temporary" --warmUp=200 --duration=1
mv -- "$temporary" "$2"
