#!/bin/bash
# Inference timing for vista3d (everything_labels, 20 spleen images)
# NVIDIA: uses inference_trt.json (trt_compile encoder)
# AMD:    uses plain inference.json (PyTorch model)
# Usage: bash run_inference.sh [--data-dir <path>]

set -e
cd "$(dirname "$0")"

# Data dir — override with --data-dir if needed
DATA_DIR="/home/AMD/nilapate/Domains_SDK_Utils/silo-engagement/data/btcv/Task09_Spleen"
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect GPU vendor — prefer /usr/bin/python3 (has TRT on NVIDIA), else fall back to python3 (ROCm)
if /usr/bin/python3 -c "import torch_tensorrt" 2>/dev/null; then
    echo "[run_inference] NVIDIA/TRT detected"
    PYTHON=/usr/bin/python3
    EXTRA_CONFIGS="'configs/inference_trt.json',"
else
    echo "[run_inference] AMD/ROCm detected"
    ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    export PYTORCH_MIOPEN_SUGGEST_NHWC=1
    export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$USER
    export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_$USER
    export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_$USER
    export TORCHINDUCTOR_MAX_AUTOTUNE=1
    export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
    export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
    export TORCHINDUCTOR_EPILOGUE_FUSION=1
    export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN,TRITON
    export MIOPEN_FIND_MODE=1
    export MIOPEN_FIND_ENFORCE=4
    echo "[run_inference] Using MIOpen cache dir: $MIOPEN_USER_DB_PATH"
    echo "[run_inference] Using TorchInductor cache dir: $TORCHINDUCTOR_CACHE_DIR"
    mkdir -p $MIOPEN_USER_DB_PATH
    mkdir -p $TORCHINDUCTOR_CACHE_DIR
    echo "[run_inference] Using ROCm Python: $(which python3)"
    echo "[run_inference] TORCHINDUCTOR_MAX_AUTOTUNE: $TORCHINDUCTOR_MAX_AUTOTUNE"
    echo "[run_inference] TORCHINDUCTOR_MAX_AUTOTUNE_GEMM: $TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"
    echo "[run_inference] TORCHINDUCTOR_COORDINATE_DESCENT_TUNING: $TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"
    echo "[run_inference] TORCHINDUCTOR_EPILOGUE_FUSION: $TORCHINDUCTOR_EPILOGUE_FUSION"
    echo "[run_inference] TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS: $TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS"
    echo "[run_inference] MIOPEN_FIND_MODE: $MIOPEN_FIND_MODE"
    echo "[run_inference] MIOPEN_FIND_ENFORCE: $MIOPEN_FIND_ENFORCE"
    PYTHON=python3
    EXTRA_CONFIGS="'configs/inference_rocm.json',"
fi

$PYTHON -c "
import glob, os
from monai.bundle.scripts import run

imgs = sorted(glob.glob('${DATA_DIR}/imagesTs/*.nii.gz'))
imgs = [f for f in imgs if os.path.exists(f)]
print(f'[run_inference] Found {len(imgs)} images')
everything_labels = list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))

run(
    config_file=['configs/inference.json', ${EXTRA_CONFIGS}],
    bundle_root='.',
    **{'dataset#data': [{'image': f, 'label_prompt': everything_labels} for f in imgs]}
)
"
