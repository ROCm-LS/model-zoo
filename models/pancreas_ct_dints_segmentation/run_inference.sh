#!/bin/bash
# Inference timing for pancreas_ct_dints_segmentation
# NVIDIA: uses inference_trt.yaml (TRT-compiled model)
# AMD:    uses inference_rocm.yaml (NHWC + torch.compile + bf16)
# Data:   testing list from configs/dataset_0.json (absolute paths to Task07_Pancreas)
# Usage:  bash run_inference.sh [--data-dir <path>]

set -e
cd "$(dirname "$0")"

# Optional dataset_dir override (dataset_0.json already holds absolute paths).
DATA_DIR=""
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
    EXTRA_CONFIGS="'configs/inference_trt.yaml',"
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
    export MIOPEN_ENABLE_LOGGING=0
    export MIOPEN_ENABLE_LOGGING_CMD=0
    export MIOPEN_LOG_LEVEL=0
    export AMD_LOG_LEVEL=0
    export TORCH_COMPILE_DEBUG=0
    export TORCHINDUCTOR_VERBOSE=0
    echo "[run_inference] Using MIOpen cache dir: $MIOPEN_USER_DB_PATH"
    echo "[run_inference] Using TorchInductor cache dir: $TORCHINDUCTOR_CACHE_DIR"
    mkdir -p $MIOPEN_USER_DB_PATH
    mkdir -p $TORCHINDUCTOR_CACHE_DIR
    echo "[run_inference] Using ROCm Python: $(which python3)"
    echo "[run_inference] MIOPEN_FIND_MODE: $MIOPEN_FIND_MODE"
    echo "[run_inference] MIOPEN_FIND_ENFORCE: $MIOPEN_FIND_ENFORCE"
    PYTHON=python3
    EXTRA_CONFIGS="'configs/inference_rocm.yaml',"
fi

DATASET_OVERRIDE=""
if [[ -n "$DATA_DIR" ]]; then
    DATASET_OVERRIDE="--dataset_dir '$DATA_DIR'"
fi

$PYTHON -m monai.bundle run \
    --config_file "['configs/inference.yaml', ${EXTRA_CONFIGS%,}]" \
    --bundle_root . \
    $DATASET_OVERRIDE
