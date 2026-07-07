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
    PYTHON=python3
    EXTRA_CONFIGS=""
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
