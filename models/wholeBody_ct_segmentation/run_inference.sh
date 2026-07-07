#!/bin/bash
# Inference timing for wholeBody_ct_segmentation
# NVIDIA: uses inference_trt.json (pre-compiled TRT model)
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
from monai.bundle.scripts import run
run(
    config_file=['configs/inference.json', ${EXTRA_CONFIGS}'configs/override_all_images.json'],
    bundle_root='.',
    **{
        'dataloader#num_workers': 0,
        'dataset_dir': '${DATA_DIR}',
        'datalist': \"\$[x for d in ['imagesTr','imagesTs'] for x in sorted(__import__('glob').glob('${DATA_DIR}/'+d+'/*.nii.gz'))]\",
    }
)
"
