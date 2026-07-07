#!/bin/bash
# Inference timing for vista3d (plain PyTorch, everything_labels)
# Usage: bash run_inference.sh
# Output: eval/<case>/<case>_trans.nii.gz + TimedInferer summary

set -e
cd "$(dirname "$0")"

DATA_DIR="/home/AMD/nilapate/Domains_SDK_Utils/silo-engagement/data/btcv/Task09_Spleen"

/usr/bin/python3 -c "
import glob, os
from monai.bundle.scripts import run

imgs = sorted(glob.glob('${DATA_DIR}/imagesTs/*.nii.gz'))
imgs = [f for f in imgs if os.path.exists(f)]
everything_labels = list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))

run(
    config_file='configs/inference.json',
    bundle_root='.',
    **{'dataset#data': [{'image': f, 'label_prompt': everything_labels} for f in imgs]}
)
"
