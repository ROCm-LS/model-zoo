#!/bin/bash
# Inference timing for wholeBody_ct_segmentation (TRT model)
# Usage: bash run_inference.sh
# Output: eval_all/<case>/<case>_trans.nii.gz + TimedInferer summary

set -e
cd "$(dirname "$0")"

DATA_DIR="/home/AMD/nilapate/Domains_SDK_Utils/silo-engagement/data/btcv/Task09_Spleen"

/usr/bin/python3 -c "
from monai.bundle.scripts import run
run(
    config_file=['configs/inference.json', 'configs/inference_trt.json', 'configs/override_all_images.json'],
    bundle_root='.',
    **{'dataloader#num_workers': 0}
)
"
