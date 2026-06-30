# SwinUNETR BTCV Inference

## Execute inference on AMD MI300X (ROCm)

Run the bundle with the `inference_amd.json` overlay. Use a local (non-NFS) cache dir and do
not run as root (the MIOpen cache is SQLite-backed and breaks on NFS / as read-only root).

**Config paths must be absolute** — a relative path causes the AMD activation check to
silently no-op and the optimization does not fire.

These environment variables are required (`PYTORCH_MIOPEN_SUGGEST_NHWC` and the MIOpen
`FIND` variables must be set **before** `python` starts — MIOpen latches them at import):

```bash
cd models/swin_unetr_btcv_segmentation
B=$(pwd)

ulimit -n 1048576
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$USER
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_$USER
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_$USER
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_EPILOGUE_FUSION=1
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN,TRITON
```

### One-time MIOpen tuning (required for full speed)

On a fresh cache, run **once** with exhaustive find enabled (`FIND_ENFORCE=4` = re-tune and
write). This is slow on the first inference but only needs to be done once per cache — the
tuned perf-db persists on disk and is reused by all later runs.

```bash
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 \
python -m monai.bundle run \
  --config_file "['$B/configs/inference.json', '$B/configs/inference_amd.json']" \
  --input_dict "{'image': '/abs/path/to/imagesTr/spleen_10.nii.gz'}"
```

### Run

After the cache is tuned, run normally (no `FIND_ENFORCE`):

```bash
python -m monai.bundle run \
  --config_file "['$B/configs/inference.json', '$B/configs/inference_amd.json']" \
  --input_dict "{'image': '/abs/path/to/imagesTr/spleen_10.nii.gz'}"
```

The inferer prints `[amd-timing] SlidingWindowInferer forward: X ms` per call. The first
1–2 inferences in each new process are slow (`torch.compile` traces on the first call and
recompiles once on the batch-size transition from 1 → ≥2 sliding windows); subsequent
inferences in the same process run at full speed.

### Steady-state results (MI300X, BTCV spleen_10)

| Configuration | steady-state inference (p50) |
| :--- | :---: |
| Baseline (no overlay) | ~8,000 ms |
| `inference_amd.json` overlay (this recipe) | **~417 ms** |

### What the overlay does

- **NHWC layout** (`channels_last_3d`): puts convolutions in the memory layout the fast
  MIOpen XDLOPS solver is tuned for.
- **bf16 autocast** (`evaluator.amp_kwargs`): halves memory bandwidth for the attention and
  conv layers.
- **`torch.compile`**: fuses elementwise ops and reduces kernel launch overhead.
- **SDPA for WindowAttention** (fork): dispatches the shifted-window self-attention to
  `torch.nn.functional.scaled_dot_product_attention` on AMD.
- **Dynamic batch marking** (fork): marks the sliding-window batch dimension as dynamic
  before the compiled predictor, bounding `torch.compile` recompiles to 2 over an entire
  multi-image run (mirrors NVIDIA TRT `dynamic_batchsize=[1,4,4]`).

### Troubleshooting

**Inference is in the seconds (not ~0.4 s):** MIOpen fell back to the naive conv kernel.
Confirm the fast solver with MIOpen logging:

```bash
MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6 python -m monai.bundle run ... 2>miopen.log
grep -E 'FW Chosen Algorithm' miopen.log | sort | uniq -c
# fast (correct): ConvHipImplicitGemm   |   slow (fallback): ConvDirectNaiveConvFwd
```

**AMD wedge does not fire (no `[amd-timing]` output):** Config paths are relative — re-run
with `B=$(pwd)` and `$B/configs/...` as shown above.
