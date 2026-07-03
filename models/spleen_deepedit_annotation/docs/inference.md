# Spleen DeepEdit Inference

## Execute inference on AMD MI300X (ROCm)

Run the bundle with the `inference_rocm.json` overlay. Use a local (non-NFS) cache dir and do
not run as root (the MIOpen cache is SQLite-backed and breaks on NFS / as read-only root).

These environment variables are required (`PYTORCH_MIOPEN_SUGGEST_NHWC` and the MIOpen
`FIND` variables must be set **before** `python` starts — MIOpen latches them at import):

```bash
cd models/spleen_deepedit_annotation

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

On a fresh cache MIOpen selects the fast XDLOPS conv solver but runs it with **default,
untuned kernel parameters**. To populate the MIOpen perf-database with tuned parameters,
run **once** with exhaustive find enabled (`FIND_ENFORCE=4` = re-tune and write). This is
slow (the first inference does the full solver search) but only needs to be done once per
cache — the tuned perf-db persists on disk and is reused by all later runs.

```bash
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 \
python -m monai.bundle run \
  --config_file "['configs/inference.json', 'configs/inference_rocm.json']" \
  --datalist "['<abs-path-to-spleen-test-volume.nii.gz>']"
```

### Run

After the cache is tuned, run normally (no `FIND_ENFORCE`):

```bash
python -m monai.bundle run \
  --config_file "['configs/inference.json', 'configs/inference_rocm.json']" \
  --datalist "['<abs-path-to-spleen-test-volume.nii.gz>']"
```

The first inference in each new process is slow (`torch.compile` re-traces per process; not
cacheable across processes on ROCm); subsequent inferences in the same process run at full speed.

### Steady-state results (MI300X, spleen_15 128³ volume)

| Configuration | Inference time (ms) |
| :--- | :---: |
| FP32 eager (baseline) | ~8.93 ms |
| BF16 + channels_last_3d + torch.compile + use_gemm_transpose | **~6.72 ms** |

**Speedup: ~25% faster** than FP32 eager baseline.

*Note: Figures from validation runs with the complete optimization stack. Actual performance depends on MIOpen perf-database tuning state (see "One-time MIOpen tuning" above) and `torch.compile` warm-up (first inference per process is slow).*

The optimization stack includes:
- **BF16 mixed precision**: reduces memory bandwidth and uses tensor cores
- **channels_last_3d memory format**: better cache locality for 3D convolutions
- **torch.compile**: fuses operators and optimizes kernel selection
- **use_gemm_transpose**: replaces decoder `ConvTranspose3d` upsamples (where `kernel_size == stride`) with an exact pixel-shuffle GEMM decomposition

If steady-state is significantly slower, verify MIOpen selected the fast solver with logging:

```bash
MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6 python -m monai.bundle run ... 2>miopen.log
grep -E 'FW Chosen Algorithm' miopen.log | sort | uniq -c
# fast (correct): ConvHipImplicitGemm   |   slow (fallback): ConvDirectNaiveConvFwd
```

### Configuration details

The `inference_rocm.json` overlay enables:

1. **use_gemm_transpose**: Activates the GEMM decomposition for DynUNet decoder upsamples (ROCm-only optimization)
2. **channels_last_3d**: Sets NHWC memory layout via `torch.channels_last_3d`
3. **BF16 mixed precision**: Configures `amp_kwargs={'dtype': torch.bfloat16}` 
4. **torch.compile**: Wraps the entire network for graph optimization

### Correctness validation

The optimized configuration has been validated for numerical correctness:

- **Anchored Dice gate**: All test rungs pass with Dice ≥ 0.999 vs FP32 reference
  - BF16 + GEMM: Dice = 0.9997
  - FP16 + GEMM: Dice = 0.9996
  - Argmax agreement: 1.0000

- **Bundle A/B test**: Real `monai.bundle run` with/without overlay yields Dice(without, with) = 0.9999

The GEMM decomposition is mathematically exact when `kernel_size == stride` (zero output-window overlap). 
All configurations falling outside these preconditions automatically fall back to the stock transposed convolution.

---

## System Requirements

- AMD MI300X GPU
- ROCm 6.2 or later
- PyTorch 2.4+ with ROCm support
- MONAI 1.5.2+
- At least 16GB GPU memory
- Local (non-NFS) filesystem for MIOpen cache

## References

This optimization leverages:
- NHWC layout recommendation for MI300X CDNA architecture
- MIOpen XDLOPS implicit-GEMM convolution solver
- PyTorch Inductor autotune with Triton kernels
- DynUNet GEMM-based transpose convolution decomposition
