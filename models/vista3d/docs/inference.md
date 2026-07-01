All the configurations for inference is stored in inference.json, change those parameters:
### `input_dict`
`input_dict` defines the image to segment and the prompt for segmentation.
```
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'label_prompt':[1]}]",
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'points':[[138,245,18], [271,343,27]], 'point_labels':[1,0]}]"
```
- The input_dict must include the key `image` which contain the absolute path to the nii image file, and includes prompt keys of `label_prompt`, `points` and `point_labels`.
- The `label_prompt` is a list of length `B`, which can perform `B` foreground objects segmentation, e.g. `[2,3,4,5]`. If `B>1`, Point prompts must NOT be provided.
- The `points` is of shape `[N, 3]` like `[[x1,y1,z1],[x2,y2,z2],...[xN,yN,zN]]`, representing `N` point coordinates **IN THE ORIGINAL IMAGE SPACE** of a single foreground object. `point_labels` is a list of length [N] like [1,1,0,-1,...], which
matches the `points`. 0 means background, 1 means foreground, -1 means ignoring this point. `points` and `point_labels` must pe provided together and match length.
- **B must be 1 if label_prompt and points are provided together**. The inferer only supports SINGLE OBJECT point click segmentatation.
- If no prompt is provided, the model will use `everything_labels` to segment 117 classes:

```Python
list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))
```

- The `points` together with `label_prompts` for "Kidney", "Lung", "Bone" (class index [2, 20, 21]) are not allowed since those prompts will be divided into sub-categories (e.g. left kidney and right kidney). Use `points` for the sub-categories as defined in the `inference.json`.
- To specify a new class for zero-shot segmentation, set the `label_prompt` to a value between 133 and 254. Ensure that `points` and `point_labels` are also provided; otherwise, the inference result will be a tensor of zeros.

### `label_prompt` and `label_dict`
The `label_dict` defined in `configs/metadata.json` has in total 132 classes. However, there are 5 we do not support and we keep them due to legacy issue. So in total
VISTA3D support 127 classes.

```
"16, # prostate or uterus" since we already have "prostate" class,
"18, # rectum", insufficient data or dataset excluded.
"130, # liver tumor" already have hepatic tumor.
"129, # kidney mass" insufficient data or dataset excluded.
"131, # vertebrae L6", insufficient data or dataset excluded.
```

These 5 are excluded in the `everything_labels`. Another 7 tumor and vessel classes are also removed since they will overlap with other organs and make the output messy. To segment those 7 classes, we recommend users to directly set `label_prompt` to those indexes and avoid using them in `everything_labels`. For "Kidney", "Lung", "Bone" (class index [2, 20, 21]), VISTA3D did not directly use the class index for segmentation, but instead convert them to their subclass indexes as defined by `subclass` dict. For example, "2-Kidney" is converted to "14-Left Kidney" + "5-Right Kidney" since "2" is defined in `subclasss` dict.

### `resample_spacing`
The optimal inference resample spacing should be changed according to the task. For monkey data, a high resolution of [1,1,1] showed better automatic inference results. This spacing applies to both automatic and interactive segmentation. For zero-shot interactive segmentation for non-human CTs e.g. mouse CT or even rock/stone CT, using original resolution (set `resample_spacing` to [-1,-1,-1]) may give better interactive results.

### `use_point_window`
When user click a point, there is no need to perform whole image sliding window inference. Set "use_point_window" to true in the inference.json to enable this function.
A window centered at the clicked points will be used for inference. All values outside of the window will set to be "NaN" unless "prev_mask" is passed to the inferer (255 is used to represent NaN).
If no point click exists, this function will not be used. Notice if "use_point_window" is true and user provided point clicks, there will be obvious cut-off box artefacts.


### Inference GPU benchmarks
Benchmarks on a 16GB V100 GPU with 400G system cpu memory.
| Volume size at 1.5x1.5x1.5 mm | 333x333x603 | 512x512x512 | 512x512x768 | 1024x1024x512 | 1024x1024x768 |
| :---: | :---: | :---: | :---: | :---: | :---: |
|RunTime| 1m07s | 2m09s | 3m25s| 9m20s| killed |



### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

By default, the argument `head_trt_enabled` is set to `false` in `configs/inference_trt.json`. This means that the `class_head` module of the network will not be converted into a TensorRT model. Setting this to `true` may accelerate the process, but there are some limitations:

Since the `label_prompt` will be converted into a tensor and input into the `class_head` module, the batch size of this input tensor will equal the length of the original `label_prompt` list (if no prompt is provided, the length is 117). To make the TensorRT model work on the `class_head` module, you should set a suitable dynamic batch size range. The maximum dynamic batch size can be configured using the argument `max_prompt_size` in `configs/inference_trt.json`. If the length of the `label_prompt` list exceeds `max_prompt_size`, the engine will fall back to using the normal PyTorch model for inference. Setting a larger `max_prompt_size` can cover more input cases but may require more GPU memory (the default value is 4, which requires 16 GB of GPU memory). Therefore, please set it to a reasonable value according to your actual requirements.


### TensorRT speedup
The `vista3d` bundle supports acceleration with TensorRT. The table below displays the speedup ratios observed on an A100 80G GPU. Please note for 32bit precision models, they are benchmarked with tf32 weight format.

| method | torch_tf32(ms) | torch_amp(ms) | trt_tf32(ms) | trt_fp16(ms) | speedup amp | speedup tf32 | speedup fp16 | amp vs fp16|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| model computation | 108.53| 91.9 | 106.84 | 60.02 | 1.18 | 1.02 | 1.81 | 1.53 |
| end2end | 6740 | 5166 | 5242 | 3386 | 1.30 | 1.29 | 1.99 | 1.53 |

Where:
- `model computation` means the speedup ratio of model's inference with a random input without preprocessing and postprocessing
- `end2end` means run the bundle end-to-end with the TensorRT based model.
- `torch_tf32` and `torch_amp` are for the PyTorch models with or without `amp` mode.
- `trt_tf32` and `trt_fp16` are for the TensorRT based models converted in corresponding precision.
- `speedup amp`, `speedup tf32` and `speedup fp16` are the speedup ratios of corresponding models versus the PyTorch float32 model
- `amp vs fp16` is the speedup ratio between the PyTorch amp model and the TensorRT float16 based model.

This result is benchmarked under:
 - TensorRT: 10.3.0+cuda12.6
 - Torch-TensorRT Version: 2.4.0
 - CPU Architecture: x86-64
 - OS: ubuntu 20.04
 - Python version:3.10.12
 - CUDA version: 12.6
 - GPU models and configuration: A100 80G

---

## Execute inference on AMD MI300X (ROCm)

Run the bundle with the `inference_amd.json` overlay. Use a local (non-NFS) cache dir and do
not run as root (the MIOpen cache is SQLite-backed and breaks on NFS / as read-only root).

These environment variables are required (`PYTORCH_MIOPEN_SUGGEST_NHWC` and the MIOpen
`FIND` variables must be set **before** `python` starts — MIOpen latches them at import):

```bash
cd models/vista3d

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
untuned kernel parameters** (~865 ms/inference). To populate the MIOpen perf-database with
tuned parameters, run **once** with exhaustive find enabled (`FIND_ENFORCE=4` = re-tune and
write). This is slow (the first inference does the full solver search) but only needs to be
done once per cache — the tuned perf-db persists on disk and is reused by all later runs.

```bash
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 \
python -m monai.bundle run \
  --config_file "['configs/inference.json', 'configs/inference_amd.json']" \
  --input_dict "{'image': '/abs/path/to/Task09_Spleen/imagesTr/spleen_10.nii.gz', 'label_prompt': [3]}"
```

### Run

After the cache is tuned, run normally (no `FIND_ENFORCE`):

```bash
python -m monai.bundle run \
  --config_file "['configs/inference.json', 'configs/inference_amd.json']" \
  --input_dict "{'image': '/abs/path/to/Task09_Spleen/imagesTr/spleen_10.nii.gz', 'label_prompt': [3]}"
```

The inferer prints `[Vista3dInferer] inference time: X ms` per call. The first inference in
each new process is slow (`torch.compile` re-traces per process; not cacheable across
processes on ROCm); subsequent inferences in the same process run at full speed.

### Steady-state results (MI300X, BTCV spleen_10, 334x300x181 volume)

| MIOpen perf-db state | steady-state inference |
| :--- | :---: |
| untuned (fresh cache, default solver params) | ~865 ms |
| tuned once with `FIND_ENFORCE=4` (this recipe) | **~540 ms** |

If steady-state is in the **seconds** (not ~0.5–0.9 s), MIOpen fell back to the naive conv
kernel. Confirm the fast solver with MIOpen logging:

```bash
MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6 python -m monai.bundle run ... 2>miopen.log
grep -E 'FW Chosen Algorithm' miopen.log | sort | uniq -c
# fast (correct): ConvHipImplicitGemm   |   slow (fallback): ConvDirectNaiveConvFwd
```

---

## Compilation strategy: compile the whole network (not just the encoder)

The `inference_amd.json` overlay wraps the **whole network** with `torch.compile(@network)`.
NVIDIA's TensorRT overlay (`inference_trt.json`) compiles only the `image_encoder.encoder`
submodule, because TensorRT cannot ingest VISTA3D's branchy `forward` (data-dependent `if`s,
`.item()`/`.any()` breaks, variable-size prompt heads). That is a TensorRT constraint, **not**
a general one — `torch.compile` handles branches via graph breaks + guards, so on the AMD
`torch.compile` path there is no reason to restrict compilation to the encoder.

### Why whole-model is the right choice

Measured on MI300X (bf16, points path unless noted), encoder-only vs whole-model:

| path | whole-model | encoder-only | delta |
| :--- | :---: | :---: | :---: |
| **label_prompt (production)** | **873 ms** | 1625 ms | whole-model **86% faster** |
| points (interactive) | 153 ms | 149 ms | tie (noise) |

Encoder-only leaves the SegResNet-DS decoder + segmentation heads running eager, which nearly
doubles latency on the production label_prompt path. Whole-model captures the decoder speedup.

Correctness is preserved. The anchored Dice gate (`dice_equivalence.py --model vista3d
--fp32-gate`) passes for whole-model bf16 compile: fp32 reference is bit-equal to the
`monai.bundle run` golden, and every rung is Dice ≥ 0.99 vs reference (bf16+compile = 0.9990,
bf16+compile+wedge = 0.9990).

### Recompilation is bounded — safe at scale

`forward` is **not** one monolithic compiled graph. Dynamo splits it at each untraceable
branch (graph break), producing many independently-compiled subgraphs
(`torch_dynamo_resume_in_forward_at_<line>`). The `torch._dynamo` `recompile_limit` (default
**8**) is **per code-object, not per model**: each compiled function may recompile up to 8
times (once per new tensor shape/stride/dtype it sees) before that one function permanently
falls back to eager. Different code objects have independent counters.

### Label-prompt path needs far fewer compilations than the points path

The two prompt types exercise different inferers and therefore different recompilation
profiles:

- **label_prompt (automatic)** routes through `SlidingWindowInfererAdapt`, which **pads every
  window to a fixed 128³**. The tensor reaching the compiled `forward` is the same shape for
  every image, so Dynamo compiles once and reuses it — measured **~0–1 recompiles** across 61
  distinct images.
- **points (interactive)** routes through `point_based_window_inferer`, which crops
  **variable-size** windows around each click. The compiled `forward` sees several distinct
  shapes during warmup — measured ~43 recompiles (still bounded; see below).

So a **label-prompt-only deployment recompiles far less** than one that also serves interactive
point prompts. If you only need automatic class-based segmentation, the points path (and its
compilation cost) is never exercised.

> Note: the stock mainline bundle's points path raises `TypeError: expected np.ndarray (got
> Tensor)` at `scripts/evaluator.py` (`torch.from_numpy(points)`) on current PyTorch/NumPy,
> because the preceding `transform_points` now returns a Tensor. This is a pre-existing
> upstream bug independent of the AMD optimizations, and it does not affect the label_prompt
> path. The measurements below used a local `torch.as_tensor(points)` workaround, which is
> **not** shipped in this overlay (we keep `scripts/` identical to mainline).

Recompilation was stress-tested on the points path (the only path with variable-size crop
windows; the label_prompt path pads every window to 128³, so it recompiles ~0–1 times):

| dataset | distinct volumes | in-plane shape classes | Z-depths | total recompiles | last recompile | any function > 8? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Task09 Spleen | 61 | 1 | 31–168 | 45 | image 5 | no |
| AMOS22 (multi-scanner) | 300 | 2 (512², 768²) | 68–353 | 43 | image 2 | no |
| TotalSegmentator | 1228 | 493 | 29–851 | 43 | image 2 | no |

Across 1228 distinct CTs from 493 in-plane resolution classes (cold cache), recompilation
converges by **image 2** and never recurs. The 43 total is summed over 22 distinct functions;
**exactly one** — `copy_meta_from` (MONAI meta-tensor plumbing, `meta_obj.py:122`, not a
compute kernel) — reaches the cap of 8 and falls back to eager harmlessly. The encoder
(`segresnet_ds`) tops out at 5/8 and stays compiled. Recompilation is bounded by the number of
distinct **tensor shapes** (small, because resampling + fixed 128³ crop windows collapse all
images into a few shapes), **not** by image count — so 1000+ images behave identically to 300.

### Graceful-degradation floor

Because the limit is per-function, whole-model compile can only degrade **toward** encoder-only,
never below it. Worst case (pathological input diversity pushing every branchy function to its
cap): the branchy heads fall back to eager but the encoder — a clean CNN with stable shapes —
stays compiled, matching NVIDIA's encoder-only baseline. Best case (what actually happens): the
branchy parts also compile and stay compiled. There is no scenario where whole-model is slower
than encoder-only, which makes compiling the whole network a free option.

### Cold-start cost

The disk caches (`TORCHINDUCTOR_CACHE_DIR`, MIOpen `USER_DB`/`CUSTOM_CACHE_DIR`) affect only
one-time warmup compile *time*, not recompile count or steady-state latency. Cold cache: first
inference ~100 s (full inductor autotune), second ~50 s, total warmup ~150 s; then flat
steady-state. Persist the cache dirs on durable (non-NFS, non-root) storage to cut warmup to
~55 s on subsequent process starts. Either way it is a fixed startup tax, fully amortized.
