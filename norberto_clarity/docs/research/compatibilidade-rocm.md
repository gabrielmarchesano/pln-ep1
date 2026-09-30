# RX 6650 XT ROCm compatibility

## Scope

Test performed on 2026-09-22 to validate the environment before any long training run. It did not execute an epoch, create a dataset split, or compute predictive metrics. The host provides the `amdgpu` driver, `/dev/kfd`, and `/dev/dri`; user-space libraries remain inside the container.

Validated environment:

- AMD Radeon RX 6650 XT, 8 GiB, native `gfx1032` architecture;
- ROCm 6.4.1 (`6.4.1-83` in the tested runtime);
- PyTorch `2.8.0+rocm6.4`, HIP `6.4.43482-0f2d60242`;
- Transformers 4.57.1, PEFT 0.17.1, and Accelerate 1.10.1;
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `AMDGPU_TARGETS=gfx1030`.

AMD specifications identify the RX 6650 XT as `gfx1032`, but it is absent from the current set of officially supported Radeon GPUs. An AMD team response in the ROCm repository confirms that it is unsupported and suggests trying the variables above for this device: [ROCm discussion 2717](https://github.com/ROCm/ROCm/discussions/2717). Treat this procedure as a workaround, not guaranteed compatibility.

## Results

Without `HSA_OVERRIDE_GFX_VERSION`, `torch.cuda.is_available()` returned `True`, PyTorch identified the GPU name correctly, and it reported the `gfx1032` architecture. The first elementary multiplication failed with `HIP error: invalid device function`.

The numerical diagnostic passed with the `gfx1030` override:

| Check | Result |
|---|---:|
| FP32 GEMM, maximum absolute error against CPU | `2.63e-5` |
| Linear loss with backward pass | `1.166804`, finite gradients |
| SDPA with backward pass | output and gradients matched CPU |
| Peak allocated memory | 64.24 MiB |
| Duration | 9.96 s |

A single technical step also exercised the project's real model path: pinned BERTimbau revision, LoRA on `query`/`value` in all 12 blocks, SDPA, batch size 2, sequence length 128, forward, backward, and one AdamW step. It produced:

| Measurement | Result |
|---|---:|
| Trainable parameters | 149,763 |
| Synthetic loss | 1.252037 |
| Finite gradient tensors | 50 |
| Peak allocated memory | 643.65 MiB |
| Peak reserved memory | 662.00 MiB |
| Total duration, including model load | 15.35 s |

These tests establish technical compatibility for this short path. They do not estimate thermal stability, sustained performance, or classifier quality.

## Usage

The container must receive the host devices and architecture variables. The concrete image is deliberately local and is not part of this repository. Generic example:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e AMDGPU_TARGETS=gfx1030 \
  -v "$PWD:/workspace" \
  <rocm-pytorch-image> \
  python3 -m clarity.hardware
```

PyTorch code continues to call the device `cuda`. Do not replace it with `rocm` or `hip` in device arguments.

Run `PYTHONPATH=src python -m clarity.hardware` first. It covers representative numerical operations without accessing the corpus. Fine-tuning also calls `probe_accelerator()` before creating output directories; the probe executes a real kernel and rejects configurations that merely detect the GPU.

Repeat validation after changing the PyTorch wheel, ROCm version, kernel, firmware, image, or architecture variables. A passing result with the override does not make `gfx1032` officially supported hardware.
