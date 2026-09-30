"""Short numerical checks for PyTorch CUDA and ROCm accelerators."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import torch
import torch.nn.functional as functional


def accelerator_info() -> dict[str, object]:
    """Describe the accelerator exposed through the PyTorch CUDA API."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA or ROCm accelerator is available to PyTorch.")
    properties = torch.cuda.get_device_properties(0)
    is_rocm = torch.version.hip is not None
    architecture = (getattr(properties, "gcnArchName", "unknown") if is_rocm
                    else f"sm_{properties.major}{properties.minor}")
    return {
        "backend": "rocm" if is_rocm else "cuda",
        "device_name": properties.name,
        "architecture": architecture,
        "runtime_version": torch.version.hip if is_rocm else torch.version.cuda,
        "torch_version": torch.__version__,
        "total_memory_mib": properties.total_memory / 2**20,
    }


def log_selected_device(info: dict[str, object], *, operation: str,
                        requested: str | None = None) -> None:
    """Show the effective PyTorch device and backend before model execution."""
    backend = str(info["backend"]).lower()
    device = "cpu" if backend == "cpu" else "cuda"
    request = f"; requested={requested}" if requested is not None else ""
    print(f"DEVICE {operation}: {device}; backend={backend.upper()}; "
          f"name={info.get('device_name', 'unknown')}{request}", flush=True)
    if operation in {"final training", "pilot", "experiment"} and backend == "cpu":
        print("WARNING: CPU training can be slow; a CUDA or ROCm GPU is strongly recommended.",
              flush=True)


def probe_accelerator() -> dict[str, object]:
    """Run one kernel so device detection alone cannot produce a false pass."""
    info = accelerator_info()
    try:
        probe = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = (probe * 2).cpu()
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError(
            f"{info['backend'].upper()} detected {info['device_name']} as "
            f"{info['architecture']}, but a device kernel failed. Check the runtime "
            "build and GPU architecture configuration before training."
        ) from error
    if result.tolist() != [2.0, 4.0, 6.0]:
        raise RuntimeError("The accelerator probe returned incorrect numerical results.")
    return info


def run_numerical_smoke(matrix_size: int = 256, sequence_length: int = 32) -> dict[str, object]:
    """Compare representative transformer operations against CPU reference values."""
    if matrix_size < 8 or sequence_length < 4:
        raise ValueError("matrix_size and sequence_length are too small for the smoke test.")
    info = probe_accelerator()
    started = time.monotonic()
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    inner_size = matrix_size // 2
    output_size = max(matrix_size // 4, 4)
    left_cpu = torch.linspace(
        -1.0, 1.0, matrix_size * inner_size, dtype=torch.float32,
    ).reshape(matrix_size, inner_size)
    right_cpu = torch.linspace(
        0.5, -0.5, inner_size * output_size, dtype=torch.float32,
    ).reshape(inner_size, output_size)
    expected_matrix = left_cpu @ right_cpu
    actual_matrix = (left_cpu.cuda() @ right_cpu.cuda()).cpu()
    torch.testing.assert_close(actual_matrix, expected_matrix, rtol=2e-4, atol=2e-4)

    input_cpu = torch.linspace(-2.0, 2.0, 8 * 32, dtype=torch.float32).reshape(8, 32)
    target_cpu = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    cpu_model = torch.nn.Linear(32, 3)
    gpu_model = copy.deepcopy(cpu_model).cuda()
    cpu_loss = functional.cross_entropy(cpu_model(input_cpu), target_cpu)
    gpu_loss = functional.cross_entropy(gpu_model(input_cpu.cuda()), target_cpu.cuda())
    cpu_loss.backward()
    gpu_loss.backward()
    torch.cuda.synchronize()
    if not math.isfinite(gpu_loss.item()):
        raise FloatingPointError("The accelerator produced a nonfinite training loss.")
    torch.testing.assert_close(gpu_loss.cpu(), cpu_loss, rtol=2e-4, atol=2e-4)
    for cpu_parameter, gpu_parameter in zip(cpu_model.parameters(), gpu_model.parameters()):
        if gpu_parameter.grad is None or not torch.isfinite(gpu_parameter.grad).all():
            raise FloatingPointError("The accelerator produced missing or nonfinite gradients.")
        torch.testing.assert_close(
            gpu_parameter.grad.cpu(), cpu_parameter.grad, rtol=3e-4, atol=3e-4,
        )

    attention_shape = (2, 4, sequence_length, 16)
    values = torch.linspace(-0.7, 0.7, math.prod(attention_shape), dtype=torch.float32)
    query_cpu = values.reshape(attention_shape).requires_grad_()
    key_cpu = torch.flip(query_cpu.detach(), dims=(-2,)).requires_grad_()
    value_cpu = torch.sin(query_cpu.detach()).requires_grad_()
    query_gpu = query_cpu.detach().cuda().requires_grad_()
    key_gpu = key_cpu.detach().cuda().requires_grad_()
    value_gpu = value_cpu.detach().cuda().requires_grad_()
    attention_cpu = functional.scaled_dot_product_attention(query_cpu, key_cpu, value_cpu)
    attention_gpu = functional.scaled_dot_product_attention(query_gpu, key_gpu, value_gpu)
    attention_cpu.square().mean().backward()
    attention_gpu.square().mean().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(attention_gpu.cpu(), attention_cpu, rtol=5e-4, atol=5e-4)
    for cpu_tensor, gpu_tensor in (
        (query_cpu, query_gpu), (key_cpu, key_gpu), (value_cpu, value_gpu),
    ):
        if gpu_tensor.grad is None or not torch.isfinite(gpu_tensor.grad).all():
            raise FloatingPointError("SDPA produced missing or nonfinite gradients.")
        torch.testing.assert_close(
            gpu_tensor.grad.cpu(), cpu_tensor.grad, rtol=8e-4, atol=8e-4,
        )

    return {
        **info,
        "matrix_size": matrix_size,
        "matrix_max_abs_error": float((actual_matrix - expected_matrix).abs().max()),
        "linear_loss": float(gpu_loss.detach()),
        "attention_shape": list(attention_shape),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "elapsed_seconds": time.monotonic() - started,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    args = parser.parse_args()
    report = run_numerical_smoke(args.matrix_size, args.sequence_length)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
