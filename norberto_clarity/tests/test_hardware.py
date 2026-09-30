from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from clarity.hardware import accelerator_info, log_selected_device, run_numerical_smoke


def test_accelerator_info_requires_available_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CUDA or ROCm accelerator"):
        accelerator_info()


def test_accelerator_info_distinguishes_rocm(monkeypatch):
    properties = SimpleNamespace(
        name="Test Radeon", gcnArchName="gfx1030", total_memory=8 * 2**30,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(torch.version, "hip", "6.4-test")
    monkeypatch.setattr(torch.version, "cuda", None)
    info = accelerator_info()
    assert info["backend"] == "rocm"
    assert info["architecture"] == "gfx1030"
    assert info["runtime_version"] == "6.4-test"
    assert info["total_memory_mib"] == 8192


def test_device_log_names_cpu_cuda_and_rocm(capsys):
    log_selected_device({"backend": "cpu", "device_name": "CPU"}, operation="final training")
    cpu_log = capsys.readouterr().out
    assert "DEVICE final training: cpu; backend=CPU; name=CPU" in cpu_log
    assert "CUDA or ROCm GPU is strongly recommended" in cpu_log

    log_selected_device({"backend": "rocm", "device_name": "Radeon"}, operation="test prediction",
                        requested="auto")
    rocm_log = capsys.readouterr().out
    assert "DEVICE test prediction: cuda; backend=ROCM; name=Radeon; requested=auto" in rocm_log
    assert "strongly recommended" not in rocm_log

    log_selected_device({"backend": "cuda", "device_name": "NVIDIA"}, operation="test prediction",
                        requested="cuda")
    cuda_log = capsys.readouterr().out
    assert "DEVICE test prediction: cuda; backend=CUDA; name=NVIDIA; requested=cuda" in cuda_log


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or ROCm device")
def test_accelerator_numerical_smoke():
    report = run_numerical_smoke(matrix_size=32, sequence_length=8)
    assert report["status"] == "passed"
    assert report["matrix_max_abs_error"] < 1e-3
    assert report["peak_allocated_mib"] > 0
