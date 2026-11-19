from types import SimpleNamespace

from laser_detection import check_environment


def test_cpu_run_can_skip_missing_pytorch(monkeypatch, capsys):
    monkeypatch.setattr(check_environment, "_optional_import", lambda _: None)
    assert check_environment.main([]) == 0
    assert "CUDA forward/backward: SKIPPED" in capsys.readouterr().out
    assert check_environment.main(["--require-cuda"]) == 1


def test_cuda_build_without_available_device_is_skipped(monkeypatch, capsys):
    torch = SimpleNamespace(
        __version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(check_environment, "_optional_import", {"torch": torch, "torchvision": None}.get)
    assert check_environment.main(["--require-cuda"]) == 1
    output = capsys.readouterr().out
    assert "PyTorch CUDA build: 12.6" in output
    assert "CUDA forward/backward: SKIPPED" in output


def test_cuda_availability_does_not_mask_smoke_failure(monkeypatch, capsys):
    torch = SimpleNamespace(
        __version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda _: "test device", get_device_capability=lambda _: (6, 1),
            get_arch_list=lambda: ["sm_60"], mem_get_info=lambda _: (512 * 2**20, 1024 * 2**20),
        ),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 9100)),
    )
    monkeypatch.setattr(check_environment, "_optional_import", {"torch": torch, "torchvision": None}.get)

    def unavailable_kernel(module, index):
        assert module is torch and index == 0
        raise RuntimeError("no kernel image is available")

    monkeypatch.setattr(check_environment, "_cuda_smoke_test", unavailable_kernel)
    assert check_environment.main(["--require-cuda"]) == 1
    result = capsys.readouterr()
    assert "compute capability 6.1" in result.out
    assert "PASSED" not in result.out
    assert "no kernel image is available" in result.err
