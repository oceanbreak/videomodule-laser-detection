"""Inspect optional PyTorch and verify CUDA forward and backward execution."""

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
import platform
import sys


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


def _cuda_smoke_test(torch, index):
    device = torch.device("cuda", index)
    layer = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False).to(device)
    with torch.no_grad():
        layer.weight.fill_(0.125)
    sample = torch.ones((2, 1, 8, 8), device=device, requires_grad=True)

    output = layer(sample)
    loss = output.square().mean()
    loss.backward()
    # CUDA can report errors asynchronously. Synchronize before declaring success.
    torch.cuda.synchronize(device)

    if tuple(output.shape) != (2, 2, 8, 8):
        raise RuntimeError(f"unexpected forward output shape {tuple(output.shape)}")
    for label, tensor in (
        ("output", output), ("loss", loss),
        ("input gradient", sample.grad), ("weight gradient", layer.weight.grad),
    ):
        if tensor is None or tensor.device != device:
            raise RuntimeError(f"{label} missing or not on {device}")
        if not torch.isfinite(tensor).all().item() or not torch.count_nonzero(tensor).item():
            raise RuntimeError(f"{label} must be finite and nonzero")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA cannot be tested")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    for package in ("numpy", "pandas", "opencv-python", "Pillow", "PyYAML", "plotly", "matplotlib", "jupyterlab"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            continue
    stage = "PyTorch import"
    try:
        torch = _optional_import("torch")
        if torch is None:
            print("PyTorch: not installed (optional)")
            print("CUDA available: False")
            print("CUDA forward/backward: SKIPPED")
            return 1 if args.require_cuda else 0

        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA build: {torch.version.cuda or 'none'}")
        stage = "torchvision import"
        torchvision = _optional_import("torchvision")
        print(f"torchvision: {torchvision.__version__ if torchvision else 'not installed (optional)'}")

        stage = "CUDA discovery"
        available = torch.cuda.is_available()
        print(f"CUDA available: {available}")
        if not available:
            print("CUDA forward/backward: SKIPPED")
            return 1 if args.require_cuda else 0

        count = torch.cuda.device_count()
        if args.device >= count:
            raise RuntimeError(f"CUDA device index {args.device} out of range ({count} visible)")
        name = torch.cuda.get_device_name(args.device)
        major, minor = torch.cuda.get_device_capability(args.device)
        print(f"CUDA device {args.device}: {name}, compute capability {major}.{minor}")
        architectures = getattr(torch.cuda, "get_arch_list", lambda: [])()
        print(f"PyTorch CUDA build architectures: {', '.join(architectures) or 'unknown'}")
        free, total = torch.cuda.mem_get_info(args.device)
        print(f"CUDA memory: {free / 2**20:.0f} MiB free / {total / 2**20:.0f} MiB total")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        stage = "CUDA forward/backward smoke test"
        _cuda_smoke_test(torch, args.device)
    except Exception as error:
        print(f"{stage}: FAILED ({type(error).__name__}: {error})", file=sys.stderr)
        return 1

    print("CUDA forward/backward: PASSED (Conv2d, loss, input and weight gradients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
