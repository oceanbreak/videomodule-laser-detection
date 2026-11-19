"""Execute all shipped notebooks with a tiny local dataset."""

import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from nbclient import NotebookClient
from PIL import Image


@pytest.fixture
def notebook_workspace(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "config.local.yaml").write_text("""
data:
  annotations_csv: lasers.csv
  setups_csv: setups.csv
  images_dir: images
image: {width: 32, height: 24}
output: {directory: results/inspection}
calibration: {training_sizes: [2], repetitions: 1, contamination_training_size: 2}
detection: {width: 16, height: 12, warmup_frames: 1}
""", encoding="utf-8")
    rows = ["image;x1;y1;x2;y2"]
    for index in range(3):
        image = Image.new("RGB", (32, 24))
        image.paste("red", (4, 4 + 4 * index, 8, 8 + 4 * index))
        image.paste("red", (20, 8 + 2 * index, 24, 12 + 2 * index))
        name = f"frame-{index}.png"
        image.save(tmp_path / "images" / name)
        rows.append(f"{name};5.5;{5.5 + 4 * index};21.5;{9.5 + 2 * index}")
    (tmp_path / "lasers.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "setups.csv").write_text("filename;setup\nframe-0.png;A\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("name", [
    "01_dataset_inspection.ipynb",
    "02_geometry_calibration.ipynb",
    "03_detection_baselines.ipynb",
])
def test_notebook_executes_on_synthetic_data(name, notebook_workspace):
    source = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    # Use the interpreter selected for pytest, even with other user kernels installed.
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=kernel, timeout=90)
    client.execute(cwd=str(notebook_workspace / "notebooks"), cleanup_kc=True)
