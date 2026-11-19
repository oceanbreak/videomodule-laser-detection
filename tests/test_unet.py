from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from laser_detection.config import load_config
from laser_detection.data import load_dataset
from laser_detection.detectors.base import scale_centers
from laser_detection.evaluate_unet import main as evaluate_main
from laser_detection.infer import main as infer_main
from laser_detection.train import main as train_main
from laser_detection.unet.data import HeatmapDataset, make_heatmap, training_split
from laser_detection.unet.inference import Predictor
from laser_detection.unet.model import UNet, heatmap_loss, peak_centers
from laser_detection.unet.settings import Profile
from laser_detection.unet.training import read_checkpoint, select_device


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_targets_and_pixel_centers():
    profile = Profile(image_size=(20, 16), radius=2, sigma=2)
    points = [[0, 0], [10, 8]]
    mapped = scale_centers(points, (20, 16), (40, 32))
    np.testing.assert_allclose(mapped, [[0.5, 0.5], [20.5, 16.5]])
    np.testing.assert_allclose(scale_centers(mapped, (40, 32), (20, 16)), points)
    binary = make_heatmap(points, profile)
    assert binary.dtype == np.float32 and binary.shape == (16, 20) and binary.sum() == 19
    assert binary[0, 0] == binary[8, 10] == 1
    gaussian = replace(profile, target="gaussian", loss="weighted_mse")
    target = make_heatmap(points, gaussian)
    assert target[0, 0] == target[8, 10] == 1
    assert target[8, 12] == pytest.approx(np.exp(-0.5))
    for settings in (profile, gaussian):
        assert not make_heatmap([np.nan] * 4, settings).any()


@pytest.mark.parametrize("loss", ["bce", "dice", "bce_dice", "weighted_mse"])
def test_cpu_forward_backward_and_empty_targets(loss):
    model = UNet(2)
    inputs = torch.rand(2, 3, 17, 19, requires_grad=True)
    logits = model(inputs)
    assert logits.shape == (2, 1, 17, 19)
    target = torch.zeros_like(logits)
    target[1, :, 3:5, 3:5] = 1
    profile = Profile(loss=loss, target="gaussian" if loss == "weighted_mse" else "binary_circle")
    value = heatmap_loss(logits, target, profile)
    value.backward()
    assert torch.isfinite(value) and value > 0
    assert torch.isfinite(inputs.grad).all()
    assert model.output.weight.grad is not None


def test_peak_detection_and_nms():
    probability = np.zeros((16, 20), dtype=np.float32)
    probability[2, 12], probability[2, 14], probability[6, 4] = 0.9, 0.95, 0.8
    profile = Profile(image_size=(20, 16), nms_radius=3, confidence=0.75)
    coordinates, confidence = peak_centers(probability, profile, (40, 32))
    assert coordinates == (8.5, 12.5, 28.5, 4.5)
    assert confidence == pytest.approx((0.8, 0.95))
    assert np.isnan(peak_centers(probability, replace(profile, confidence=0.92), (40, 32))[0]).all()
    assert np.isnan(peak_centers(np.ones_like(probability), profile, (40, 32))[0]).all()


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "images").mkdir()
    rows = ["image;x1;y1;x2;y2"]
    for i in range(5):
        name = f"{i}.png"
        image = Image.new("RGB", (32, 24))
        image.paste("red", (4, 4, 8, 8))
        image.paste("red", (20, 8, 24, 12))
        image.save(tmp_path / "images" / name)
        rows.append(f"{name};5.5;5.5;21.5;9.5")
    (tmp_path / "lasers.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "setups.csv").write_text("filename;setup\n0.png;A\n")
    path = tmp_path / "config.local.yaml"
    path.write_text("""data:
  annotations_csv: lasers.csv
  setups_csv: setups.csv
  images_dir: images
image: {width: 32, height: 24}
output: {directory: results/inspection}
detection: {width: 16, height: 12}
unet:
  cpu: {image_size: [16, 16], base_channels: 2, epochs: 1, batch_size: 1}
""")
    return path


def test_saved_split_series_and_exploratory_protocol(config_path):
    config = load_config(config_path)
    frame = load_dataset(config)
    split = training_split(frame, config, 2)
    assert set(split["split"]) == {"train", "test", "validation"}
    assert split["protocol"].eq("exploratory").all()
    split_path = config_path.parent / "dataset_split.csv"
    split["series_id"] = split["split"]
    split.to_csv(split_path, index=False)
    preserved = training_split(frame, config, 99, saved=split_path)
    assert preserved["split"].equals(split["split"])
    assert preserved["protocol"].eq("capture_series").all()
    split["series_id"] = "one_video"
    split.to_csv(split_path, index=False)
    with pytest.raises(ValueError, match="series cannot cross splits"):
        training_split(frame, config, 99, saved=split_path)


def test_train_resume_infer_and_evaluate_cli(config_path, capsys):
    config = load_config(config_path)
    frame = load_dataset(config)
    image, target, meta = HeatmapDataset(frame, config, config.unet.cpu)[0]
    assert image.shape == (3, 16, 16) and target.shape == (1, 16, 16)
    assert float(image[0].max()) == 1 and not image[1:].any()
    assert meta == {"image": "0.png", "setup_id": "A"}
    cli = ["--config", str(config_path)]
    assert train_main([*cli, "--profile", "cpu", "--run-name", "synthetic"]) == 0
    run = config.unet_output_dir / "synthetic"
    assert read_checkpoint(run / "last.pt")["epoch"] == 1
    initial = capsys.readouterr()
    assert "Training" in initial.err and "Validation" in initial.err
    assert "Epoch 1: train_loss=" in initial.out and "val_loss=" in initial.out and "time=" in initial.out
    config_path.write_text(config_path.read_text().replace("epochs: 1", "epochs: 2, show_progress: false"))
    assert train_main([*cli, "--resume", str(run / "last.pt")]) == 0
    resumed = capsys.readouterr()
    assert "Training" not in resumed.err and "Validation" not in resumed.err
    assert "Epoch 2: train_loss=" in resumed.out
    state = read_checkpoint(run / "last.pt")
    assert "show_progress" not in state["profile"]
    assert state["epoch"] == 2 and len(state["history"]) == 2 and state["optimizer"]["state"]
    checkpoint = run / "best.pt"
    options = [*cli, "--checkpoint", str(checkpoint), "--device", "cpu"]
    assert Predictor(checkpoint, "cpu").profile.image_size == (16, 16)
    assert infer_main([*options, "--image", "0.png", "--run-name", "frame"]) == 0
    assert (run / "inferences/frame/prediction.png").is_file()
    assert evaluate_main([*options, "--run-name", "heldout"]) == 0
    evaluation = run / "evaluations/heldout"
    predictions = pd.read_csv(evaluation / "predictions.csv")
    assert set(predictions["method"]) == {"unet", "hsv_legacy"}
    assert predictions["image"].nunique() == 1
    assert predictions["split"].eq("test").all() and predictions["protocol"].eq("exploratory").all()
    timing = pd.read_csv(evaluation / "timings.csv")
    assert timing["device"].eq("cpu").all()
    assert dict(zip(timing["method"], timing["input_size"])) == {"unet": "16x16", "hsv_legacy": "16x12"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_unet_forward_backward():
    device = select_device("cuda")
    model = UNet(2).to(device)
    image = torch.ones(1, 3, 16, 16, device=device, requires_grad=True)
    model(image).square().mean().backward()
    torch.cuda.synchronize(device)
    assert image.grad is not None and torch.isfinite(image.grad).all()


@pytest.mark.parametrize("enabled", [True, False])
def test_epoch_progress_weights_incomplete_batch(monkeypatch, enabled):
    from laser_detection.unet import training

    bars = []

    class Progress:
        def __init__(self, **options):
            self.options = options
            self.updates = 0
            self.postfixes = []
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def set_postfix(self, *, loss, lr, refresh):
            self.postfixes.append((loss, lr, refresh))

        def update(self):
            self.updates += 1

    monkeypatch.setattr(training, "tqdm", Progress)
    monkeypatch.setattr(training, "heatmap_loss", lambda logits, target, profile: logits.mean())
    images = torch.tensor([[1.0], [3.0], [8.0]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, images, images), batch_size=2,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    profile = Profile(show_progress=enabled)
    for phase_optimizer in (optimizer, None):
        average = training._epoch(
            model, loader, profile, torch.device("cpu"), epoch=3, learning_rate=0.001,
            optimizer=phase_optimizer,
        )
        assert average == 4

    assert len(bars) == 2
    for phase, bar in zip(("Training", "Validation"), bars):
        assert bar.options["desc"] == f"Epoch 3 · {phase}"
        assert bar.options["total"] == 2
        assert bar.options["disable"] is not enabled
        assert bar.closed
        assert bar.updates == (2 if enabled else 0)
        assert bar.postfixes == ([("2.000000", "0.001", False), ("4.000000", "0.001", False)] if enabled else [])
