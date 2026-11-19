"""Run a saved laser heatmap U-Net on one local image."""

import csv
from pathlib import Path

import cv2

from .benchmarking import read_image
from .unet.cli import load_cli_config, parser_for, require_torch


def main(argv=None):
    parser = parser_for(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", required=True, help="Filename relative to configured images_dir")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args(argv)
    require_torch(parser)
    from .unet.data import image_tensor
    from .unet.inference import Predictor
    from .unet.plots import heatmap_figure
    from .unet.training import _new_run

    config = load_cli_config(parser, args.config)
    try:
        predictor = Predictor(args.checkpoint, args.device)
        if [config.width, config.height] != predictor.original_size:
            raise ValueError("The configured original image size differs from the checkpoint")
        image = read_image(args.image, config)
        coordinates, confidence, heatmap, duration = predictor.predict(image)
        rgb = image_tensor(image, predictor.profile.image_size).permute(1, 2, 0)
        output = _new_run(args.checkpoint.resolve().parent / "inferences", args.run_name)
        with (output / "prediction.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image", "device", "x1", "y1", "x2", "y2", "confidence1", "confidence2", "total_ns"))
            writer.writerow((args.image, str(predictor.device), *coordinates, *confidence, duration))
        heatmap_figure(rgb, heatmap, coordinates=coordinates, original_size=image.shape[1::-1]).savefig(
            output / "prediction.png",
        )
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
