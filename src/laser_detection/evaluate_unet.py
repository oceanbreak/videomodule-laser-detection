"""Compare a saved U-Net with HSV Legacy on a recorded split."""

from pathlib import Path

from .data import load_dataset
from .unet.cli import load_cli_config, parser_for, require_torch


def main(argv=None):
    parser = parser_for(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    require_torch(parser)
    from .unet.inference import Predictor, compare_checkpoint

    config = load_cli_config(parser, args.config)
    try:
        predictor = Predictor(args.checkpoint, args.device)
        output, tables = compare_checkpoint(
            load_dataset(config), config, predictor,
            run_name=args.run_name, partition=args.partition,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(tables["metrics"].to_string(index=False))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
