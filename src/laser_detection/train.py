"""Train a laser heatmap U-Net on local data."""

from pathlib import Path

from .data import load_dataset
from .unet.cli import load_cli_config, parser_for, require_torch


def main(argv=None):
    parser = parser_for(__doc__)
    parser.add_argument("--profile", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path, help="Existing run's last.pt checkpoint")
    args = parser.parse_args(argv)
    require_torch(parser)
    from .unet.training import train_run

    config = load_cli_config(parser, args.config)
    try:
        run = train_run(load_dataset(config), config, getattr(config.unet, args.profile),
                        run_name=args.run_name, resume=args.resume)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
