"""Shared argument handling for the U-Net entry points."""

import argparse
import importlib.util
from pathlib import Path

from ..config import load_config


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Path to config.local.yaml")
    parser.add_argument("--run-name", help="New local directory name (default: unique timestamp)")
    return parser


def require_torch(parser):
    if importlib.util.find_spec("torch") is None:
        parser.error("U-Net requires PyTorch installed separately. See the CPU and CUDA setup in README.md.")


def load_cli_config(parser, path):
    try:
        return load_config(path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
