"""Read and validate the original CSVs in acquisition-list order."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

COORDINATES = ["x1", "y1", "x2", "y2"]


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, delimiter=";", strict=True)
            header = next(reader, [])
            missing = set(columns) - set(header)
            if missing:
                raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
            if len(set(header)) != len(header):
                raise ValueError("duplicate column names")
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"line {reader.line_num}: expected {len(header)} fields")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return pd.DataFrame(rows, columns=header)[columns]


def _unique_names(frame, column, path):
    names = frame[column]
    invalid = names.str.strip().eq("") | names.duplicated(keep=False)
    if invalid.any():
        raise ValueError(f"{path}: empty or duplicate {column}: {names[invalid].iloc[0]!r}")


def load_annotations(path: Path, width: int = 3072, height: int = 2048) -> pd.DataFrame:
    path = Path(path)
    frame = _read_csv(path, ["image", *COORDINATES])
    _unique_names(frame, "image", path)
    for column in COORDINATES:
        raw = frame[column].str.strip()
        missing = raw.str.lower().isin(["", "nan"])
        values = pd.to_numeric(raw.mask(missing), errors="coerce")
        upper = width if column.startswith("x") else height
        invalid = ~missing & (~np.isfinite(values) | (values < 0) | (values >= upper))
        if invalid.any():
            index = invalid.idxmax()
            raise ValueError(
                f"{path}: {frame.at[index, 'image']!r}, {column}={raw[index]!r}: "
                f"expected a number in [0, {upper}) or NaN"
            )
        frame[column] = values.astype(float)
    counts = frame[COORDINATES].isna().sum(axis=1)
    partial = ~counts.isin([0, 4])
    if partial.any():
        raise ValueError(f"{path}: {frame.loc[partial, 'image'].iloc[0]!r}: expected all four coordinates or none")
    # Normalize laser order at the data boundary: point 1 is always left.
    # For equal X, prefer the upper point first for deterministic ordering.
    swap = (frame["x1"] > frame["x2"]) | (
        (frame["x1"] == frame["x2"]) & (frame["y1"] > frame["y2"])
    )
    if swap.any():
        frame.loc[swap, COORDINATES] = frame.loc[
            swap, ["x2", "y2", "x1", "y1"]
        ].to_numpy()
    return frame


def load_setups(path: Path) -> pd.DataFrame:
    path = Path(path)
    frame = _read_csv(path, ["filename", "setup"])
    _unique_names(frame, "filename", path)
    ids = frame["setup"].str.strip()
    if ids.eq("").any():
        raise ValueError(f"{path}: empty setup identifier")
    frame["setup"] = ids
    return frame


def assign_setups(annotations: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    positions = pd.Index(annotations["image"]).get_indexer(setups["filename"])
    if (positions < 0).any():
        name = setups.loc[positions < 0, "filename"].iloc[0]
        raise ValueError(f"Setup boundary {name!r} is missing from annotations")
    if not len(positions) or positions[0] != 0:
        raise ValueError("The first setup boundary must match the first annotation row")
    if (np.diff(positions) <= 0).any():
        raise ValueError("Setup boundaries must be unique and follow annotation row order")
    result = annotations.copy()
    result["setup_id"] = np.repeat(setups["setup"].to_numpy(), np.diff([*positions, len(result)]))
    return result


def load_dataset(config: Config) -> pd.DataFrame:
    annotations = load_annotations(config.annotations_csv, config.width, config.height)
    setups = load_setups(config.setups_csv)
    try:
        return assign_setups(annotations, setups)
    except ValueError as error:
        raise ValueError(f"{config.setups_csv}: {error}") from error
