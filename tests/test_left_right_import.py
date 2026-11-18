"""Laser point ordering is normalized on CSV import."""

import pandas as pd

from laser_detection.data import load_annotations


def test_load_annotations_orders_left_to_right(tmp_path):
    source = tmp_path / "lasers.csv"
    source.write_text(
        "image;x1;y1;x2;y2\n"
        "reversed.png;900;200;300;400\n"
        "normal.png;100;300;600;200\n"
        "same_x.png;500;800;500;100\n"
        "missing.png;;;;\n",
        encoding="utf-8",
    )
    result = load_annotations(source)
    assert result["image"].tolist() == [
        "reversed.png", "normal.png", "same_x.png", "missing.png"
    ]
    assert result.loc[0, ["x1", "y1", "x2", "y2"]].tolist() == [300, 400, 900, 200]
    assert result.loc[1, ["x1", "y1", "x2", "y2"]].tolist() == [100, 300, 600, 200]
    assert result.loc[2, ["x1", "y1", "x2", "y2"]].tolist() == [500, 100, 500, 800]
    assert result.loc[3, ["x1", "y1", "x2", "y2"]].isna().all()
