"""Write standalone charts and inspection tables to local disk."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import Config
from .statistics import dataset_summary, setup_summary


def export_inspection(frame, figures, config: Config) -> list[Path]:
    tables = {
        "dataset_summary.csv": dataset_summary(frame),
        "setup_summary.csv": setup_summary(frame),
        "geometry.csv": frame,
    }
    charts = {f"{name}.html": figure for name, figure in figures.items()}
    output = config.output_dir.resolve()
    inputs = {config.annotations_csv.resolve(), config.setups_csv.resolve()}
    destinations = [output / name for name in (*tables, *charts)]
    for path in destinations:
        if path.parent != output or path.resolve().parent != output or path.resolve() in inputs:
            raise ValueError(f"Unsafe export destination: {path}")
        if path.exists() and any(source.exists() and path.samefile(source) for source in inputs):
            raise ValueError(f"Export would overwrite an input CSV: {path}")

    output.mkdir(parents=True, exist_ok=True)
    for path in destinations:
        with NamedTemporaryFile(dir=output, delete=False) as file:
            temporary = Path(file.name)
        try:
            if path.name in tables:
                tables[path.name].to_csv(temporary, index=False, lineterminator="\n")
            else:
                charts[path.name].write_html(
                    temporary, include_plotlyjs=True, div_id=f"inspection-{path.stem}",
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return destinations
