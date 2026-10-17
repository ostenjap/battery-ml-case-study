"""Load the tidy per-cell CSVs that explore_nasa.py writes.

WHY THIS FILE WAS REWRITTEN
---------------------------
The original version did `pd.read_csv(path)[["cycle", "capacity"]]` against
`data/raw/*.csv`. Two things were wrong with that, and both were fatal:

  - NASA ships nested MATLAB structs, not CSVs. `data/raw` contains zero .csv
    files, so the `RAW_DIR.glob("*.csv")` in every caller found nothing.
  - The tidy CSVs `explore_nasa.py` writes name the column `capacity_ah`, not
    `capacity`. So the loader raised

        KeyError: "['capacity'] not in index"

    on the only data in the repo -- while pinn_model.py, baseline_arima.py and
    baseline_lstm.py all imported it.

That `capacity` / `capacity_ah` slip caused two independent bugs: this loader,
and a notebook cleaning cell that silently wrote dimensionless SOH into a
column named `delta_capacity` because `if "capacity" in df.columns` was False
and the else-branch took over. The unit belongs in the name. Nothing here
accepts a bare `capacity` column, so that slip cannot recur silently.

WHY raise AND NOT assert
------------------------
This module is the boundary between disk and model. `python -O` strips asserts,
and a loader whose integrity checks vanish under an optimisation flag is worse
than one with no checks, because it looks safe. `extract_discharges` uses
asserts because it is a probe run by hand; this runs under training.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# `elapsed_h` is required, not optional: the SEI law runs in wall-clock time and
# cycle index is not proportional to it (rests up to 310 h -- D-008). A frame
# without it cannot support a coherent physics residual.
REQUIRED_COLUMNS = ("cycle", "capacity_ah", "elapsed_h")

# Same bound as extract_discharges: 18650 cells rated 1.8-2.2 Ah. Catches a
# mAh-scaled file, which would otherwise read as a cell that lost 99.9% of its
# capacity in one cycle.
CAPACITY_AH_MIN = 0.5
CAPACITY_AH_MAX = 3.0


def load_nasa_capacity_fade(csv_path: Path) -> pd.DataFrame:
    """Read one cell's discharge series, validated.

    Returns every column in the file, not a two-column subset. The original
    subsetted to `["cycle", "capacity"]` and threw away the timestamps, the
    temperatures and the rest gaps -- i.e. everything the physics needs.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Run `python src/explore_nasa.py` to "
            f"generate the per-cell CSVs in {PROCESSED_DIR}."
        )

    df = pd.read_csv(csv_path)

    # A one-column parse means the delimiter is wrong -- a semicolon-separated
    # file read with the default comma yields a single fused column and then
    # fails much later with an unrelated-looking error.
    if df.shape[1] == 1:
        raise ValueError(
            f"{csv_path.name} parsed as a single column ({df.columns[0]!r}). "
            "The delimiter is probably not a comma."
        )

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv_path.name} is missing required column(s) {missing}. "
            f"Found: {sorted(df.columns)}. Note the capacity column must be "
            "named `capacity_ah` -- a bare `capacity` is rejected because it "
            "does not state its unit."
        )

    # Drop rows missing a REQUIRED value only. A blanket .dropna() would delete
    # row 0, whose `delta_capacity` and `rest_h` are NaN purely because .diff()
    # has nothing before the first cycle to subtract. Row 0 is the only row
    # where elapsed_h == 0 and soh == 1.0, so it anchors the hard initial
    # condition that pinn_model bakes into its functional form. Losing it
    # silently biases every fit.
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    if len(df) < before:
        print(f"  {csv_path.name}: dropped {before - len(df)} row(s) missing "
              f"one of {REQUIRED_COLUMNS}")

    if df.empty:
        raise ValueError(f"{csv_path.name} has no usable rows after validation.")

    df = df.sort_values("cycle").reset_index(drop=True)

    cap = df["capacity_ah"]
    out_of_range = df[(cap <= CAPACITY_AH_MIN) | (cap >= CAPACITY_AH_MAX)]
    if not out_of_range.empty:
        raise ValueError(
            f"{csv_path.name}: {len(out_of_range)} capacity value(s) outside "
            f"({CAPACITY_AH_MIN}, {CAPACITY_AH_MAX}) Ah -- observed "
            f"{cap.min():.4f} to {cap.max():.4f}. Check the units."
        )

    if not df["elapsed_h"].is_monotonic_increasing:
        raise ValueError(
            f"{csv_path.name}: elapsed_h is not monotonic after sorting by "
            "cycle -- cycle index and wall-clock time disagree."
        )

    return df


def train_test_split_by_cycle(df: pd.DataFrame, train_frac: float = 0.7):
    """Split temporally: the first `train_frac` of cycles train, the rest test.

    No shuffle, by construction -- this is a time series, and a random split
    would let the model interpolate between training points that straddle every
    test point, reporting an accuracy it would never achieve forward in time.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    # The caller may have filtered or reordered since loading. Splitting an
    # unsorted frame by position would leak later cycles into train.
    if not df["cycle"].is_monotonic_increasing:
        raise ValueError("df must be sorted by cycle before splitting")

    split_idx = int(len(df) * train_frac)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError(
            f"train_frac={train_frac} on {len(df)} rows leaves one side empty"
        )

    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def processed_csvs() -> list[Path]:
    """The per-cell CSVs, in cell order. Callers used to glob RAW_DIR for
    *.csv, which is always empty -- the raw download is .mat."""
    return sorted(PROCESSED_DIR.glob("B0*_discharge.csv"))


if __name__ == "__main__":
    files = processed_csvs()
    print(f"Found {len(files)} per-cell CSV(s) in {PROCESSED_DIR}")
    if not files:
        print("None yet. Run: python src/explore_nasa.py")
        raise SystemExit(1)

    for path in files:
        df = load_nasa_capacity_fade(path)
        train, test = train_test_split_by_cycle(df)
        print(f"  {path.stem:<22} {len(df):>4} cycles  "
              f"{df['capacity_ah'].iloc[0]:.4f} -> {df['capacity_ah'].iloc[-1]:.4f} Ah  "
              f"| split {len(train)}/{len(test)}  "
              f"| train ends at {train['elapsed_h'].iloc[-1]:.0f} h")
