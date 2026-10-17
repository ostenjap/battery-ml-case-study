"""Look at the NASA .mat files before writing any model code.

WHY THIS FILE EXISTS
--------------------
The previous loader assumed `pd.read_csv` on columns `cycle,capacity`. NASA
ships nested MATLAB structs. That loader could never have run, and nobody found
out until day 1 of a 3-day budget (devlog, 2026-07-28).

So this script *probes*. It walks whatever is actually in the file and reports
the field names it finds, rather than indexing fields it expects to exist. If
the schema differs from what anyone assumed, the output says so instead of
raising a KeyError forty lines into a training script.

    python src/explore_nasa.py

Prints: the record types present, the fields on each, and a per-cell summary of
the discharge capacity series. Writes a tidy CSV per cell plus one overview
figure. Extracts nothing into the model -- that is the next step, deliberately.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")


# The 24 C room-temperature group (D-001). The full download ships 38 .mat files
# across six aging campaigns at several ambient temperatures; mixing groups would
# mean fitting one rate constant across incompatible thermal conditions. Note
# that B0025-B0028 are physically duplicated across two campaign folders, so an
# unfiltered rglob also yields the same cell twice.
CELLS_24C = ("B0005", "B0006", "B0007", "B0018")


def find_mat_files(cells: tuple[str, ...] | None = CELLS_24C) -> list[Path]:
    """Locate the .mat files, restricted to one temperature group by default.

    The zip nests them a few directories deep and the layout differs between
    mirrors, so search rather than hard-code a path. Pass `cells=None` to get
    everything (deduplicated by stem) when exploring the other campaigns.
    """
    found = sorted(RAW.rglob("*.mat"))
    if not found:
        # Fetch rather than instruct. A script that stops to tell the user to go
        # run a different script is a step that can be got wrong, and this one
        # was: the nested archives stayed zipped and the directory looked full.
        print(f"No .mat files under {RAW.resolve()} -- fetching the dataset now.\n")
        from fetch_nasa import main as fetch
        if fetch() != 0:
            sys.exit("Automatic download failed. Run: python src/fetch_nasa.py")
        found = sorted(RAW.rglob("*.mat"))
        if not found:
            sys.exit("Download reported success but no .mat files appeared.")

    if cells is None:
        seen: dict[str, Path] = {}
        for p in found:                      # first occurrence wins
            seen.setdefault(p.stem, p)
        return list(seen.values())

    by_stem = {}
    for p in found:
        by_stem.setdefault(p.stem, p)
    missing = [c for c in cells if c not in by_stem]
    if missing:
        sys.exit(f"Expected cells not found: {missing}\nPresent: {sorted(by_stem)}")
    return [by_stem[c] for c in cells]


def unwrap(x):
    """scipy wraps almost everything in 0-d or 1-element object arrays.

    Unwrapping is not cosmetic: without it, `record.type` is
    array(['discharge'], dtype='<U9') rather than the string 'discharge', and
    every comparison downstream silently fails instead of erroring.
    """
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    if isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "US":
        return str(x.item())
    if isinstance(x, np.ndarray) and x.dtype.kind in "US":
        return str(x)
    return x


def fields_of(obj) -> list[str]:
    """Field names of a scipy mat_struct, or [] if it isn't one."""
    return list(getattr(obj, "_fieldnames", []))


def to_datetime(tvec) -> datetime | None:
    """NASA stores each cycle's start as a MATLAB datevec:
    [year, month, day, hour, minute, second-with-fraction].

    This timestamp is the single most important field in the file for this
    project. The SEI law runs in wall-clock time, the data is indexed by cycle,
    and the two are not proportional because the cells sat idle for days between
    test runs (D-008). Without this field the physics residual is incoherent.
    """
    try:
        v = np.asarray(unwrap(tvec), dtype=float).ravel()
        if v.size < 6:
            return None
        sec = float(v[5])
        return datetime(int(v[0]), int(v[1]), int(v[2]),
                        int(v[3]), int(v[4])) + timedelta(seconds=sec)
    except Exception:
        return None


def probe(path: Path) -> None:
    """Report the schema of one file without assuming any of it."""
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]

    print(f"\n{'=' * 78}")
    print(f"{path.name}   top-level keys: {keys}")
    print("=" * 78)

    for key in keys:
        top = mat[key]
        cycles = getattr(top, "cycle", None)
        if cycles is None:
            print(f"  '{key}' has no .cycle field; fields = {fields_of(top)}")
            continue

        cycles = np.atleast_1d(cycles)
        print(f"  '{key}'.cycle -> {len(cycles)} records")

        # Which record types exist, and what fields does each carry?
        seen: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        for rec in cycles:
            rtype = unwrap(getattr(rec, "type", "?"))
            counts[rtype] = counts.get(rtype, 0) + 1
            if rtype not in seen:
                seen[rtype] = fields_of(getattr(rec, "data", None))

        print(f"  record types: {counts}")
        for rtype, flds in seen.items():
            print(f"    {rtype:<12} data fields: {flds}")

        # Show the shape of one discharge record so array lengths are concrete.
        for rec in cycles:
            if unwrap(getattr(rec, "type", "")) == "discharge":
                d = rec.data
                print(f"    first discharge  ambient={unwrap(rec.ambient_temperature)} C"
                      f"  start={to_datetime(rec.time)}")
                for f in fields_of(d):
                    arr = np.atleast_1d(np.asarray(unwrap(getattr(d, f))))
                    print(f"      {f:<22} shape={arr.shape} "
                          f"dtype={arr.dtype} first={arr.ravel()[0] if arr.size else '-'}")
                break


def extract_discharges(path: Path) -> pd.DataFrame:
    """One row per discharge cycle: the series the model will actually consume.

    Only discharge records carry a Capacity field -- charge and impedance
    records are skipped here, not because they are useless (impedance is the
    A4 stretch observable) but because capacity fade is what the brief asks to
    predict, and mixing record types is how off-by-one cycle indices happen.
    """
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    rows = []

    for key in keys:
        cycles = np.atleast_1d(getattr(mat[key], "cycle", []))
        for i, rec in enumerate(cycles):
            if unwrap(getattr(rec, "type", "")) != "discharge":
                continue
            d = rec.data
            cap = unwrap(getattr(d, "Capacity", None))
            if cap is None:
                continue
            cap = float(np.asarray(cap).ravel()[0])
            # NASA's docs say this field is Ah, but nothing enforces it -- a
            # mislabeled mAh source would read as a cell that lost 99.9% of its
            # capacity in one cycle. 18650 cells this dataset uses are rated
            # 1.8-2.2 Ah, so bound generously around that rather than assume.
            assert 0.5 < cap < 3.0, f"capacity {cap} outside plausible Ah range -- check units"

            temp = np.atleast_1d(np.asarray(
                unwrap(getattr(d, "Temperature_measured", np.nan)), dtype=float))

            rows.append({
                "record_index": i,
                "start": to_datetime(rec.time),
                "capacity_ah": cap,
                "ambient_c": float(unwrap(rec.ambient_temperature)),
                # Mean cell temperature over the discharge. This is what drives
                # the Arrhenius factor (D-010) -- nominal ambient is constant
                # and would make the residual degenerate.
                "temp_mean_c": float(np.nanmean(temp)),
                "temp_max_c": float(np.nanmax(temp)),
            })

    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    if df.empty:
        return df

    df["cycle"] = np.arange(1, len(df) + 1)
    # Elapsed hours from the first discharge -- the physics time base (D-008).
    df["elapsed_h"] = (df["start"] - df["start"].iloc[0]).dt.total_seconds() / 3600.0
    df["soh"] = df["capacity_ah"] / df["capacity_ah"].iloc[0]
    df["delta_capacity"] = df["capacity_ah"].diff()
    # Rest before this cycle. Long rests are where regeneration comes from, so
    # this column is the feature the A3 reversible channel is built on.
    df["rest_h"] = df["elapsed_h"].diff()
    return df


def validate(df: pd.DataFrame, name: str) -> None:
    """Missing values and outliers, checked rather than assumed.

    Two different failure modes get conflated if this isn't split up:
    - a NaN mid-series would mean NASA actually skipped logging a cycle (a
      real gap worth knowing about), vs. the row-0 NaN from .diff() having
      nothing before it to subtract (structural, expected, harmless).
    - a large |delta_capacity| could be a sensor glitch, or it could be a
      genuine regeneration event -- the exact signal this project studies
      (section 5). A generic z-score/IQR filter would silently strip real
      physics, so this only flags for a human to look at; nothing is dropped.
    """
    if df.empty:
        return

    for col in ("delta_capacity", "rest_h"):
        mid_series_nulls = df[col].iloc[1:].isna().sum()
        assert mid_series_nulls == 0, (
            f"{name}: {mid_series_nulls} unexpected NaN(s) in {col} after row 0 "
            "-- NASA may have skipped logging a cycle"
        )

    # Time is monotonic by construction (sorted by `start` above); a negative
    # rest would mean two records shared or reversed a timestamp.
    bad_rest = df["rest_h"].iloc[1:] < 0
    assert not bad_rest.any(), f"{name}: {bad_rest.sum()} cycle(s) with negative rest_h"

    # Typical single-cycle fade for these cells is a few mAh; a jump 10x that
    # is worth a human look, but the largest known regeneration events (D-0xx,
    # section 5) are real and must not be discarded automatically.
    jump_threshold = 0.05  # Ah
    jumps = df[df["delta_capacity"].abs() > jump_threshold]
    if not jumps.empty:
        print(f"  {name}: {len(jumps)} cycle(s) with |delta_capacity| > "
              f"{jump_threshold} Ah -- inspect, do not auto-drop:")
        for _, row in jumps.iterrows():
            print(f"    cycle {int(row['cycle']):>4}  "
                  f"delta_capacity={row['delta_capacity']:+.4f} Ah")


def summarise(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"  {name}: no discharge records with a Capacity field")
        return

    regen = int((df["delta_capacity"] > 0).sum())
    print(f"\n  {name}")
    print(f"    discharge cycles     {len(df)}")
    print(f"    capacity   {df['capacity_ah'].iloc[0]:.4f} -> "
          f"{df['capacity_ah'].iloc[-1]:.4f} Ah   "
          f"(SOH {df['soh'].iloc[-1] * 100:.1f}%)")
    print(f"    elapsed              {df['elapsed_h'].iloc[-1]:.1f} h "
          f"({df['elapsed_h'].iloc[-1] / 24:.1f} days)")
    print(f"    cell temp     mean {df['temp_mean_c'].mean():.1f} C   "
          f"range {df['temp_mean_c'].min():.1f}-{df['temp_mean_c'].max():.1f} C")
    print(f"    longest rest         {df['rest_h'].max():.1f} h")
    # The headline number for this project: every positive step is capacity the
    # cell recovered while resting. A strictly monotone model cannot express
    # these, which is precisely why rung A2 is expected to fail.
    print(f"    REGENERATION steps   {regen} / {len(df) - 1} "
          f"({100 * regen / max(len(df) - 1, 1):.0f}% of cycles gained capacity)")


def plot(frames: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for name, df in frames.items():
        if df.empty:
            continue
        axes[0].plot(df["cycle"], df["capacity_ah"], lw=1.3, marker=".", ms=2.5, label=name)
        axes[1].plot(df["elapsed_h"], df["capacity_ah"], lw=1.3, marker=".", ms=2.5, label=name)
        axes[2].plot(df["cycle"], df["temp_mean_c"], lw=1.1, label=name)

    axes[0].set(xlabel="cycle index", ylabel="capacity (Ah)", title="Fade vs cycle")
    axes[1].set(xlabel="elapsed time (h)", title="Fade vs wall-clock time (D-008)")
    axes[2].set(xlabel="cycle index", ylabel="mean cell temp (C)",
                title="Temperature drives Arrhenius (D-010)")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "nasa_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\n  figure written: {out}")


def main() -> int:
    files = find_mat_files()
    print(f"Found {len(files)} .mat file(s) under {RAW.resolve()}")

    # Probe only the first file in detail -- the rest share its schema, and
    # dumping all of them buries the useful output.
    probe(files[0])

    print(f"\n{'=' * 78}")
    print("DISCHARGE CAPACITY SERIES")
    print("=" * 78)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    frames = {}
    for path in files:
        name = path.stem
        df = extract_discharges(path)
        frames[name] = df
        summarise(name, df)
        if not df.empty:
            out = PROCESSED / f"{name}_discharge.csv"
            df.to_csv(out, index=False)
            print(f"    written -> {out}")

    plot(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
