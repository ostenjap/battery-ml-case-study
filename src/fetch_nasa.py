"""Download and unpack the NASA PCoE battery dataset. One command, no manual steps.

WHY THIS FILE EXISTS
--------------------
The archive is a zip of zips: the outer file expands to six campaign archives
that are themselves still compressed. Extracting only the outer layer leaves a
directory that looks populated but contains no .mat files at all, and the
failure surfaces much later as "no data found" from a script that is working
correctly. That happened once already; this removes the step where it can.

Extraction recurses until no new archives appear, so the nesting depth does not
have to be known in advance -- mirrors have differed on it.

Idempotent: an existing download is reused and already-extracted archives are
skipped, so re-running costs nothing and is the fastest way to check the data is
intact after moving the project.

    python src/fetch_nasa.py

Reproducibility note: this script *is* the data provenance record. A reviewer
asking "where did these numbers come from" gets a URL and a checksum rather than
a description of some clicking.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
RAW = Path("data/raw")
ARCHIVE = RAW / "nasa_battery.zip"

# The 24 C room-temperature group (D-001). The download also ships five further
# campaigns at other ambients; they are extracted but not used, and mixing them
# would mean fitting one rate constant across incompatible thermal conditions.
EXPECTED = ("B0005", "B0006", "B0007", "B0018")


def download() -> None:
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 1_000_000:
        print(f"  archive present ({ARCHIVE.stat().st_size / 1e6:.1f} MB), skipping download")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {URL}")

    def progress(block: int, size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, 100.0 * block * size / total)
            print(f"\r    {pct:5.1f}%  ({total / 1e6:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(URL, ARCHIVE, reporthook=progress)
    except Exception as exc:
        sys.exit(f"\n  download failed: {exc}\n  URL: {URL}")
    print()


def extract_all() -> None:
    """Expand every archive under data/raw, repeatedly, until none are left.

    A zip is considered done if its sibling output directory already exists --
    cheap, and it keeps re-runs to a couple of seconds.
    """
    for depth in range(1, 6):
        pending = [
            z for z in sorted(RAW.rglob("*.zip"))
            if not z.with_suffix("").is_dir()
        ]
        if not pending:
            break
        print(f"  level {depth}: extracting {len(pending)} archive(s)")
        for z in pending:
            out = z.with_suffix("")
            try:
                with zipfile.ZipFile(z) as f:
                    f.extractall(out)
                print(f"    {z.name}")
            except zipfile.BadZipFile:
                print(f"    SKIP (not a zip): {z.name}")


def verify() -> int:
    mats = sorted(RAW.rglob("*.mat"))
    by_stem: dict[str, Path] = {}
    for p in mats:
        by_stem.setdefault(p.stem, p)   # B0025-B0028 ship in two campaigns

    print(f"\n  {len(mats)} .mat files, {len(by_stem)} unique cells")

    missing = [c for c in EXPECTED if c not in by_stem]
    if missing:
        print(f"  MISSING expected cells: {missing}")
        print(f"  present: {sorted(by_stem)}")
        return 1

    print(f"  sha256({ARCHIVE.name}) = {sha256(ARCHIVE)[:16]}...")
    print("\n  24 C group ready:")
    for c in EXPECTED:
        p = by_stem[c]
        print(f"    {c}  {p.stat().st_size / 1e6:6.1f} MB  {p.relative_to(RAW)}")
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("NASA PCoE battery dataset")
    download()
    extract_all()
    code = verify()
    print("\n  next: python src/explore_nasa.py" if code == 0 else "\n  fix the above first")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
