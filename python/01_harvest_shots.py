"""Fetch shot-detail (and play-by-play) bulk season files — no per-game calls.

Source: shufinskiy/nba_data — pre-scraped stats.nba.com rows, one small
download per season instead of hundreds of rate-limited calls (see
../../docs/public-data-availability.md). The asset URL is read the same way
the package's own loader reads it: from list_data.txt on the repo. The same
archive naming covers the NBA (`shotdetail_YYYY`), the WNBA
(`wnba_shotdetail_YYYY`, 1997-present), and play-by-play (`nbastats_YYYY`).

Idempotent per dataset: skips any whose extracted CSV is already present.

Run: python python/01_harvest_shots.py [dataset ...]
     (default: shotdetail_2023 — the core study's season)
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/master/list_data.txt"
DEFAULT = ["shotdetail_2023"]


def asset_url(name: str) -> str:
    """Resolve a dataset name to its release asset URL via list_data.txt."""
    with urllib.request.urlopen(LIST_URL, timeout=30) as r:
        pairs = dict(line.split("=", 1) for line in
                     r.read().decode().splitlines() if "=" in line)
    return pairs[name]


def fetch(name: str) -> Path:
    """Download + extract one dataset; returns the CSV path."""
    DATA.mkdir(exist_ok=True)
    csv = DATA / f"{name}.csv"
    if csv.exists():
        print(f"already present: {csv}")
        return csv
    url = asset_url(name)
    archive = DATA / f"{name}.tar.xz"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, mode="r:xz") as tar:
        tar.extractall(DATA, filter="data")
    print(f"wrote {csv}")
    return csv


def main() -> int:
    for name in (sys.argv[1:] or DEFAULT):
        fetch(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
