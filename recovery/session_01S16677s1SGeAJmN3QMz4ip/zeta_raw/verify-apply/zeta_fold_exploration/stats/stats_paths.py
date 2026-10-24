"""Shared locations for the stats scripts (all relative to this file).

OUT  results/stats/  small committed summaries (JSON, CSV, SVG, .out logs)
RAW  results/raw/    regenerable large samples (.npz); not committed (see results/raw/.gitignore).
     Set the environment variable ZETA_FOLD_RAW to read or write raw samples elsewhere.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "stats"
RAW = Path(os.environ.get("ZETA_FOLD_RAW", str(ROOT / "results" / "raw")))


def out_path(name):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def raw_path(name):
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / name
