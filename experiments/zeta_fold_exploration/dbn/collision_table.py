"""Collect the collision events of the two kept tracker runs into one table.

Usage: python3 dbn/collision_table.py   (after `track.py 270` and `track.py 103 -40 0.02 low10`)
Reads results/dbn/track_log_z270.txt and track_log_low10.txt; writes
results/dbn/collision_events.csv (one row per EVENT line, in log order). Standard library only.
"""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "dbn"
RUNS = (("track_log_z270.txt", "zeros 1-46 (z<=270, gamma<135)"),
        ("track_log_low10.txt", "zeros 1-10 (z<=103)"))

with open(OUT / "collision_events.csv", "w", newline="") as fh:
    wr = csv.writer(fh, lineterminator="\n")
    wr.writerow(["window", "k", "k+1", "gamma_k(t=0)", "gamma_k+1(t=0)", "t_star", "x0",
                 "A=H_zz/2", "B=H_zzz/6"])
    for name, label in RUNS:
        for line in (OUT / name).read_text().splitlines():
            e = json.loads(line)
            wr.writerow([label, e["pair"][0], e["pair"][1], e["gamma_pair_t0"][0],
                         e["gamma_pair_t0"][1], e["t_star"], e["x0"], e["A"], e["B"]])
