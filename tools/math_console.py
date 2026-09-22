#!/usr/bin/env python3
"""Non-certifying OPEN/HOLD snapshot for JETMOD and RN-UNIF.

Prints the fail-closed status board and exits 0 when the controlling flags
are still false / zero. Writes nothing. Reads no receipt that could flip a
flag. A green exit is not obligation discharge.

  python3 tools/math_console.py
  python3 tools/math_console.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# Constants. There is no switch, environment variable, or receipt that
# changes them. lemma_closed stays false. prizes stay 0.
BASE_BRANCH = "chatgpt/drive-github-hardening-20260919"
BASE_COMMIT = "988b0db6fb8cc4debff5d1b5db9d90f72cfdceaf"

SNAPSHOT: dict[str, Any] = {
    "schema": "q0.math-console-open-hold/v1",
    "as_of": "2026-09-22",
    "base_branch": BASE_BRANCH,
    "base_commit": BASE_COMMIT,
    "disposition": "OPEN/HOLD",
    "authority": "NONE",
    "float_path": "NON-CERTIFYING",
    "certifying": False,
    "lemma_closed": False,
    "prizes_solved": 0,
    "original_prizes_solved": 0,
    "independence_credit": 0,
    "lemma_flips": 0,
    "green_is_discharge": False,
    "tip_bernstein_discharges": False,
    "OBL-H5-JETMOD": {
        "status": "OPEN",
        "hold": "HOLD",
        "grade": "display_only",
        "falsifier": "width_exceeds_claimed_modulus",
        "falsifier_text": (
            "A band enclosure whose width exceeds the claimed modulus."
        ),
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
    },
    "D3-LEMMA-RN-UNIF": {
        "status": "OPEN",
        "hold": "HOLD",
        "piece_1": "OPEN",
        "piece_2": "OPEN",
        "lemma_closed": False,
        "discharges_lemma": False,
        "freeze": False,
    },
    "does_not_establish": (
        "This console does not discharge OBL-H5-JETMOD, does not close "
        "Piece 1 or Piece 2 of D3-LEMMA-RN-UNIF, does not flip lemma_closed, "
        "does not solve a prize (prizes_solved=0, original_prizes_solved=0), "
        "and does not award independence_credit. Tip Bernstein is not "
        "discharge. A green exit is not discharge."
    ),
}


def _board_lines(snap: dict[str, Any]) -> list[str]:
    jet = snap["OBL-H5-JETMOD"]
    rn = snap["D3-LEMMA-RN-UNIF"]
    return [
        "disposition OPEN/HOLD",
        "lemma_closed false",
        "original_prizes_solved 0",
        "prizes_solved 0",
        "independence_credit 0",
        "lemma_flips 0",
        f"OBL-H5-JETMOD {jet['status']} {jet['hold']}",
        f"width-vs-modulus falsifier: {jet['falsifier_text']}",
        "width-vs-modulus falsifier does not discharge OBL-H5-JETMOD",
        f"D3-LEMMA-RN-UNIF Piece 1 {rn['piece_1']}",
        f"D3-LEMMA-RN-UNIF Piece 2 {rn['piece_2']}",
        "tip Bernstein != discharge",
        "green != discharge",
        "certifying false",
        "authority NONE",
        snap["does_not_establish"],
    ]


def render(snap: dict[str, Any]) -> str:
    return "\n".join(_board_lines(snap)) + "\n"


def guard(snap: dict[str, Any]) -> str | None:
    """Return a failure line when a controlling flag has left false/zero."""
    jet = snap.get("OBL-H5-JETMOD")
    rn = snap.get("D3-LEMMA-RN-UNIF")
    if not isinstance(jet, dict) or not isinstance(rn, dict):
        return "CK-FAIL: snapshot missing OPEN/HOLD objects"
    checks = (
        (snap.get("lemma_closed") is False, "lemma_closed"),
        (snap.get("prizes_solved") == 0, "prizes_solved"),
        (snap.get("original_prizes_solved") == 0, "original_prizes_solved"),
        (snap.get("independence_credit") == 0, "independence_credit"),
        (snap.get("lemma_flips") == 0, "lemma_flips"),
        (snap.get("certifying") is False, "certifying"),
        (snap.get("green_is_discharge") is False, "green_is_discharge"),
        (snap.get("tip_bernstein_discharges") is False, "tip_bernstein_discharges"),
        (snap.get("disposition") == "OPEN/HOLD", "disposition"),
        (jet.get("status") == "OPEN", "OBL-H5-JETMOD.status"),
        (jet.get("hold") == "HOLD", "OBL-H5-JETMOD.hold"),
        (jet.get("discharges_OBL_H5_JETMOD") is False, "discharges_OBL_H5_JETMOD"),
        (jet.get("lemma_closed") is False, "OBL-H5-JETMOD.lemma_closed"),
        (rn.get("piece_1") == "OPEN", "piece_1"),
        (rn.get("piece_2") == "OPEN", "piece_2"),
        (rn.get("lemma_closed") is False, "D3-LEMMA-RN-UNIF.lemma_closed"),
        (rn.get("discharges_lemma") is False, "discharges_lemma"),
        (rn.get("freeze") is False, "freeze"),
        (rn.get("status") == "OPEN", "D3-LEMMA-RN-UNIF.status"),
    )
    for ok, name in checks:
        if not ok:
            return f"CK-FAIL: {name} left its fail-closed value"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the OPEN/HOLD snapshot as JSON",
    )
    args = parser.parse_args(argv)
    snap = json.loads(json.dumps(SNAPSHOT))
    snap["board_lines"] = _board_lines(snap)
    failure = guard(snap)
    if failure:
        print(failure, file=sys.stderr)
        return 1
    if args.json:
        text = json.dumps(snap, indent=2) + "\n"
    else:
        text = render(snap)
    if "lemma_closed false" not in text or "original_prizes_solved 0" not in text:
        print("CK-FAIL: snapshot omitted the fail-closed lines", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
