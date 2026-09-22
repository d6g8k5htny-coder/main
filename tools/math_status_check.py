#!/usr/bin/env python3
"""Refuse a math-status packet that leaves OPEN/HOLD.

The packet under ``docs/math_status/`` is an execution-workspace mirror of
2026-09-21 memos and a 2026-09-22 display snapshot. This tool checks the
recorded flags. It does not grade a proof, and it never changes a status.

It fails when ``lemma_closed``, ``prizes_solved`` or ``original_prize_closed``
is true, when ``independence_credit`` is nonzero, when JETMOD or RN-UNIF is
recorded discharged, frozen, or certified, or when the live console emits
any of those. Display figures stay NON-CERTIFYING.

What this does not establish: closure of any obligation, a certified
enclosure, a novelty claim, a prize solution, independence credit, or
deployment of the execution bridge.

Exit status is non-zero when any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKET_REL = os.path.join("docs", "math_status")

REQUIRED_FILES = (
    "README.md",
    "PACKET.json",
    "STATUS.md",
    "STATUS_JETMOD.md",
    "STATUS_MATH_PUSH_2026-09-21.md",
    "math_console.py",
    "math_console_snapshot.json",
)

BANNER = (
    "This repository does not adopt any CERTIFIED label in a transcribed memo "
    "as a certified enclosure."
)

README_PHRASES = (
    "OPEN/HOLD",
    "PROPOSED EXECUTION CONTRACT / NOT DEPLOYED",
    "source of truth",
    "execution/workspace mirror",
    "no novelty claim",
    "No original prize problem is solved",
    "does not establish",
    "RUNG2",
    "RUNG3",
    "green CI",
    "lemma_closed",
    "prizes_solved",
    "independence_credit",
)

# Assignments that would record a discharge. Refusal branches in the console
# ("is True") are a different shape and are not matches.
DENY = (
    (re.compile(r"""lemma_closed["']?\s*[:=]\s*true\b""", re.I), "lemma_closed set true"),
    (re.compile(r"""prizes_solved["']?\s*[:=]\s*true\b""", re.I), "prizes_solved set true"),
    (re.compile(r"""original_prize_closed["']?\s*[:=]\s*true\b""", re.I),
     "original_prize_closed set true"),
    (re.compile(r"""independence_credit["']?\s*[:=]\s*(?!0\b)\d+"""),
     "independence_credit set nonzero"),
    (re.compile(r"""discharges_OBL["']?\s*[:=]\s*true\b"""), "discharges_OBL set true"),
    (re.compile(r"""discharges_lemma["']?\s*[:=]\s*true\b"""), "discharges_lemma set true"),
    (re.compile(r"""["']certified_enclosure["']\s*:\s*true\b"""), "certified_enclosure set true"),
    (re.compile(r"""["']jetmod_discharged["']\s*:\s*true\b"""), "jetmod_discharged set true"),
    (re.compile(r"""["']rn_unif_discharged["']\s*:\s*true\b"""), "rn_unif_discharged set true"),
    (re.compile(r"""["']freeze["']\s*:\s*true\b"""), "freeze set true"),
    (re.compile(r"""["']U_certified["']\s*:\s*true\b"""), "U_certified set true"),
    (re.compile(r"""["']certified["']\s*:\s*true\b"""), "certified set true"),
    (re.compile(r"""["']novelty_claim["']\s*:\s*true\b"""), "novelty_claim set true"),
    (re.compile(r"""["']scientific_status_changed["']\s*:\s*true\b"""),
     "scientific_status_changed set true"),
    (re.compile(r"""OBL-H5-JETMOD (?:is|remains|stays) (?:CLOSED|DISCHARGED|FREEZE)\b"""),
     "JETMOD recorded closed"),
    (re.compile(r"""D3-LEMMA-RN-UNIF (?:is|remains|stays) (?:CLOSED|DISCHARGED|FREEZE)\b"""),
     "RN-UNIF recorded closed"),
)

SEAL_LINES = (
    'board["lemma_closed"] = False',
    'board["prizes_solved"] = False',
    'board["original_prize_closed"] = False',
    'board["independence_credit"] = 0',
    'j["status"] = "OPEN"',
    'j["lemma_closed"] = False',
    'j["discharges_OBL"] = False',
    'r["status"] = "OPEN"',
    'r["lemma_closed"] = False',
    'r["U_certified"] = False',
    'r["discharges_lemma"] = False',
    'm["certified"] = False',
)

PACKET_EQUALS = {
    "schema": "q0.math-status-packet/v1",
    "disposition": "OPEN/HOLD",
    "authority": "NONE",
    "drive_is_source_of_truth": True,
    "git_role": "execution/workspace mirror only",
    "bridge_disposition": "PROPOSED EXECUTION CONTRACT / NOT DEPLOYED",
    "lemma_closed": False,
    "prizes_solved": False,
    "original_prize_closed": False,
    "scientific_status_changed": False,
    "obl_h5_jetmod": "OPEN",
    "d3_lemma_rn_unif": "OPEN",
    "piece_1": "OPEN",
    "piece_2": "OPEN",
    "jetmod_discharged": False,
    "rn_unif_discharged": False,
    "freeze": False,
    "certified_enclosure": False,
    "display_is_certified": False,
    "rung2_discharges_jetmod": False,
    "rung3_discharges_jetmod": False,
    "green_ci_discharges_obligation": False,
    "novelty_claim": False,
}


def _load_json(path: str, problems: list[str]):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{path}: {exc}")
        return None


def _require_false(obj: dict, key: str, where: str, problems: list[str]) -> None:
    if key not in obj:
        problems.append(f"{where}: missing {key}")
        return
    if obj[key] is not False:
        problems.append(f"{where}: {key} must be false, found {obj[key]!r}")


def _require_open(obj: dict, key: str, where: str, problems: list[str]) -> None:
    if obj.get(key) != "OPEN":
        problems.append(f"{where}: {key} must be OPEN, found {obj.get(key)!r}")


def _check_packet(packet: dict, problems: list[str]) -> None:
    for key, expected in PACKET_EQUALS.items():
        found = packet.get(key)
        if found != expected or type(found) is not type(expected):
            problems.append(f"PACKET.json: {key} must be {expected!r}, found {found!r}")
    credit = packet.get("independence_credit")
    if type(credit) is not int or credit != 0:
        problems.append(f"PACKET.json: independence_credit must be 0, found {credit!r}")
    sentences = packet.get("does_not_establish")
    if not isinstance(sentences, list) or not sentences or not all(
            isinstance(s, str) and s.strip() for s in sentences):
        problems.append("PACKET.json: does_not_establish must be a non-empty list of sentences")
    else:
        blob = " ".join(sentences)
        for needle in ("OBL-H5-JETMOD", "D3-LEMMA-RN-UNIF", "prize", "independence",
                       "certified enclosure", "bridge"):
            if needle not in blob:
                problems.append(f"PACKET.json: does_not_establish is missing {needle!r}")
    files = packet.get("files")
    if not isinstance(files, list) or files != list(REQUIRED_FILES):
        problems.append(f"PACKET.json: files must be {list(REQUIRED_FILES)!r}")


def _check_snapshot(snap: dict, problems: list[str]) -> None:
    jet = snap.get("OBL_H5_JETMOD")
    rn = snap.get("D3_LEMMA_RN_UNIF")
    mesh = snap.get("mesh_plan_toy")
    if not isinstance(jet, dict) or not isinstance(rn, dict) or not isinstance(mesh, dict):
        problems.append("math_console_snapshot.json: missing JETMOD, RN-UNIF or mesh object")
        return
    _require_open(jet, "status", "snapshot OBL_H5_JETMOD", problems)
    _require_false(jet, "lemma_closed", "snapshot OBL_H5_JETMOD", problems)
    _require_false(jet, "discharges_OBL", "snapshot OBL_H5_JETMOD", problems)
    _require_open(rn, "status", "snapshot D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "lemma_closed", "snapshot D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "discharges_lemma", "snapshot D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "U_certified", "snapshot D3_LEMMA_RN_UNIF", problems)
    _require_false(mesh, "certified", "snapshot mesh_plan_toy", problems)
    for key in ("lemma_closed", "prizes_solved", "original_prize_closed"):
        if key in snap:
            _require_false(snap, key, "snapshot", problems)
    if "independence_credit" in snap:
        credit = snap["independence_credit"]
        if type(credit) is not int or credit != 0:
            problems.append(f"snapshot: independence_credit must be 0, found {credit!r}")


def _check_live(board: dict, problems: list[str]) -> None:
    _require_false(board, "lemma_closed", "live console", problems)
    _require_false(board, "prizes_solved", "live console", problems)
    _require_false(board, "original_prize_closed", "live console", problems)
    _require_false(board, "historical_snapshot_rewritten", "live console", problems)
    credit = board.get("independence_credit")
    if type(credit) is not int or credit != 0:
        problems.append(f"live console: independence_credit must be 0, found {credit!r}")
    if board.get("arithmetic") != "NON-CERTIFYING":
        problems.append(f"live console: arithmetic must be NON-CERTIFYING, found {board.get('arithmetic')!r}")
    if board.get("bridge_disposition") != "PROPOSED EXECUTION CONTRACT / NOT DEPLOYED":
        problems.append("live console: bridge_disposition moved")
    if board.get("authority") != "NONE":
        problems.append(f"live console: authority must be NONE, found {board.get('authority')!r}")
    jet = board.get("OBL_H5_JETMOD") if isinstance(board.get("OBL_H5_JETMOD"), dict) else {}
    rn = board.get("D3_LEMMA_RN_UNIF") if isinstance(board.get("D3_LEMMA_RN_UNIF"), dict) else {}
    mesh = board.get("mesh_plan_toy") if isinstance(board.get("mesh_plan_toy"), dict) else {}
    _require_open(jet, "status", "live OBL_H5_JETMOD", problems)
    _require_false(jet, "lemma_closed", "live OBL_H5_JETMOD", problems)
    _require_false(jet, "discharges_OBL", "live OBL_H5_JETMOD", problems)
    _require_open(rn, "status", "live D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "lemma_closed", "live D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "discharges_lemma", "live D3_LEMMA_RN_UNIF", problems)
    _require_false(rn, "U_certified", "live D3_LEMMA_RN_UNIF", problems)
    _require_false(mesh, "certified", "live mesh_plan_toy", problems)


def _scan_text(rel: str, text: str, problems: list[str]) -> None:
    for line_no, line in enumerate(text.splitlines(), 1):
        for pattern, label in DENY:
            if pattern.search(line):
                problems.append(f"{rel}:{line_no}: {label}")


def check(root: str) -> list[str]:
    problems: list[str] = []
    packet_dir = os.path.join(root, PACKET_REL)
    if not os.path.isdir(packet_dir):
        return [f"missing packet directory {PACKET_REL}"]
    present = sorted(
        name for name in os.listdir(packet_dir)
        if name != "__pycache__" and not name.endswith(".pyc")
    )
    for name in REQUIRED_FILES:
        if name not in present:
            problems.append(f"missing {PACKET_REL}/{name}")
    for name in present:
        if name not in REQUIRED_FILES:
            problems.append(f"unexpected file in packet: {name}")

    for name in REQUIRED_FILES:
        path = os.path.join(packet_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".md") or name.endswith(".py") or name.endswith(".json"):
            text = open(path, encoding="utf-8").read()
            _scan_text(f"{PACKET_REL}/{name}", text, problems)
            if name.endswith(".md") and BANNER not in text:
                problems.append(f"{PACKET_REL}/{name}: missing the certified-enclosure refusal")
            if name == "README.md":
                for phrase in README_PHRASES:
                    if phrase not in text:
                        problems.append(f"README.md is missing {phrase!r}")
            if name == "math_console.py":
                for line in SEAL_LINES:
                    if line not in text:
                        problems.append(f"math_console.py is missing the fail-closed assignment {line}")

    packet_path = os.path.join(packet_dir, "PACKET.json")
    if os.path.isfile(packet_path):
        packet = _load_json(packet_path, problems)
        if isinstance(packet, dict):
            _check_packet(packet, problems)
        elif packet is not None:
            problems.append("PACKET.json: not an object")

    snap_path = os.path.join(packet_dir, "math_console_snapshot.json")
    if os.path.isfile(snap_path):
        before = open(snap_path, "rb").read()
        snap = _load_json(snap_path, problems)
        if isinstance(snap, dict):
            _check_snapshot(snap, problems)
        elif snap is not None:
            problems.append("math_console_snapshot.json: not an object")
    else:
        before = None

    script = os.path.join(packet_dir, "math_console.py")
    if os.path.isfile(script):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        done = subprocess.run(
            [sys.executable, script, "--json"],
            capture_output=True, text=True, cwd=packet_dir, env=env,
        )
        if before is not None and open(snap_path, "rb").read() != before:
            problems.append("math_console.py rewrote math_console_snapshot.json")
        if done.returncode != 0:
            tail = (done.stdout + done.stderr).strip().splitlines()
            problems.append("math_console.py exited "
                            f"{done.returncode}: {tail[-1] if tail else ''}")
        else:
            try:
                board = json.loads(done.stdout)
            except json.JSONDecodeError as exc:
                problems.append(f"math_console.py --json did not print an object: {exc}")
            else:
                if isinstance(board, dict):
                    _check_live(board, problems)
                else:
                    problems.append("math_console.py --json printed a non-object")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="repository root (default: this checkout)")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root or REPO)
    problems = check(root)
    for problem in problems:
        print(problem)
    print(f"math_status_check: files={len(REQUIRED_FILES)} problems={len(problems)}")
    print("A pass here is a flag check on the math-status packet. It establishes "
          "no closure, no certificate, no prize, and no independence credit.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
