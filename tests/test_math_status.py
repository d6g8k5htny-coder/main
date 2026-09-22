"""Negative controls for the OPEN/HOLD math-status packet.

The checker is invoked through its CLI against a copy of the packet, so a
mutation cannot be hidden by a path bound at import time.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _checker() -> str:
    return os.path.join(_root(), "tools", "math_status_check.py")


def _packet() -> str:
    return os.path.join(_root(), "docs", "math_status")


def _run(root: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, _checker(), "--root", root],
        capture_output=True, text=True, cwd=_root(),
    )


def _stage(tmp_path) -> str:
    root = tmp_path / "repo"
    shutil.copytree(_packet(), root / "docs" / "math_status")
    return str(root)


def _write_packet(root: str, mutate) -> None:
    path = os.path.join(root, "docs", "math_status", "PACKET.json")
    packet = json.loads(open(path, encoding="utf-8").read())
    mutate(packet)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(packet, handle)


def test_packet_passes():
    out = _run(_root())
    assert out.returncode == 0, out.stdout + out.stderr
    assert "problems=0" in out.stdout


def test_lemma_closed_true_is_refused(tmp_path):
    root = _stage(tmp_path)
    _write_packet(root, lambda packet: packet.__setitem__("lemma_closed", True))
    out = _run(root)
    assert out.returncode == 1
    assert "lemma_closed" in out.stdout


def test_prizes_solved_true_is_refused(tmp_path):
    root = _stage(tmp_path)
    _write_packet(root, lambda packet: packet.__setitem__("prizes_solved", True))
    out = _run(root)
    assert out.returncode == 1
    assert "prizes_solved" in out.stdout


def test_independence_credit_nonzero_is_refused(tmp_path):
    root = _stage(tmp_path)
    _write_packet(root, lambda packet: packet.__setitem__("independence_credit", 1))
    out = _run(root)
    assert out.returncode == 1
    assert "independence_credit" in out.stdout


def test_jetmod_discharged_true_is_refused(tmp_path):
    root = _stage(tmp_path)
    _write_packet(root, lambda packet: packet.__setitem__("jetmod_discharged", True))
    out = _run(root)
    assert out.returncode == 1
    assert "jetmod_discharged" in out.stdout


def test_snapshot_certified_true_is_refused(tmp_path):
    root = _stage(tmp_path)
    path = os.path.join(root, "docs", "math_status", "math_console_snapshot.json")
    snap = json.loads(open(path, encoding="utf-8").read())
    snap["D3_LEMMA_RN_UNIF"]["U_certified"] = True
    snap["mesh_plan_toy"]["certified"] = True
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snap, handle)
    out = _run(root)
    assert out.returncode == 1
    assert "U_certified" in out.stdout
    assert "certified" in out.stdout


def test_banner_removal_is_refused(tmp_path):
    root = _stage(tmp_path)
    path = os.path.join(root, "docs", "math_status", "STATUS.md")
    text = open(path, encoding="utf-8").read().replace(
        "This repository does not adopt any CERTIFIED label in a transcribed memo "
        "as a certified enclosure.",
        "This repository adopts the memo's CERTIFIED label.",
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    out = _run(root)
    assert out.returncode == 1
    assert "certified-enclosure refusal" in out.stdout


def test_console_seal_flip_is_refused(tmp_path):
    """A console that prints lemma_closed true and exits 0 is still refused."""
    root = _stage(tmp_path)
    path = os.path.join(root, "docs", "math_status", "math_console.py")
    text = open(path, encoding="utf-8").read()
    text = text.replace('board["lemma_closed"] = False', 'board["lemma_closed"] = True', 1)
    text = text.replace(
        "def fail_closed_ok(board: dict) -> bool:\n",
        "def fail_closed_ok(board: dict) -> bool:\n    return True\n",
        1,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    out = _run(root)
    assert out.returncode == 1
    assert "lemma_closed" in out.stdout
