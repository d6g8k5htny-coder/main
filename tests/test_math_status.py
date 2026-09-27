"""Negative controls for the OPEN/HOLD math status packet.

The real packet must pass. Each other test copies it, breaks one flag or one
byte, and drives ``tools/math_status_check.py`` through its ``--packet`` flag.
A checker that only hashes, or that treats ``0`` as false, fails these tests.
Nothing here discharges ``OBL-H5-JETMOD`` or ``D3-LEMMA-RN-UNIF``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "math_status_check.py")
PACKET = os.path.join(ROOT, "docs", "math_status")


def run(packet_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, CHECKER, "--packet", packet_dir],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def copy_packet(tmp_path) -> str:
    dest = os.path.join(tmp_path, "math_status")
    shutil.copytree(PACKET, dest)
    return dest


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")


def refresh_pin(packet_dir: str, name: str) -> None:
    path = os.path.join(packet_dir, name)
    with open(path, "rb") as handle:
        data = handle.read()
    packet = load(os.path.join(packet_dir, "PACKET.json"))
    packet["transcriptions"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    dump(os.path.join(packet_dir, "PACKET.json"), packet)


def test_real_packet_passes_and_writes_nothing():
    before = {}
    for dirpath, _dirs, files in os.walk(PACKET):
        for name in files:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                before[path] = handle.read()
    result = run(PACKET)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "problems=0" in result.stdout
    assert "does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF" in result.stdout
    after = {}
    for path in before:
        with open(path, "rb") as handle:
            after[path] = handle.read()
    assert after == before


def test_negative_lemma_closed_true_is_refused_even_with_a_refreshed_digest(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "math_console_snapshot.json")
    snapshot = load(path)
    snapshot["D3_LEMMA_RN_UNIF"]["lemma_closed"] = True
    dump(path, snapshot)
    refresh_pin(packet_dir, "math_console_snapshot.json")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "lemma_closed must be false" in result.stdout


def test_negative_discharges_lemma_true_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "math_console_snapshot.json")
    snapshot = load(path)
    snapshot["D3_LEMMA_RN_UNIF"]["discharges_lemma"] = True
    dump(path, snapshot)
    refresh_pin(packet_dir, "math_console_snapshot.json")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "discharges_lemma must be false" in result.stdout


def test_negative_integer_zero_is_not_false(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "PACKET.json")
    packet = load(path)
    packet["prizes_solved"] = 0
    packet["lemma_closed"] = 0
    dump(path, packet)
    result = run(packet_dir)
    assert result.returncode != 0
    assert "prizes_solved must be false" in result.stdout
    assert "lemma_closed must be false" in result.stdout


def test_negative_prizes_and_credit_and_prize_flag(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "PACKET.json")
    packet = load(path)
    packet["prizes_solved"] = True
    packet["original_prize_closed"] = True
    packet["independence_credit"] = 1
    dump(path, packet)
    result = run(packet_dir)
    assert result.returncode != 0
    assert "prizes_solved must be false" in result.stdout
    assert "original_prize_closed must be false" in result.stdout
    assert "independence_credit must be the integer 0" in result.stdout


def test_negative_obligation_status_discharged_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "math_console_snapshot.json")
    snapshot = load(path)
    snapshot["OBL_H5_JETMOD"]["status"] = "DISCHARGED"
    dump(path, snapshot)
    refresh_pin(packet_dir, "math_console_snapshot.json")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "status must be OPEN" in result.stdout


def test_negative_mesh_certified_true_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "math_console_snapshot.json")
    snapshot = load(path)
    snapshot["mesh_plan_toy"]["certified"] = True
    dump(path, snapshot)
    refresh_pin(packet_dir, "math_console_snapshot.json")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "certified must be false" in result.stdout


def test_negative_transcription_byte_drift_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "STATUS.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace("still OPEN", "still OPEN ")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    result = run(packet_dir)
    assert result.returncode != 0
    assert "STATUS.md: sha256/bytes drifted" in result.stdout


def test_negative_readme_without_the_certified_quarantine_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "README.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
        "STATUS.md uses the word CERTIFIED",
        "STATUS.md mentions a form-level envelope",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    result = run(packet_dir)
    assert result.returncode != 0
    assert "missing required phrase" in result.stdout


def test_negative_evening_wall_phrase_dropped_is_refused_even_with_a_refreshed_digest(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "STATUS_RN_UNIF.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
        "does not discharge D3-LEMMA-RN-UNIF",
        "records D3-LEMMA-RN-UNIF",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    refresh_pin(packet_dir, "STATUS_RN_UNIF.md")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "missing required phrase" in result.stdout
    assert "lemma_closed must be false" not in result.stdout


def test_negative_chart_factor_phrase_dropped_is_refused_even_with_a_refreshed_digest(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "STATUS_JETMOD.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
        "Inventing φ/r^α is refused.",
        "A bridge formula is unnamed.",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    refresh_pin(packet_dir, "STATUS_JETMOD.md")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "missing required phrase" in result.stdout
    assert "discharges_OBL_H5_JETMOD must be false" not in result.stdout


def test_negative_sibling_sweep_nondischarge_dropped_is_refused_even_with_a_refreshed_digest(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "STATUS_JETMOD.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
        "Sibling sweep CLOSED EMPTY does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
        "Sibling sweep CLOSED EMPTY is noted beside the walls.",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    refresh_pin(packet_dir, "STATUS_JETMOD.md")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "missing required phrase" in result.stdout
    assert "discharges_OBL_H5_JETMOD must be false" not in result.stdout


JETMOD_STATUS_VOCAB = (
    ("jetmod_first_band_proto", "PARTIAL_C2_ONLY / REFUSED_NOT_24JET"),
    ("jetmod_first_band_multi_gram_v1", "PARTIAL_GRAM_BLOCKS / REFUSED_NOT_24JET"),
    ("jetmod_first_band_interval_r", "PARTIAL_C2_SMOKE / REFUSED_NOT_24JET"),
    ("jetmod_multi_jet_band", "PARTIAL_6_MS_DIAG / REFUSED_NOT_24JET"),
    ("jetmod_g12_ext_named", "PARTIAL_8_NAMED / REFUSED_NOT_24JET"),
)


def test_jetmod_instrumentation_status_vocab_is_partial_or_refused():
    with open(os.path.join(PACKET, "STATUS_JETMOD.md"), encoding="utf-8") as handle:
        text = handle.read()
    for prototype, token in JETMOD_STATUS_VOCAB:
        assert prototype in text
        assert token in text
    assert "CERTIFIED_24JET" not in text
    assert "| READY |" not in text
    assert "| DISCHARGED |" not in text
    assert "`discharges_OBL_H5_JETMOD` stays **false**" in text
    assert "`lemma_closed` stays **false**" in text
    assert "inventable_attempt_accepted" in text
    assert "certified_C_H=false" in text
    packet = load(os.path.join(PACKET, "PACKET.json"))
    assert packet["lemma_closed"] is False
    assert packet["OBL-H5-JETMOD"]["discharges_OBL_H5_JETMOD"] is False
    assert packet["OBL-H5-JETMOD"]["status"] == "OPEN"


def test_negative_jetmod_status_vocab_token_dropped_is_refused_even_with_a_refreshed_digest(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "STATUS_JETMOD.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
        "PARTIAL_C2_ONLY / REFUSED_NOT_24JET",
        "CERTIFIED_24JET",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    refresh_pin(packet_dir, "STATUS_JETMOD.md")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "PARTIAL_C2_ONLY / REFUSED_NOT_24JET" in result.stdout
    assert "forbidden status token" in result.stdout
    assert "discharges_OBL_H5_JETMOD must be false" not in result.stdout


def test_negative_extra_file_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    with open(os.path.join(packet_dir, "CLOSED.md"), "w", encoding="utf-8") as handle:
        handle.write("CLOSED\n")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "unexpected files" in result.stdout


def test_negative_console_source_flag_flip_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "math_console.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read().replace(
        '"lemma_closed": False',
        '"lemma_closed": True',
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    refresh_pin(packet_dir, "math_console.py")
    result = run(packet_dir)
    assert result.returncode != 0
    assert "controlling flag assigned true" in result.stdout


README_UNMIX_PHRASES = (
    "Instrumentation STATUS labels are the 2026-09-23 STATUS_JETMOD vocab: `PARTIAL_*` paired with `REFUSED_NOT_24JET`.",
    "That second group is honesty receipts, not instrumentation STATUS.",
    "Both groups are not discharge. eng ≠ discharge.",
    "SoT ABSENT.",
    "The SIDE24 ABSENT triad (RN_SIDE24, DENSITY, CELL) is navigation only and not a source of truth.",
    "ABSENT means the Drive SoT carriers are absent. Nothing is invented to fill them.",
)


def test_readme_unmix_keeps_status_distinct_from_honesty_receipts():
    with open(os.path.join(PACKET, "README.md"), encoding="utf-8") as handle:
        text = " ".join(handle.read().split())
    for phrase in README_UNMIX_PHRASES:
        assert phrase in text


def test_negative_readme_unmix_phrase_dropped_is_refused(tmp_path):
    packet_dir = copy_packet(tmp_path)
    path = os.path.join(packet_dir, "README.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read().replace(
            "honesty receipts, not instrumentation STATUS",
            "honesty labels",
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    result = run(packet_dir)
    assert result.returncode != 0
    assert "missing required phrase" in result.stdout
    assert "lemma_closed must be false" not in result.stdout


def test_packet_path_is_resolved_per_call(tmp_path):
    good = run(PACKET)
    assert good.returncode == 0
    bad = copy_packet(tmp_path)
    packet = load(os.path.join(bad, "PACKET.json"))
    packet["bridge"] = "DEPLOYED"
    dump(os.path.join(bad, "PACKET.json"), packet)
    refused = run(bad)
    assert refused.returncode != 0
    assert "PROPOSED_NOT_DEPLOYED" in refused.stdout
    again = run(PACKET)
    assert again.returncode == 0
