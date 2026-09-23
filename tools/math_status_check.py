#!/usr/bin/env python3
"""Fail-closed check for the OPEN/HOLD math status packet.

``docs/math_status/`` is an execution/workspace mirror of a 2026-09-21/22
display. Drive remains the source of truth. This checker keeps the packet
from quietly saying otherwise. It asserts:

  1. **Flags.** ``PACKET.json`` matches the closed schema below. ``lemma_closed``,
     ``prizes_solved``, ``original_prize_closed``, every discharge flag,
     ``U_certified``, ``certified``, ``freeze``, and the three "does this
     display discharge anything" booleans are ``false`` the boolean, not
     ``0``. ``independence_credit`` is the integer ``0``. Obligation ``status``
     values are ``OPEN``. The packet disposition is ``OPEN_HOLD``. The bridge
     string is ``PROPOSED_NOT_DEPLOYED``.

  2. **Snapshot.** ``math_console_snapshot.json`` keeps the same false flags,
     ``piece2_annulus_driver`` ``UNWRITTEN``, and the console's own non-claims.
     Floats in that file are a display. This checker does not certify them.

  3. **Transcriptions.** The six uploaded bodies match the digests pinned in
     ``PACKET.json``. Refreshing a digest to match an edited body does not
     excuse a flag that left false: both checks run. The 2026-09-22 evening
     CT wall notes are part of that pin. They do not discharge either
     obligation. The 2026-09-22 evening CT sibling sweep CLOSED EMPTY is
     part of that pin. It does not discharge either obligation.

  4. **Prose.** The packet README carries the OPEN/HOLD, Drive-source-of-truth,
     bridge, prize, independence, RUNG2/3, and certified-enclosure sentences.
     No file in the packet assigns a controlling flag to true or a nonzero
     independence credit.

  5. **Closure of the directory.** The packet holds exactly the expected
     names. An extra file is a problem.

The packet directory is taken from ``--packet`` at call time.

WHAT A PASS DOES NOT ESTABLISH. It does not discharge ``OBL-H5-JETMOD`` or
``D3-LEMMA-RN-UNIF``, FREEZE either, certify an enclosure, close a prize,
award independence credit, or deploy the bridge. The five validity premises
of Theorem D1 v2.2(2) stay OPEN. A green checker is not obligation discharge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any


FALSE_KEYS = frozenset({
    "lemma_closed",
    "prizes_solved",
    "original_prize_closed",
    "discharges_OBL",
    "discharges_lemma",
    "discharges_OBL_H5_JETMOD",
    "U_certified",
    "certified",
    "freeze",
    "display_is_certified_enclosure",
    "rung2_or_rung3_discharges_jetmod",
    "green_ci_discharges_obligations",
})

PACKET_KEYS = frozenset({
    "schema",
    "as_of",
    "base_branch",
    "base_commit",
    "disposition",
    "authority",
    "drive_is_source_of_truth",
    "git_role",
    "bridge",
    "lemma_closed",
    "prizes_solved",
    "original_prize_closed",
    "independence_credit",
    "display_is_certified_enclosure",
    "rung2_or_rung3_discharges_jetmod",
    "green_ci_discharges_obligations",
    "freeze",
    "float_path",
    "OBL-H5-JETMOD",
    "D3-LEMMA-RN-UNIF",
    "does_not_establish",
    "transcriptions",
})

TRANSCRIPTION_NAMES = (
    "STATUS.md",
    "STATUS_JETMOD.md",
    "STATUS_RN_UNIF.md",
    "STATUS_MATH_PUSH_2026-09-21.md",
    "math_console.py",
    "math_console_snapshot.json",
)

EXPECTED_NAMES = frozenset(TRANSCRIPTION_NAMES + ("README.md", "PACKET.json"))

README_PHRASES = (
    "OPEN / HOLD",
    "Drive is the source of truth",
    "execution/workspace mirror only",
    "PROPOSED / NOT DEPLOYED",
    "`lemma_closed` is false",
    "`prizes_solved` is false",
    "`original_prize_closed` is false",
    "`independence_credit` is 0",
    "`OBL-H5-JETMOD` stays OPEN",
    "`D3-LEMMA-RN-UNIF` stays OPEN",
    "RUNG2 and RUNG3 do not discharge OBL-H5-JETMOD",
    "A display is not a certified enclosure",
    "NON-CERTIFYING",
    "A green run of this checker is not obligation discharge.",
    "does not establish",
    "chatgpt/drive-github-hardening-20260919",
    "d107ab121d230de33c09e727c7804098ec4e8249",
    "Default branch `main` is untouched.",
    "no novelty claim",
    "five validity premises of Theorem D1 v2.2(2) stay OPEN",
    "STATUS.md uses the word CERTIFIED for a transcribed form-level whitened "
    "residual-form q=2 envelope. That sentence does not establish a certified "
    "enclosure of D3-LEMMA-RN-UNIF, does not FREEZE the lemma, and does not "
    "discharge it.",
    "STATUS_JETMOD.md records the 2026-09-22 evening CT JETMOD walls and does not discharge OBL-H5-JETMOD.",
    "STATUS_RN_UNIF.md records the 2026-09-22 evening CT RN-UNIF walls and does not discharge D3-LEMMA-RN-UNIF.",
)

NOTE_PHRASES = {
    "STATUS.md": (
        "lemma_closed: false",
        "still OPEN",
        "NOT FREEZE-grade",
        "Lemma remains OPEN.",
        "Walls recorded 2026-09-22 evening CT",
        "certified_C_H=false",
        "FORM/√λ proxy not promoted",
        "ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT",
        "does not discharge D3-LEMMA-RN-UNIF",
        "Sibling sweep CLOSED EMPTY does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
        "REFUSED_IA_STRADDLES",
        "jetmod_interval_schur_detgg_via_ainv_receipt.json",
        "jetmod_eval_F_G12box_sibling_probe_receipt.json",
        "jetmod_joint_ry_cancel_rewrite_hunt_receipt.json",
        "correlated_C_Ainv_Ct_cancellation_under_joint_r_y",
        "cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM",
        "still **ABSENT** (prior)",
    ),
    "STATUS_JETMOD.md": (
        "**OPEN (display only)**",
        "`discharges_OBL_H5_JETMOD` | **false**",
        "`lemma_closed` | **false**",
        "RUNG2/3 do NOT discharge JETMOD.",
        "**OBL-H5-JETMOD remains OPEN.**",
        "Walls recorded 2026-09-22 evening CT",
        "These lines do not discharge OBL-H5-JETMOD.",
        "F(G12)/Lip **REFUSED**",
        "cover pipeline F **REFUSED**",
        "explicit_interval_map_F_G12box_to_Rplus",
        "joint-(r,y) `cancelled_detgg_s_t2` identity **ABSENT**",
        "documented detgg=ad-c^2 still straddles",
        "det(A)=det(G6)det(T)^2",
        "is not a StationBox detgg enclosure",
        "Inventing φ/r^α is refused.",
        "jetmod_lat_k1_detgg_factor_probe_receipt.json",
        "That identity does not discharge OBL-H5-JETMOD.",
        "Sibling sweep CLOSED EMPTY does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
        "REFUSED_IA_STRADDLES",
        "jetmod_interval_schur_detgg_via_ainv_receipt.json",
        "jetmod_eval_F_G12box_sibling_probe_receipt.json",
        "jetmod_joint_ry_cancel_rewrite_hunt_receipt.json",
        "correlated_C_Ainv_Ct_cancellation_under_joint_r_y",
        "cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM",
        "still **ABSENT** (prior)",
        "Inventable probes (REFUSED receipts only)",
        "inventable_jetmod_probes.py",
        "REFUSED_IA_STRADDLES",
        "inventable_attempt_accepted: false",
        "These receipts do not discharge OBL-H5-JETMOD",
    ),
    "STATUS_RN_UNIF.md": (
        "D3-LEMMA-RN-UNIF remains OPEN",
        "lemma_closed: false",
        "does not discharge D3-LEMMA-RN-UNIF",
        "certified_C_H=false",
        "FORM/√λ proxy not promoted",
        "ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT",
        "rnu_env.py",
        "CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip",
        "allcell_fdz_enclosures.json",
        "**ABSENT**",
        "Piece-2 annulus driver stays **UNWRITTEN**",
        "Lemma remains OPEN.",
    ),
    "STATUS_MATH_PUSH_2026-09-21.md": (
        "## OBL-H5-JETMOD — OPEN",
        "RUNG2/3 do **not** discharge JETMOD",
        "## D3-LEMMA-RN-UNIF — OPEN (`lemma_closed: false`)",
        "(NOT certified)",
    ),
}

CONSOLE_PHRASES = (
    '"lemma_closed": False',
    '"discharges_OBL": False',
    '"discharges_lemma": False',
    "CK-FAIL: console tried to claim discharge",
    "This console never discharges OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
)

NON_CLAIMS = (
    "This console never discharges OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
    "Toy mesh plan is not a certificate.",
    "Provisional U / empirical envelope ≠ FREEZE.",
)

ASSIGN_TRUE = re.compile(
    r"""(?ix)
    ["']?(lemma_closed|prizes_solved|original_prize_closed|discharges_obl
        |discharges_lemma|discharges_obl_h5_jetmod|u_certified)["']?
        \s*[:=]\s*(?:true)\b
    |["']?certified["']?\s*[:=]\s*(?:true)\b
    |["']?independence_credit["']?\s*[:=]\s*(?!0(?:\b|[^0-9]))[0-9]+
    """
)


def _reject_duplicates(pairs: list) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.loads(
            handle.read(),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )


def walk_flags(obj: Any, prefix: str, problems: list, rel: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in FALSE_KEYS:
                if value is not False:
                    problems.append(f"{rel}: {path} must be false")
            elif key == "independence_credit":
                if type(value) is not int or value != 0:
                    problems.append(f"{rel}: {path} must be the integer 0")
            elif key == "status":
                if value != "OPEN":
                    problems.append(f"{rel}: {path} must be OPEN")
            elif key == "piece2_annulus_driver":
                if value != "UNWRITTEN":
                    problems.append(f"{rel}: {path} must be UNWRITTEN")
            elif key == "disposition":
                if value != "OPEN_HOLD":
                    problems.append(f"{rel}: {path} must be OPEN_HOLD")
            elif key == "bridge":
                if value != "PROPOSED_NOT_DEPLOYED":
                    problems.append(f"{rel}: {path} must be PROPOSED_NOT_DEPLOYED")
            elif key == "authority":
                if value != "NONE":
                    problems.append(f"{rel}: {path} must be NONE")
            elif key == "float_path":
                if value != "NON-CERTIFYING":
                    problems.append(f"{rel}: {path} must be NON-CERTIFYING")
            elif key == "drive_is_source_of_truth":
                if value is not True:
                    problems.append(f"{rel}: {path} must be true")
            walk_flags(value, path, problems, rel)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            walk_flags(value, f"{prefix}[{index}]", problems, rel)


def check_packet_schema(packet: Any, problems: list) -> None:
    if not isinstance(packet, dict):
        problems.append("PACKET.json: must be an object")
        return
    missing = sorted(PACKET_KEYS - set(packet))
    extra = sorted(set(packet) - PACKET_KEYS)
    if missing:
        problems.append(f"PACKET.json: missing keys {missing}")
    if extra:
        problems.append(f"PACKET.json: unexpected keys {extra}")
    if packet.get("schema") != "q0.math-status-packet/v1":
        problems.append("PACKET.json: schema must be q0.math-status-packet/v1")
    if packet.get("base_branch") != "chatgpt/drive-github-hardening-20260919":
        problems.append("PACKET.json: base_branch must be the hardening tip branch")
    if packet.get("base_commit") != "d107ab121d230de33c09e727c7804098ec4e8249":
        problems.append("PACKET.json: base_commit must be the hardening tip")
    if packet.get("git_role") != "execution_workspace_mirror_only":
        problems.append("PACKET.json: git_role must be execution_workspace_mirror_only")
    jet = packet.get("OBL-H5-JETMOD")
    if not isinstance(jet, dict) or set(jet) != {
        "status", "grade", "discharges_OBL_H5_JETMOD", "lemma_closed", "freeze",
    }:
        problems.append("PACKET.json: OBL-H5-JETMOD keys drifted")
    elif jet.get("grade") != "display_only":
        problems.append("PACKET.json: OBL-H5-JETMOD.grade must be display_only")
    rn = packet.get("D3-LEMMA-RN-UNIF")
    if not isinstance(rn, dict) or set(rn) != {
        "status", "lemma_closed", "discharges_lemma", "freeze", "piece2_annulus_driver",
    }:
        problems.append("PACKET.json: D3-LEMMA-RN-UNIF keys drifted")
    text = packet.get("does_not_establish")
    if not isinstance(text, str) or "does not discharge" not in text:
        problems.append("PACKET.json: does_not_establish must say it does not discharge")
    transcriptions = packet.get("transcriptions")
    if not isinstance(transcriptions, dict) or set(transcriptions) != set(TRANSCRIPTION_NAMES):
        problems.append("PACKET.json: transcriptions must name exactly the transcribed files")


def check_snapshot(snapshot: Any, problems: list) -> None:
    if not isinstance(snapshot, dict):
        problems.append("math_console_snapshot.json: must be an object")
        return
    discipline = snapshot.get("discipline")
    if not isinstance(discipline, str) or "fail-closed" not in discipline or "Drive SoT" not in discipline:
        problems.append("math_console_snapshot.json: discipline must stay fail-closed and name Drive SoT")
    jet = snapshot.get("OBL_H5_JETMOD")
    rn = snapshot.get("D3_LEMMA_RN_UNIF")
    mesh = snapshot.get("mesh_plan_toy")
    action = snapshot.get("next_action")
    if not isinstance(jet, dict) or not isinstance(rn, dict):
        problems.append("math_console_snapshot.json: both obligation objects are required")
        return
    if jet.get("jets_claimed_by_OBL") != 24:
        problems.append("math_console_snapshot.json: jets_claimed_by_OBL must stay 24")
    if not isinstance(mesh, dict):
        problems.append("math_console_snapshot.json: mesh_plan_toy is required")
    elif mesh.get("certified") is not False:
        problems.append("math_console_snapshot.json: mesh_plan_toy.certified must be false")
    if not isinstance(action, dict) or "Do not flip lemma_closed" not in str(action.get("do_not")):
        problems.append("math_console_snapshot.json: next_action must refuse flipping lemma_closed")
    claims = snapshot.get("non_claims")
    if claims != list(NON_CLAIMS):
        problems.append("math_console_snapshot.json: non_claims drifted")


def check_assignment_text(rel: str, text: str, problems: list) -> None:
    for lineno, line in enumerate(text.splitlines(), 1):
        if ASSIGN_TRUE.search(line):
            problems.append(f"{rel}:{lineno}: controlling flag assigned true or nonzero")


# Tip-aligned shortcut receipts. Same refusal bar as the sibling-sweep four,
# plus explicit discharges_lemma / certified_C_H false. Green ≠ discharge.
SHORTCUT_RECEIPTS = frozenset({
    "inventable_24jet_roster_without_Drive_list_REFUSED_receipt.json",
    "inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json",
    "inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json",
})


def check_inventable_probes(root: str, problems: list) -> None:
    """Named-wall inventable probes must exist and stay REFUSED/EMPTY/ABSENT."""
    probes_dir = os.path.join(root, "docs", "math_status_probes")
    expected = {
        "inventable_interval_schur_ainv_REFUSED_receipt.json": "REFUSED_IA_STRADDLES",
        "inventable_eval_F_G12box_REFUSED_receipt.json": "REFUSED",
        "inventable_joint_ry_cancel_EMPTY_receipt.json": "EMPTY",
        "inventable_phi_bridge_ABSENT_receipt.json": "ABSENT",
        "inventable_24jet_roster_without_Drive_list_REFUSED_receipt.json": "REFUSED",
        "inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json": "REFUSED",
        "inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json": "REFUSED",
    }
    if not os.path.isdir(probes_dir):
        problems.append("math_status_probes: directory missing")
        return
    for name, status in expected.items():
        path = os.path.join(probes_dir, name)
        if not os.path.isfile(path):
            problems.append(f"math_status_probes: missing {name}")
            continue
        try:
            obj = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"math_status_probes/{name}: {exc}")
            continue
        if obj.get("status") != status:
            problems.append(f"math_status_probes/{name}: status must be {status}")
        if obj.get("inventable_attempt_accepted") is not False:
            problems.append(f"math_status_probes/{name}: inventable_attempt_accepted must be false")
        if obj.get("discharges_OBL_H5_JETMOD") is not False:
            problems.append(f"math_status_probes/{name}: discharges_OBL_H5_JETMOD must be false")
        if obj.get("lemma_closed") is not False:
            problems.append(f"math_status_probes/{name}: lemma_closed must be false")
        if obj.get("freeze") is not False:
            problems.append(f"math_status_probes/{name}: freeze must be false")
        if name in SHORTCUT_RECEIPTS:
            for key in ("discharges_lemma", "certified_C_H"):
                if obj.get(key) is not False:
                    problems.append(f"math_status_probes/{name}: {key} must be false")
    index_path = os.path.join(probes_dir, "INVENTABLE_PROBES_INDEX.json")
    if not os.path.isfile(index_path):
        problems.append("INVENTABLE_PROBES_INDEX.json: missing")
    else:
        try:
            index = load_json(index_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"INVENTABLE_PROBES_INDEX.json: {exc}")
        else:
            if index.get("discharges_OBL_H5_JETMOD") is not False:
                problems.append("INVENTABLE_PROBES_INDEX.json: discharges_OBL_H5_JETMOD must be false")
            if index.get("lemma_closed") is not False:
                problems.append("INVENTABLE_PROBES_INDEX.json: lemma_closed must be false")
            if index.get("OBL_H5_JETMOD") != "OPEN":
                problems.append("INVENTABLE_PROBES_INDEX.json: OBL_H5_JETMOD must be OPEN")
            if index.get("freeze") is not False:
                problems.append("INVENTABLE_PROBES_INDEX.json: freeze must be false")
            if index.get("discharges_lemma") is not False:
                problems.append("INVENTABLE_PROBES_INDEX.json: discharges_lemma must be false")
            if index.get("certified_C_H") is not False:
                problems.append("INVENTABLE_PROBES_INDEX.json: certified_C_H must be false")
            if index.get("inventable_attempt_accepted") is not False:
                problems.append(
                    "INVENTABLE_PROBES_INDEX.json: inventable_attempt_accepted must be false"
                )
            if index.get("disposition") != "OPEN_HOLD":
                problems.append("INVENTABLE_PROBES_INDEX.json: disposition must stay OPEN_HOLD")
            if index.get("pr12_action") != "NONE_left_unmerged":
                problems.append("INVENTABLE_PROBES_INDEX.json: PR #12 must stay unmerged")
            rows = index.get("receipts")
            by_file = {}
            if isinstance(rows, list):
                by_file = {
                    row.get("file"): row
                    for row in rows
                    if isinstance(row, dict)
                }
            else:
                problems.append("INVENTABLE_PROBES_INDEX.json: receipts must be a list")
            walls = index.get("named_walls_only")
            if not isinstance(walls, list):
                problems.append("INVENTABLE_PROBES_INDEX.json: named_walls_only must be a list")
                walls = []
            for name, status in expected.items():
                row = by_file.get(name)
                if not isinstance(row, dict):
                    problems.append(f"INVENTABLE_PROBES_INDEX.json: missing receipt {name}")
                    continue
                if row.get("status") != status:
                    problems.append(
                        f"INVENTABLE_PROBES_INDEX.json: {name} status must be {status}"
                    )
                wall = row.get("named_wall")
                if not isinstance(wall, str) or wall not in walls:
                    problems.append(
                        f"INVENTABLE_PROBES_INDEX.json: named wall for {name} missing"
                    )


def check_packet(packet_dir: str) -> list:
    problems: list = []
    if not os.path.isdir(packet_dir):
        return [f"{packet_dir}: packet directory is missing"]
    names = set(os.listdir(packet_dir))
    missing = sorted(EXPECTED_NAMES - names)
    extra = sorted(names - EXPECTED_NAMES)
    if missing:
        problems.append(f"packet: missing {missing}")
    if extra:
        problems.append(f"packet: unexpected files {extra}")
    for name in sorted(names):
        path = os.path.join(packet_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            problems.append(f"{name}: packet entries must be regular files")

    packet_path = os.path.join(packet_dir, "PACKET.json")
    snapshot_path = os.path.join(packet_dir, "math_console_snapshot.json")
    packet = None
    if os.path.isfile(packet_path):
        try:
            packet = load_json(packet_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"PACKET.json: {exc}")
    if packet is not None:
        check_packet_schema(packet, problems)
        walk_flags(packet, "", problems, "PACKET.json")
    if os.path.isfile(snapshot_path):
        try:
            snapshot = load_json(snapshot_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"math_console_snapshot.json: {exc}")
        else:
            check_snapshot(snapshot, problems)
            walk_flags(snapshot, "", problems, "math_console_snapshot.json")

    if isinstance(packet, dict) and isinstance(packet.get("transcriptions"), dict):
        for name in TRANSCRIPTION_NAMES:
            spec = packet["transcriptions"].get(name)
            path = os.path.join(packet_dir, name)
            if not isinstance(spec, dict) or not os.path.isfile(path):
                problems.append(f"{name}: transcription pin missing")
                continue
            data = open(path, "rb").read()
            digest = hashlib.sha256(data).hexdigest()
            if spec.get("sha256") != digest or spec.get("bytes") != len(data):
                problems.append(f"{name}: sha256/bytes drifted from PACKET.json")

    for name, phrases in {"README.md": README_PHRASES, **NOTE_PHRASES}.items():
        path = os.path.join(packet_dir, name)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        flat = re.sub(r"\s+", " ", text)
        for phrase in phrases:
            if phrase not in flat:
                problems.append(f"{name}: missing required phrase {phrase!r}")
        check_assignment_text(name, text, problems)

    console_path = os.path.join(packet_dir, "math_console.py")
    if os.path.isfile(console_path):
        source = open(console_path, encoding="utf-8").read()
        for phrase in CONSOLE_PHRASES:
            if phrase not in source:
                problems.append(f"math_console.py: missing {phrase!r}")
        check_assignment_text("math_console.py", source, problems)
    if os.path.isfile(snapshot_path):
        check_assignment_text(
            "math_console_snapshot.json",
            open(snapshot_path, encoding="utf-8").read(),
            problems,
        )
    if os.path.isfile(packet_path):
        check_assignment_text(
            "PACKET.json",
            open(packet_path, encoding="utf-8").read(),
            problems,
        )
    return problems


def main(argv: list | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        default=None,
        help="packet directory (default: docs/math_status under this repository)",
    )
    args = parser.parse_args(argv)
    packet_dir = args.packet or os.path.join(root, "docs", "math_status")
    problems = check_packet(packet_dir)
    check_inventable_probes(root, problems)
    for problem in problems:
        print(f"PROBLEM {problem}")
    print(
        "math_status_check: "
        f"problems={len(problems)} disposition=OPEN_HOLD "
        "lemma_closed=false prizes_solved=false independence_credit=0"
    )
    print(
        "math_status_check: does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF; "
        "a display is not a certified enclosure; a green checker is not obligation "
        "discharge; bridge stays PROPOSED / NOT DEPLOYED."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
