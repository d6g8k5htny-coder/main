#!/usr/bin/env python3
"""Inventable JETMOD instrumentation STATUS — PARTIAL / REFUSED_NOT_24JET only.

Fills local code_prototypes inventory rows previously marked `?`:
  first_band_proto / first_band_smoke / first_band_multi_gram → PARTIAL
  multi_jet_band / g12_ext_named → REFUSED_NOT_24JET

PARTIAL and REFUSED_NOT_24JET are honesty labels, not discharge, not a source
of truth, and not FREEZE. Source-named subset stays 8. A later hardening tip
does not upgrade these tokens to PRESENT or SUCCESS and is not a re-run.
Does not invent a 24-jet roster, does not promote display/κ, does not claim OBL
discharge, does not merge PR #12, does not reopen skim-trap PRs #7/#8.
Green ≠ discharge.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _now() -> tuple[str, str]:
    utc = datetime.now(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ"), utc.strftime("%Y-%m-%d %H:%M UTC")


def _write(name: str, body: dict[str, Any]) -> Path:
    path = HERE / name
    text = json.dumps(body, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"WROTE {path.name} status={body.get('status')}")
    return path


def _common(utc: str, human: str) -> dict[str, Any]:
    return {
        "inventable_attempt_accepted": False,
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "works": False,
        "certified_C_H": False,
        "prizes_solved": 0,
        "generated_at_utc": utc,
        "generated_at_human": human,
    }


def probe_first_band_proto() -> dict[str, Any]:
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_instrumentation_status_jetmod_first_band_proto",
        "inventory_row": "jetmod_first_band_receipt.json",
        "local_code_prototypes_lane": "jetmod_first_band_proto",
        "named_wall": "Inventory row jetmod_first_band_receipt.json previously marked ?",
        "status": "PARTIAL",
        "inventable_attempt": (
            "Invent that first_band_proto c2-only band enclosure (jets_done=1) is the "
            "certified 24-jet OBL-H5-JETMOD discharge."
        ),
        "refusal_reason": (
            "first_band_proto is PARTIAL instrumentation only (c2 on one r-band). "
            "jets_done=1 of 24; display residual / struct κ comparison is diagnostic. "
            "Not a 24-jet certified band enclosure."
        ),
        "exact_missing_object": {
            "name": "certified_24jet_band_enclosure_with_uniform_lattice_tail_constants",
            "also_missing": [
                "Drive_sourced_24jet_roster_with_p_J",
                "explicit_interval_map_F_G12box_to_Rplus",
            ],
        },
        **_common(utc, human),
        "constraints_honored": [
            "inventory_row_only",
            "no_24jet_roster_invention",
            "no_display_kappa_promotion",
            "no_discharge_flip",
            "honest_PARTIAL_receipt",
            "PR12_left_unmerged",
        ],
    }


def probe_first_band_smoke() -> dict[str, Any]:
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_instrumentation_status_jetmod_first_band_interval_r",
        "inventory_row": "jetmod_first_band_smoke_receipt.json",
        "local_code_prototypes_lane": "jetmod_first_band_interval_r",
        "named_wall": "Inventory row jetmod_first_band_smoke_receipt.json previously marked ?",
        "status": "PARTIAL",
        "inventable_attempt": (
            "Invent that first_band_smoke / interval-r c2 smoke (jets_done=1) discharges OBL-H5-JETMOD."
        ),
        "refusal_reason": (
            "Smoke receipt is PARTIAL: certifies_24jet_band=false; discharges_OBL_H5_JETMOD=false; "
            "lemma_closed=false. Thin-cell / smoke ≠ certified 24-jet band."
        ),
        "exact_missing_object": {
            "name": "certified_24jet_band_enclosure_with_uniform_lattice_tail_constants",
            "also_missing": [
                "Drive_sourced_24jet_roster_with_p_J",
                "explicit_interval_map_F_G12box_to_Rplus",
            ],
        },
        **_common(utc, human),
        "constraints_honored": [
            "inventory_row_only",
            "no_24jet_roster_invention",
            "no_display_kappa_promotion",
            "no_discharge_flip",
            "honest_PARTIAL_receipt",
            "PR12_left_unmerged",
        ],
    }


def probe_first_band_multi_gram() -> dict[str, Any]:
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_instrumentation_status_jetmod_first_band_multi_gram_v1",
        "inventory_row": "jetmod_first_band_multi_gram_receipt.json",
        "local_code_prototypes_lane": "jetmod_first_band_multi_gram_v1",
        "named_wall": "Inventory row jetmod_first_band_multi_gram_receipt.json previously marked ?",
        "status": "PARTIAL",
        "inventable_attempt": (
            "Invent that multi-gram G00/G0S/c2 blocks (jets_done_gram_blocks=3) are the full 24-jet set."
        ),
        "refusal_reason": (
            "Multi-gram receipt is PARTIAL Gram scaffolding for c2 — explicitly NOT the full 24-jet set. "
            "OBL-H5-JETMOD remains OPEN."
        ),
        "exact_missing_object": {
            "name": "certified_24jet_band_enclosure_with_uniform_lattice_tail_constants",
            "also_missing": [
                "Drive_sourced_24jet_roster_with_p_J",
                "explicit_interval_map_F_G12box_to_Rplus",
            ],
        },
        **_common(utc, human),
        "constraints_honored": [
            "inventory_row_only",
            "no_24jet_roster_invention",
            "no_display_kappa_promotion",
            "no_discharge_flip",
            "honest_PARTIAL_receipt",
            "PR12_left_unmerged",
        ],
    }


def probe_multi_jet_band() -> dict[str, Any]:
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_instrumentation_status_jetmod_multi_jet_band",
        "inventory_row": "jetmod_multi_jet_band_receipt.json",
        "local_code_prototypes_lane": "jetmod_multi_jet_band",
        "named_wall": "Inventory row jetmod_multi_jet_band_receipt.json previously marked ?",
        "status": "REFUSED_NOT_24JET",
        "jets_done_in_row": 6,
        "jets_total_obligation": 24,
        "inventable_attempt": (
            "Invent that six MS-diag scaled jets (c2_f..c2_fyy) are the still-unenumerated full "
            "24-jet roster and discharge OBL-H5-JETMOD."
        ),
        "refusal_reason": (
            "multi_jet_band is REFUSED_NOT_24JET: jets_done=6 of jets_total_claimed_by_OBL=24; "
            "full_24_jet_roster_and_powers_p_J remains BLOCKED_UNENUMERATED. Inventing the 24-list "
            "from six MS-diag jets is refused."
        ),
        "exact_missing_object": {
            "name": "Drive_sourced_24jet_roster_with_p_J",
            "also_missing": [
                "order_gt_2_jet_API_DER_JETS_MS_JETS_H",
                "certified_24jet_band_enclosure_with_uniform_lattice_tail_constants",
            ],
        },
        **_common(utc, human),
        "constraints_honored": [
            "inventory_row_only",
            "no_24jet_roster_invention",
            "no_display_kappa_promotion",
            "no_discharge_flip",
            "honest_REFUSED_NOT_24JET_receipt",
            "PR12_left_unmerged",
        ],
    }


def probe_g12_ext_named() -> dict[str, Any]:
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_instrumentation_status_jetmod_g12_ext_named",
        "inventory_row": "jetmod_g12_ext_named_receipt.json",
        "local_code_prototypes_lane": "jetmod_g12_ext_named",
        "named_wall": "Inventory row jetmod_g12_ext_named_receipt.json previously marked ?",
        "status": "REFUSED_NOT_24JET",
        "jets_done_in_row": 8,
        "jets_total_obligation": 24,
        "inventable_attempt": (
            "Invent further named jets toward 24 from kappa_c2 + s_f_fx without a Drive/PROMOTE "
            "roster, or promote display/κ widths as certified enclosure."
        ),
        "refusal_reason": (
            "g12_ext_named is REFUSED_NOT_24JET: jets_named_total_now=8 of 24; order>2 API BLOCKED; "
            "full_24_jet_roster_p_J BLOCKED_UNENUMERATED. STOP: inventing named jets toward 24 without "
            "a source roster invents the 24-list. Display/κ falsifier is diagnostic only."
        ),
        "exact_missing_object": {
            "name": "Drive_sourced_24jet_roster_with_p_J",
            "also_missing": [
                "order_gt_2_jet_API_DER_JETS_MS_JETS_H",
                "certified_24jet_band_enclosure_with_uniform_lattice_tail_constants",
            ],
        },
        **_common(utc, human),
        "constraints_honored": [
            "inventory_row_only",
            "no_24jet_roster_invention",
            "no_display_kappa_promotion",
            "no_discharge_flip",
            "honest_REFUSED_NOT_24JET_receipt",
            "PR12_left_unmerged",
        ],
    }


def main() -> int:
    probes = [
        ("inventable_first_band_proto_PARTIAL_receipt.json", probe_first_band_proto),
        ("inventable_first_band_smoke_PARTIAL_receipt.json", probe_first_band_smoke),
        ("inventable_first_band_multi_gram_PARTIAL_receipt.json", probe_first_band_multi_gram),
        ("inventable_multi_jet_band_REFUSED_NOT_24JET_receipt.json", probe_multi_jet_band),
        ("inventable_g12_ext_named_REFUSED_NOT_24JET_receipt.json", probe_g12_ext_named),
    ]
    index: dict[str, Any] = {
        "schema": "q0.inventable-jetmod-instrumentation-status/v1",
        "as_of_note": "48h coding lane under Dylan autonomy; instrumentation STATUS vocabulary only for inventory ? rows",
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "certified_C_H": False,
        "prizes_solved": 0,
        "OBL_H5_JETMOD": "OPEN",
        "inventory_rows_filled": [
            "jetmod_first_band_receipt.json: PARTIAL",
            "jetmod_first_band_smoke_receipt.json: PARTIAL",
            "jetmod_first_band_multi_gram_receipt.json: PARTIAL",
            "jetmod_multi_jet_band_receipt.json: REFUSED_NOT_24JET",
            "jetmod_g12_ext_named_receipt.json: REFUSED_NOT_24JET",
        ],
        "receipts": [],
        "does_not_establish": (
            "These instrumentation STATUS receipts do not discharge OBL-H5-JETMOD, do not FREEZE, "
            "do not invent a 24-jet roster, do not promote display/κ, do not merge PR #12, and do not "
            "reopen certificate skim-trap PRs. Green ≠ discharge."
        ),
    }
    for name, fn in probes:
        body = fn()
        assert body.get("discharges_OBL_H5_JETMOD") is False
        assert body.get("lemma_closed") is False
        assert body.get("inventable_attempt_accepted") is False
        assert body.get("certified_C_H") is False
        assert body.get("prizes_solved") == 0
        assert body["status"] in ("PARTIAL", "REFUSED_NOT_24JET")
        path = _write(name, body)
        index["receipts"].append(
            {
                "file": name,
                "status": body["status"],
                "inventory_row": body["inventory_row"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    idx_path = HERE / "INVENTABLE_INSTRUMENTATION_STATUS_INDEX.json"
    idx_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {idx_path.name} receipts={len(index['receipts'])} discharges=false")
    print("OBL-H5-JETMOD remains OPEN. No discharge flips. No 24-jet invention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
