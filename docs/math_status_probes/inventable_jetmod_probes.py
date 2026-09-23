#!/usr/bin/env python3
"""Inventable JETMOD probes — honest REFUSED / EMPTY / ABSENT receipts only.

Named walls only (sibling sweep CLOSED EMPTY; do not invent new walls):
  1. Interval Schur via Ainv → REFUSED_IA_STRADDLES
  2. eval_F(G12_box) → REFUSED (missing explicit_interval_map_F_G12box_to_Rplus)
  3. Joint (r,y) cancel rewrite → EMPTY
  4. φ(det A)→detgg bridge → ABSENT

Each probe *attempts* an inventable shortcut and refuses it. Never flips
discharges_OBL_H5_JETMOD / lemma_closed / FREEZE. Green ≠ discharge.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CT = "America/Chicago"


def _now() -> tuple[str, str]:
    utc = datetime.now(timezone.utc)
    # CT label without inventing DST math beyond fixed offset note
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ"), utc.strftime("%Y-%m-%d %H:%M UTC")


def _write(name: str, body: dict[str, Any]) -> Path:
    path = HERE / name
    text = json.dumps(body, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    body_meta = {
        "path": str(path.relative_to(HERE.parent.parent)),
        "sha256": digest,
        "bytes": len(text.encode()),
    }
    print(f"WROTE {path.name} status={body.get('status')} sha256={digest[:16]}…")
    return path


def probe_interval_schur_ainv() -> dict[str, Any]:
    """Inventable attempt: pretend Ainv Schur alone certifies positive detgg on a fat (r,y) cell."""
    utc, human = _now()
    inventable_claim = (
        "Invent φ≡0 and declare Schur S1=Grr−C Ainv C^T automatically "
        "non-straddling for any positive-width (r,y) cell."
    )
    # Honest refusal: inventable claim is not a certificate; named wall stands.
    return {
        "prototype": "inventable_jetmod_probe_interval_schur_ainv",
        "named_wall": "Interval Schur via Ainv: REFUSED_IA_STRADDLES",
        "status": "REFUSED_IA_STRADDLES",
        "inventable_attempt": inventable_claim,
        "inventable_attempt_accepted": False,
        "refusal_reason": (
            "Documented Schur via Ainv is not an inventable non-straddle warranty. "
            "Prior wall: positive-width (r,y) detgg straddles (width≈1.26). "
            "Inventing φ≡0 or a fantasy cancellation is refused."
        ),
        "exact_missing_object": {
            "name": "correlated_C_Ainv_Ct_cancellation_under_joint_r_y",
            "also_missing": [
                "cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM",
                "Schur_detgg_non_straddle_enclosure_on_positive_width_ry_via_interval_C_Grr_Ainv",
            ],
        },
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "works": False,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": [
            "named_wall_only",
            "no_new_wall_invention",
            "no_discharge_flip",
            "honest_REFUSED_receipt",
        ],
    }


def probe_eval_F_G12box() -> dict[str, Any]:
    inventable_claim = (
        "Invent eval_F(G12_box):=max(|Gij|) or Lip·width as a stand-in for "
        "explicit_interval_map_F_G12box_to_Rplus."
    )
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_probe_eval_F_G12box",
        "named_wall": "eval_F(G12_box): REFUSED (missing explicit_interval_map_F_G12box_to_Rplus)",
        "status": "REFUSED",
        "inventable_attempt": inventable_claim,
        "inventable_attempt_accepted": False,
        "refusal_reason": (
            "No proof-grade F:G12-box→R+ map exists in corpus/API. "
            "Inventing max-norm / Lip·width / endpoint sampling as F is refused."
        ),
        "exact_missing_object": {
            "name": "explicit_interval_map_F_G12box_to_Rplus",
            "aka": ["eval_F(G12_box)", "certified_F_on_G12_band_box"],
        },
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "works": False,
        "implemented_runner_callable": False,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": [
            "named_wall_only",
            "no_new_wall_invention",
            "no_discharge_flip",
            "honest_REFUSED_receipt",
        ],
    }


def probe_joint_ry_cancel() -> dict[str, Any]:
    inventable_claim = (
        "Invent a joint-(r,y) cancel identity for detgg=ad−c² / C Ainv C^T "
        "from point-r scales alone."
    )
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_probe_joint_ry_cancel",
        "named_wall": "Joint (r,y) cancel rewrite: EMPTY",
        "status": "EMPTY",
        "found": False,
        "inventable_attempt": inventable_claim,
        "inventable_attempt_accepted": False,
        "refusal_reason": (
            "Corpus hunt remains EMPTY for a documented joint-(r,y) cancel rewrite. "
            "Inventing one from point-r numerics is refused."
        ),
        "exact_missing_object": {
            "name": "cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM",
            "also_missing": ["correlated_C_Ainv_Ct_cancellation_under_joint_r_y"],
        },
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "works": False,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": [
            "named_wall_only",
            "no_new_wall_invention",
            "no_discharge_flip",
            "honest_EMPTY_receipt",
        ],
    }


def probe_phi_bridge() -> dict[str, Any]:
    inventable_claim = (
        "Invent φ such that det(Σ_gg)=φ(det A, det T, det G6) with a free r^α factor."
    )
    utc, human = _now()
    return {
        "prototype": "inventable_jetmod_probe_phi_bridge",
        "named_wall": "φ(det A)→detgg bridge: ABSENT",
        "status": "ABSENT",
        "found": False,
        "documented_phi_bridge": False,
        "inventable_attempt": inventable_claim,
        "inventable_attempt_accepted": False,
        "refusal_reason": (
            "No documented φ:det A→detgg bridge. Schur uses Ainv, not φ(det A). "
            "Inventing φ/r^α is refused."
        ),
        "exact_missing_object": {
            "name": "det(Sigma_gg)=phi(det A, det T, det G6, A_ser)",
        },
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "works": False,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": [
            "named_wall_only",
            "no_new_wall_invention",
            "no_discharge_flip",
            "honest_ABSENT_receipt",
        ],
    }


def main() -> int:
    probes = [
        ("inventable_interval_schur_ainv_REFUSED_receipt.json", probe_interval_schur_ainv),
        ("inventable_eval_F_G12box_REFUSED_receipt.json", probe_eval_F_G12box),
        ("inventable_joint_ry_cancel_EMPTY_receipt.json", probe_joint_ry_cancel),
        ("inventable_phi_bridge_ABSENT_receipt.json", probe_phi_bridge),
    ]
    index = {
        "schema": "q0.inventable-jetmod-probes/v1",
        "as_of_note": "48h coding lane under Bot 5; named walls only",
        "discharges_OBL_H5_JETMOD": False,
        "lemma_closed": False,
        "freeze": False,
        "OBL_H5_JETMOD": "OPEN",
        "named_walls_only": [
            "Interval Schur via Ainv: REFUSED_IA_STRADDLES",
            "eval_F(G12_box): REFUSED (missing explicit_interval_map_F_G12box_to_Rplus)",
            "Joint (r,y) cancel rewrite: EMPTY",
            "φ(det A)→detgg bridge: ABSENT",
        ],
        "receipts": [],
        "does_not_establish": (
            "These inventable probes do not discharge OBL-H5-JETMOD, do not FREEZE, "
            "do not certify an enclosure, and do not invent new walls. Green ≠ discharge."
        ),
    }
    for name, fn in probes:
        body = fn()
        assert body.get("discharges_OBL_H5_JETMOD") is False
        assert body.get("lemma_closed") is False
        assert body.get("inventable_attempt_accepted") is False
        path = _write(name, body)
        index["receipts"].append(
            {
                "file": name,
                "status": body["status"],
                "named_wall": body["named_wall"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    idx_path = HERE / "INVENTABLE_PROBES_INDEX.json"
    idx_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {idx_path.name} receipts={len(index['receipts'])} discharges=false")
    print("OBL-H5-JETMOD remains OPEN. No discharge flips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
