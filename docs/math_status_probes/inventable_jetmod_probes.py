#!/usr/bin/env python3
"""Inventable JETMOD probes — honest REFUSED / EMPTY / ABSENT receipts only.

Sibling-sweep named walls (CLOSED EMPTY; do not invent new walls or lemmas):
  1. Interval Schur via Ainv → REFUSED_IA_STRADDLES
  2. eval_F(G12_box) → REFUSED (missing explicit_interval_map_F_G12box_to_Rplus)
  3. Joint (r,y) cancel rewrite → EMPTY
  4. φ(det A)→detgg bridge → ABSENT

Tip-aligned shortcut refusals (already-recorded false-progress paths; not in #15):
  5. 24-jet roster + p_J without Drive enumeration → REFUSED_NOT_24JET
  6. Promote display residual / struct κ as certified enclosure → REFUSED
  7. Merge draft PR #12 as status catch-up OR claim RUNG2/3 discharges JETMOD → REFUSED

Each probe *attempts* an inventable shortcut and refuses it. Never flips
discharges_OBL_H5_JETMOD / discharges_lemma / lemma_closed / certified_C_H /
FREEZE / inventable_attempt_accepted. Green ≠ discharge. Draft PR #12 is not
merged or rebased.

Vault ids and quarantine paths do not activate an inventable source of truth.
`tools/vault_hygiene_check.py` and `tools/quarantine_check.py` are engineering
hygiene; a green run is not discharge. A KNOWN line from
`tools/registers_check.py` whose allowlist text says "Accepted as-is" for
class `EXISTING_CONTAINER` on `Q-R17-VAULT` leaves
`inventable_attempt_accepted` false. This runner does not read the vault
and does not edit `quarantine/EXCLUSIONS.json`. See this directory's README.

The local SIDE24 prep commands are not an inventable source of truth. SIDE24
has no new source-of-truth carrier after 2026-08-06. Objects those notes mark
ABSENT stay ABSENT. Quarantine is not a source of truth.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CT = "America/Chicago"
# Generation-time pin only (historical). Not the current hardening LOCK.
# Do not rewrite BASE_TIP to a later SHA unless these probes are actually
# re-run and that re-run is stated. Advancing the tip does not upgrade
# REFUSED / EMPTY / ABSENT / REFUSED_NOT_24JET into PRESENT or SUCCESS.
# Honesty note LOCK (not a re-run): b3da6688a55d34681bb27f17ba6c6c5e16ad534c.
BASE_TIP = "1ea0ae8183fb0459c6678243946295518fded1ba"
BASE_BRANCH = "chatgpt/drive-github-hardening-20260919"


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


def _shortcut_flags() -> dict[str, Any]:
    """Fail-closed flags for tip-aligned shortcut receipts. All stay false."""
    return {
        "inventable_attempt_accepted": False,
        "discharges_OBL_H5_JETMOD": False,
        "discharges_lemma": False,
        "lemma_closed": False,
        "certified_C_H": False,
        "freeze": False,
        "works": False,
    }


def _with_shortcut_flags(body: dict[str, Any]) -> dict[str, Any]:
    """Place fail-closed flags beside the inventable attempt, matching other receipts."""
    out: dict[str, Any] = {}
    for key, value in body.items():
        out[key] = value
        if key == "inventable_attempt":
            out.update(_shortcut_flags())
    return out


def _shortcut_constraints() -> list[str]:
    return [
        "named_wall_or_named_nonclaim_only",
        "no_lemma_invention",
        "no_new_wall_invention",
        "no_discharge_flip",
        "no_RN_slice_expansion",
        "PR12_left_unmerged",
        "honest_REFUSED_receipt",
    ]


def probe_24jet_roster_without_drive_list() -> dict[str, Any]:
    """Refuse inventing a 24-jet roster and p_J without a Drive enumeration."""
    inventable_claim = (
        "Invent a 24-jet roster and p_J weights from OBL text alone, padding "
        "beyond the 8 source-named jets (6 MS-diag + kappa_c2 + s_f_fx) without "
        "a Drive H5_ANALYTIC_ADVANCE / PROMOTE enumeration."
    )
    utc, human = _now()
    body: dict[str, Any] = {
        "prototype": "inventable_jetmod_probe_24jet_roster_without_Drive_list",
        "named_wall": (
            "24-jet roster + p_J without Drive enumeration: REFUSED_NOT_24JET"
        ),
        "status": "REFUSED_NOT_24JET",
        "refused_not_24jet": True,
        "partial_roster": False,
        "roster_invented": False,
        "inventable_attempt": inventable_claim,
        "refusal_reason": (
            "REFUSED_NOT_24JET. Inventing named jets toward 24 without a Drive "
            "source roster would invent the 24-list. This receipt is not PARTIAL "
            "and it does not carry a jet roster. Recorded wall in STATUS_JETMOD: "
            "full 24-jet roster + p_J is unenumerated (Drive H5_ANALYTIC_ADVANCE; "
            "PROMOTE never lists them). Order>2 jets (fxxx…) stay blocked: "
            "DER/JETS_MS/JETS_H stop at order 2 (h5_kernel.py:33-35, cited by "
            "STATUS_JETMOD; that file is not vendored here and this probe does not "
            "re-quote it). The recorded source-named subset is 8 jets, which is "
            "not a 24-jet roster."
        ),
        "exact_missing_object": {
            "name": "Drive_sourced_24jet_roster_with_p_J",
            "also_missing": [
                "order_gt_2_jet_API_DER_JETS_MS_JETS_H",
                "H5_ANALYTIC_ADVANCE_or_PROMOTE_enumeration_of_24",
            ],
        },
        "recorded_wall_cite": {
            "path": "docs/math_status/STATUS_JETMOD.md",
            "fact": (
                "STOP honored: roster + p_J missing; order>2 API missing. "
                "jets_done=8 (6 MS-diag + kappa_c2 + s_f_fx). "
                "h5_kernel.py is not in this tree."
            ),
        },
        "cited_existing_receipts_by_name_only": [
            "jetmod_multi_jet_band_receipt.json",
            "jetmod_g12_ext_named_receipt.json",
        ],
        "receipts_vendored_in_this_tree": False,
        "does_not_establish": (
            "REFUSED_NOT_24JET. Does not enumerate 24 jets, does not invent p_J, "
            "does not publish a PARTIAL roster, does not close OBL-H5-JETMOD, "
            "does not set lemma_closed or FREEZE."
        ),
        "aligned_to_base_branch": BASE_BRANCH,
        "aligned_to_base_tip": BASE_TIP,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": _shortcut_constraints(),
    }
    return _with_shortcut_flags(body)


def probe_promote_display_residual_struct_kappa() -> dict[str, Any]:
    """Refuse promoting display residual or struct κ into a certified enclosure."""
    inventable_claim = (
        "Promote STATUS display residual bound 3.4e-6 (UPDATE §5) and/or "
        "struct κ=1/8 halfwidth ≈7.81e-5 from jetmod_g12_ext_named / multi_jet "
        "band widths into a certified enclosure that discharges OBL-H5-JETMOD."
    )
    utc, human = _now()
    body: dict[str, Any] = {
        "prototype": "inventable_jetmod_probe_promote_display_residual_struct_kappa",
        "named_wall": (
            "Promote display residual / struct κ as certified enclosure: "
            "REFUSED (display ≠ certified)"
        ),
        "status": "REFUSED",
        "inventable_attempt": inventable_claim,
        "refusal_reason": (
            "Fail-closed discipline: display ≠ certified enclosure. Recorded "
            "diagnostic widths (kappa_c2 subdiv width ≈10.322061) are ≫ the "
            "display residual 3.4e-6 and struct κ=1/8 halfwidth ≈7.81e-5. The "
            "falsifier shape is diagnostic only. PACKET.json "
            "display_is_certified_enclosure is false and OBL-H5-JETMOD.grade is "
            "display_only. Promoting display residual or struct κ to a certificate "
            "is refused."
        ),
        "exact_missing_object": {
            "name": "certified_enclosure_of_OBL_H5_JETMOD_jets_under_interval_r",
            "also_missing": [
                "proof_grade_band_cover_F_G12",
                "Lip_F_G12",
                "explicit_interval_map_F_G12box_to_Rplus",
            ],
        },
        "packet_flags_cited": {
            "path": "docs/math_status/PACKET.json",
            "display_is_certified_enclosure": False,
            "OBL-H5-JETMOD.grade": "display_only",
            "discharges_OBL_H5_JETMOD": False,
        },
        "cited_existing_receipts_by_name_only": [
            "jetmod_g12_ext_named_receipt.json",
            "jetmod_multi_jet_band_receipt.json",
            "jetmod_certified_F_G12_band_and_Lip_F_G12_receipt.json",
        ],
        "receipts_vendored_in_this_tree": False,
        "does_not_establish": (
            "Does not certify an enclosure, does not promote display residual or "
            "struct κ, does not discharge OBL-H5-JETMOD."
        ),
        "aligned_to_base_branch": BASE_BRANCH,
        "aligned_to_base_tip": BASE_TIP,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": _shortcut_constraints(),
    }
    return _with_shortcut_flags(body)


def probe_merge_pr12_or_rung_discharge() -> dict[str, Any]:
    """Refuse merging draft PR #12 or treating RUNG2/3 / green CI as discharge."""
    inventable_claim = (
        "Merge draft PR #12 ([DRAFT] Fail-closed JETMOD/RN-UNIF status + "
        "math_console) as a status catch-up that closes or discharges "
        "OBL-H5-JETMOD, and/or claim that RUNG2/RUNG3 green or CI green "
        "discharges JETMOD."
    )
    utc, human = _now()
    body: dict[str, Any] = {
        "prototype": "inventable_jetmod_probe_merge_PR12_or_rung_discharge",
        "named_wall": (
            "Merge draft PR #12 as status catch-up OR claim RUNG2/3 discharges "
            "JETMOD: REFUSED"
        ),
        "status": "REFUSED",
        "inventable_attempt": inventable_claim,
        "refusal_reason": (
            "PR #12 stays draft and unmerged. Its body says \"Draft only. Do not "
            "merge.\" Merging a status mirror does not discharge OBL-H5-JETMOD. "
            "PACKET.json rung2_or_rung3_discharges_jetmod is false and "
            "green_ci_discharges_obligations is false. STATUS_JETMOD: RUNG2/3 do "
            "NOT discharge JETMOD. Green ≠ discharge. This probe does not merge "
            "or rebase PR #12."
        ),
        "exact_missing_object": {
            "name": "obligation_discharge_predicate_for_OBL_H5_JETMOD",
            "also_missing": [
                "RUNG2_or_RUNG3_as_JETMOD_discharge_license",
                "green_CI_as_obligation_discharge",
            ],
        },
        "pr12_observed": {
            "html_url": "https://github.com/d6g8k5htny-coder/main/pull/12",
            "title": "[DRAFT] Fail-closed JETMOD/RN-UNIF status + math_console (OPEN/HOLD)",
            "draft": True,
            "merged": False,
            "state": "OPEN",
            "mergeable": "CONFLICTING",
            "merge_state_status": "DIRTY",
            "head_sha": "b8b8634fb72ae09383ee8bb50bcf2b8c17a43bb0",
            "body_says": "Draft only. Do not merge.",
            "action_taken": "NONE_left_unmerged",
            "observed_against_base_tip": BASE_TIP,
        },
        "packet_flags_cited": {
            "path": "docs/math_status/PACKET.json",
            "rung2_or_rung3_discharges_jetmod": False,
            "green_ci_discharges_obligations": False,
            "freeze": False,
            "lemma_closed": False,
            "disposition": "OPEN_HOLD",
        },
        "does_not_establish": (
            "Does not merge PR #12, does not rebase PR #12, does not claim "
            "RUNG2/3 discharge JETMOD, does not FREEZE, does not close "
            "OBL-H5-JETMOD. Green ≠ discharge."
        ),
        "aligned_to_base_branch": BASE_BRANCH,
        "aligned_to_base_tip": BASE_TIP,
        "generated_at_utc": utc,
        "generated_at_human": human,
        "constraints_honored": _shortcut_constraints(),
    }
    return _with_shortcut_flags(body)


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
        (
            "inventable_24jet_roster_without_Drive_list_REFUSED_NOT_24JET_receipt.json",
            probe_24jet_roster_without_drive_list,
        ),
        (
            "inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json",
            probe_promote_display_residual_struct_kappa,
        ),
        (
            "inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json",
            probe_merge_pr12_or_rung_discharge,
        ),
    ]
    # Checked-in INVENTABLE_PROBES_INDEX.json carries extra honesty fields
    # (HISTORICAL_NONCURRENT, hardening_tip_observed_at_edit) that this
    # generator does not emit. Running main() rewrites receipts; a later tip
    # is not a reason to bump BASE_TIP without an honest re-run.
    index = {
        "schema": "q0.inventable-jetmod-probes/v1",
        "as_of_note": (
            "Sibling-sweep named walls plus three tip-aligned shortcut refusals "
            f"on {BASE_BRANCH} @ {BASE_TIP}. No new lemmas. PR #12 left unmerged."
        ),
        "aligned_to_base_branch": BASE_BRANCH,
        "aligned_to_base_tip": BASE_TIP,
        "discharges_OBL_H5_JETMOD": False,
        "discharges_lemma": False,
        "lemma_closed": False,
        "certified_C_H": False,
        "freeze": False,
        "inventable_attempt_accepted": False,
        "OBL_H5_JETMOD": "OPEN",
        "disposition": "OPEN_HOLD",
        "pr12_action": "NONE_left_unmerged",
        "named_walls_only": [
            "Interval Schur via Ainv: REFUSED_IA_STRADDLES",
            "eval_F(G12_box): REFUSED (missing explicit_interval_map_F_G12box_to_Rplus)",
            "Joint (r,y) cancel rewrite: EMPTY",
            "φ(det A)→detgg bridge: ABSENT",
            "24-jet roster + p_J without Drive enumeration: REFUSED_NOT_24JET",
            (
                "Promote display residual / struct κ as certified enclosure: "
                "REFUSED (display ≠ certified)"
            ),
            (
                "Merge draft PR #12 as status catch-up OR claim RUNG2/3 discharges "
                "JETMOD: REFUSED"
            ),
        ],
        "receipts": [],
        "does_not_establish": (
            "These inventable probes do not discharge OBL-H5-JETMOD, do not FREEZE, "
            "do not certify an enclosure, do not invent lemmas, and do not merge or "
            "rebase draft PR #12. Green ≠ discharge. OBL-H5-JETMOD stays OPEN."
        ),
    }
    false_keys = (
        "discharges_OBL_H5_JETMOD",
        "lemma_closed",
        "inventable_attempt_accepted",
        "freeze",
    )
    shortcut_false_keys = false_keys + ("discharges_lemma", "certified_C_H")
    for name, fn in probes:
        body = fn()
        required = (
            shortcut_false_keys
            if any(token in name for token in ("PR12", "24jet", "kappa"))
            else false_keys
        )
        for key in required:
            assert body.get(key) is False, (name, key, body.get(key))
        if body.get("status") not in (
            "REFUSED",
            "REFUSED_IA_STRADDLES",
            "REFUSED_NOT_24JET",
            "EMPTY",
            "ABSENT",
        ):
            raise AssertionError(f"{name} status {body.get('status')}")
        if "24jet" in name:
            if body.get("status") != "REFUSED_NOT_24JET" or body.get("refused_not_24jet") is not True:
                raise AssertionError(f"{name} must be REFUSED_NOT_24JET")
            if body.get("partial_roster") is not False or body.get("roster_invented") is not False:
                raise AssertionError(f"{name} must not invent a partial 24-jet roster")
            if isinstance(body.get("jet_roster"), list) or isinstance(body.get("roster"), list):
                raise AssertionError(f"{name} must not carry a jet roster")
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
