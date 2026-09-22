#!/usr/bin/env python3
"""math_console.py — fail-closed OPEN/HOLD board for this packet.

Reads receipt JSON only from this directory (``docs/math_status/``). This
repository does not mirror the prototype receipts, so a live run reports the
obligations OPEN and the numeric fields empty. ``math_console_snapshot.json``
is a transcribed display from another workspace (2026-09-22) and is never
overwritten.

Float arithmetic on the toy mesh plan is NON-CERTIFYING. The console never
sets ``lemma_closed``, ``prizes_solved`` or ``original_prize_closed`` true,
never sets ``independence_credit`` nonzero, and never discharges
OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.

Usage:
  python3 docs/math_status/math_console.py
  python3 docs/math_status/math_console.py --json
  python3 docs/math_status/math_console.py --budget 0.68 --margin 2e-4
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROTO = Path(__file__).resolve().parent
HISTORICAL_SNAPSHOT = PROTO / "math_console_snapshot.json"

# Receipt names from the 2026-09-21 workspace memos. They are not in this
# packet. A present file would be displayed and then sealed back to OPEN.
CANDIDATES = {
    "jetmod_multi": [
        "jetmod_multi_jet_band_receipt.json",
    ],
    "jetmod_ext": [
        "jetmod_g12_ext_named_receipt.json",
    ],
    "rnu_white_box": [
        "RN_UNIF_WHITE_BOX_GRAD_BOUND_V1_RECEIPT.json",
    ],
    "rnu_envelope": [
        "rnu_white_grad_box_envelope_receipt.json",
    ],
    "rnu_hess": [
        "rnu_white_hess_calibrate_v1_receipt.json",
    ],
    "piece2": [
        "piece2_first_cell_failclosed_receipt.json",
    ],
    "rnu_exact_grad": [
        "rnu_chi2_white_exact_grad_receipt.json",
    ],
}


def _load(name_keys: list[str]) -> tuple[Optional[Path], Optional[dict]]:
    for name in name_keys:
        p = PROTO / name
        if p.is_file():
            try:
                return p, json.loads(p.read_text())
            except Exception as e:
                return p, {"_load_error": str(e)}
    return None, None


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _dig(d: dict, *path, default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class MeshPlan:
    budget: float
    margin: float
    kap_far_center: Optional[float]
    envelope_or_U: Optional[float]
    source: str
    hw_max_toy: Optional[float]
    dd_max_if_theta_zero: Optional[float]
    note: str
    certified: bool


def mesh_plan(envelope: Optional[float], kap: Optional[float], budget: float,
              margin: float, source: str) -> MeshPlan:
    """Toy first-order plan. ``certified`` is always false. Floats are NON-CERTIFYING."""
    note_tail = (
        " Arithmetic is NON-CERTIFYING (float). "
        "Envelope is EMPIRICAL / provisional — not a certificate."
    )
    if envelope is None or kap is None or envelope <= 0:
        return MeshPlan(
            budget, margin, kap, envelope, source, None, None,
            "Need kap_far + |∇κ| envelope to size the first polar ring." + note_tail,
            False,
        )
    room = budget - margin - kap
    if room <= 0:
        return MeshPlan(
            budget, margin, kap, envelope, source, None, None,
            f"No room under budget: kap={kap:.6g} already uses the budget minus margin."
            + note_tail,
            False,
        )
    hw = room / envelope
    return MeshPlan(
        budget=budget,
        margin=margin,
        kap_far_center=kap,
        envelope_or_U=envelope,
        source=source,
        hw_max_toy=hw,
        dd_max_if_theta_zero=2 * hw,
        note=(
            "Toy first-order only (no Hessian remainder). "
            "Engine Δd=0.5 is too coarse if dd_max_if_theta_zero < 0.5."
            + note_tail
        ),
        certified=False,
    )


def build(budget: float, margin: float) -> dict:
    loaded = {k: _load(v) for k, v in CANDIDATES.items()}

    jet_multi_p, jet_multi = loaded["jetmod_multi"]
    jet_ext_p, jet_ext = loaded["jetmod_ext"]
    white_p, white = loaded["rnu_white_box"]
    env_p, env = loaded["rnu_envelope"]
    hess_p, hess = loaded["rnu_hess"]
    p2_p, p2 = loaded["piece2"]
    ex_p, ex = loaded["rnu_exact_grad"]

    contradictions: list[str] = []

    jets_named = None
    max_w = None
    struct_hw = None
    jetmod_blocked = []
    if jet_ext:
        jets_named = jet_ext.get("jets_named_total_now") or jet_ext.get("jets_after")
        max_w = _f(jet_ext.get("combined_max_enclosure_width"))
        struct_hw = _f(jet_ext.get("struct_model_halfwidth") or jet_ext.get("struct_model_half_width"))
        if jet_ext.get("discharges_OBL_H5_JETMOD") is True:
            contradictions.append("jetmod receipt set discharges_OBL_H5_JETMOD true")
        if jet_ext.get("lemma_closed") is True:
            contradictions.append("jetmod receipt set lemma_closed true")
        stop = jet_ext.get("stop_reason") or jet_ext.get("halt_reason")
        if stop:
            jetmod_blocked.append(str(stop)[:240])
    if jet_multi and jets_named is None:
        jets_named = jet_multi.get("jets_after") or jet_multi.get("jets_done")
        max_w = _f(jet_multi.get("max_enclosure_width"))
        struct_hw = _f(jet_multi.get("struct_model_halfwidth"))
        if jet_multi.get("discharges_OBL_H5_JETMOD") is True:
            contradictions.append("jetmod multi receipt set discharges_OBL_H5_JETMOD true")
        if jet_multi.get("lemma_closed") is True:
            contradictions.append("jetmod multi receipt set lemma_closed true")
    jetmod_blocked.append("24-jet roster + p_J unenumerated in H5 sources")
    jetmod_blocked.append("PinFrame DER stops at order 2; no F(G12-band) cover runner")

    abs_grad = None
    if ex:
        abs_grad = _f(ex.get("abs_grad_kap_pair_from_white") or ex.get("abs_grad_kap_pair"))
        if ex.get("lemma_closed") is True:
            contradictions.append("exact-grad receipt set lemma_closed true")
    U = None
    U_source = "missing"
    kap = None
    if white:
        if white.get("lemma_closed") is True:
            contradictions.append("white-box receipt set lemma_closed true")
        claimed = white.get("D3_LEMMA_RN_UNIF")
        if isinstance(claimed, str) and claimed not in ("OPEN", "HOLD"):
            contradictions.append(f"white-box receipt status {claimed!r} ignored; board stays OPEN")
        U = _f(_dig(white, "provisional_box_majorant_grad_kap_pair", "U_abs_grad_kap_pair"))
        if U is not None:
            U_source = "white_box_provisional_U"
        kap = _f(_dig(white, "box", "kap_far_center")) or _f(
            _dig(white, "provisional_box_majorant_grad_kap_pair", "kap_far_center"))
        pt = _dig(white, "certified", "pointwise_exact_grad_kap_pair_at_center") or {}
        if isinstance(pt, dict) and abs_grad is None:
            abs_grad = _f(pt.get("abs_grad_kap_pair") or pt.get("abs_grad"))
        if kap is None and isinstance(pt, dict):
            kap = _f(pt.get("kap_far"))
    env_upper = None
    if env:
        env_upper = _f(env.get("envelope_upper"))
        if kap is None:
            kap = _f(env.get("kap_far_center"))
        if abs_grad is None:
            abs_grad = _f(env.get("grid_max_abs_grad_kap_pair"))
        if U is None and env_upper is not None:
            U = env_upper
            U_source = "empirical_grid_envelope"
        if env.get("lemma_closed") is True:
            contradictions.append("envelope receipt set lemma_closed true")

    plan_U = env_upper if env_upper is not None else U
    plan_src = "empirical_envelope" if env_upper is not None else U_source

    c_h_prior = None
    ratio_op = None
    if hess:
        c_h_prior = _f(hess.get("C_H_prior_provisional") or hess.get("C_H_prior"))
        ratio_op = _f(hess.get("ratio_hess_op_over_rss") or hess.get("ratio_hess_op_over_rss"))
        if hess.get("lemma_closed") is True:
            contradictions.append("hess receipt set lemma_closed true")

    piece2 = "UNWRITTEN"
    if p2:
        piece2 = str(p2.get("piece2_driver_status") or p2.get("piece2_annulus_driver") or "UNWRITTEN")
        if p2.get("lemma_closed") is True:
            contradictions.append("piece2 receipt set lemma_closed true")
        if p2.get("would_certify") is True:
            contradictions.append("piece2 receipt set would_certify true")

    plan = mesh_plan(plan_U, kap, budget, margin, plan_src)

    if white is None and env is None:
        next_action = {
            "id": "rnu_white_box_absent",
            "why": (
                "No whitened box or envelope receipt is in this packet. "
                "The live board therefore has no numeric RN envelope."
            ),
            "command": (
                "UNAVAILABLE in this repository. The 2026-09-21 memo names "
                "code_prototypes/rnu_white_box_grad_bound_v1.py; that script is not mirrored here."
            ),
            "do_not": (
                "Do not flip lemma_closed; do not invent a 24-jet roster; "
                "do not treat CI green as discharge; do not treat a display as a certified enclosure."
            ),
        }
    else:
        next_action = {
            "id": "rnu_tighten_CH",
            "why": (
                "Whitening is the recorded direction of the slack; C_H stays provisional. "
                "A tight Lip-of-Hess bound is still required before any cell certificate."
            ),
            "command": (
                "UNAVAILABLE in this repository. The 2026-09-21 memo names "
                "code_prototypes/rnu_white_hess_calibrate_v1.py; that script is not mirrored here."
            ),
            "do_not": (
                "Do not flip lemma_closed; do not invent a 24-jet roster; "
                "do not treat CI green as discharge; do not treat a display as a certified enclosure."
            ),
        }

    board = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "discipline": "fail-closed; display is not a certified enclosure; Drive is source of truth",
        "arithmetic": "NON-CERTIFYING",
        "paths": {k: (str(v[0].name) if v[0] else None) for k, v in loaded.items()},
        "receipt_contradictions": contradictions,
        "OBL_H5_JETMOD": {
            "status": "OPEN",
            "jets_named": jets_named,
            "jets_claimed_by_OBL": 24,
            "max_enclosure_width": max_w,
            "struct_model_halfwidth": struct_hw,
            "width_exceeds_struct": (max_w is not None and struct_hw is not None and max_w > struct_hw),
            "blocked": jetmod_blocked,
            "lemma_closed": False,
            "discharges_OBL": False,
        },
        "D3_LEMMA_RN_UNIF": {
            "status": "OPEN",
            "lemma_closed": False,
            "abs_grad_kap_pair_white_center": abs_grad,
            "U_or_envelope": U,
            "U_source": U_source,
            "U_certified": False,
            "empirical_envelope_upper": env_upper,
            "C_H_prior_provisional": c_h_prior,
            "measured_hess_op_over_rss": ratio_op,
            "C_H_loose_factor_vs_center": (
                (c_h_prior / ratio_op) if (c_h_prior and ratio_op and ratio_op > 0) else None
            ),
            "piece2_annulus_driver": piece2,
            "discharges_lemma": False,
        },
        "mesh_plan_toy": asdict(plan),
        "next_action": next_action,
        "non_claims": [],
    }
    return seal(board)


def seal(board: dict) -> dict:
    """Force the fail-closed flags. A receipt cannot override this function."""
    board["arithmetic"] = "NON-CERTIFYING"
    board["lemma_closed"] = False
    board["prizes_solved"] = False
    board["original_prize_closed"] = False
    board["independence_credit"] = 0
    board["authority"] = "NONE"
    board["drive_is_source_of_truth"] = True
    board["git_role"] = "execution/workspace mirror only"
    board["bridge_disposition"] = "PROPOSED EXECUTION CONTRACT / NOT DEPLOYED"
    j = board["OBL_H5_JETMOD"]
    j["status"] = "OPEN"
    j["lemma_closed"] = False
    j["discharges_OBL"] = False
    r = board["D3_LEMMA_RN_UNIF"]
    r["status"] = "OPEN"
    r["lemma_closed"] = False
    r["U_certified"] = False
    r["discharges_lemma"] = False
    m = board["mesh_plan_toy"]
    m["certified"] = False
    board["historical_snapshot"] = HISTORICAL_SNAPSHOT.name
    board["historical_snapshot_rewritten"] = False
    board["non_claims"] = [
        "This console never discharges OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
        "Toy mesh plan is not a certificate. Arithmetic is NON-CERTIFYING (float).",
        "A provisional U or an empirical envelope is not a FREEZE and is not a certified enclosure.",
        "RUNG2 and RUNG3 do not discharge JETMOD. A green CI run does not discharge an obligation.",
        "prizes_solved is false. independence_credit is 0. No original prize problem is solved.",
        "Drive remains the source of truth. This git packet is an execution/workspace mirror only.",
        "The execution bridge stays PROPOSED EXECUTION CONTRACT / NOT DEPLOYED.",
    ]
    return board


def render(board: dict) -> str:
    j = board["OBL_H5_JETMOD"]
    r = board["D3_LEMMA_RN_UNIF"]
    m = board["mesh_plan_toy"]
    n = board["next_action"]
    lines = []
    lines.append("Math console (fail-closed, OPEN/HOLD)")
    lines.append(f"generated {board['generated_at_utc']}")
    lines.append(f"arithmetic {board['arithmetic']}")
    lines.append("")
    lines.append(f"OBL-H5-JETMOD …… {j['status']}")
    lines.append(
        f"  jets named {j['jets_named']}/24 · max width {_fmt(j['max_enclosure_width'])}"
        f" · struct hw {_fmt(j['struct_model_halfwidth'])}"
    )
    lines.append(f"  exceeds struct? {j['width_exceeds_struct']} · discharges? False")
    if j["blocked"]:
        lines.append(f"  blocked: {j['blocked'][0]}")
    lines.append("")
    lines.append(f"D3-LEMMA-RN-UNIF … {r['status']}")
    lines.append(f"  |∇κ|_white center {_fmt(r['abs_grad_kap_pair_white_center'])}")
    lines.append(f"  U/envelope {_fmt(r['U_or_envelope'])} ({r['U_source']}; certified={r['U_certified']})")
    if r["empirical_envelope_upper"] is not None:
        lines.append(f"  empirical envelope {_fmt(r['empirical_envelope_upper'])}")
    if r["C_H_prior_provisional"] is not None:
        lines.append(
            f"  C_H prior {_fmt(r['C_H_prior_provisional'])}"
            f" · measured ‖H‖/RSS {_fmt(r['measured_hess_op_over_rss'])}"
            + (f" · loose ×{_fmt(r['C_H_loose_factor_vs_center'])}" if r["C_H_loose_factor_vs_center"] else "")
        )
    lines.append(f"  Piece-2 annulus driver: {r['piece2_annulus_driver']}")
    lines.append(f"  lemma_closed={r['lemma_closed']} discharges_lemma={r['discharges_lemma']}")
    lines.append("")
    lines.append("Mesh plan (toy, NON-CERTIFYING, not a certificate)")
    if m["hw_max_toy"] is not None:
        lines.append(
            f"  hw_max ≲ {m['hw_max_toy']:.4g} · Δd_max(θ→0) ≲ {m['dd_max_if_theta_zero']:.4g}"
            f" · via {m['source']}"
        )
        if m["dd_max_if_theta_zero"] < 0.5:
            lines.append("  engine first ring Δd=0.5 is coarser than this toy radial step.")
    else:
        lines.append(f"  {m['note']}")
    lines.append("")
    lines.append("Next action")
    lines.append(f"  {n['id']}: {n['why']}")
    lines.append(f"  $ {n['command']}")
    lines.append(f"  do not: {n['do_not']}")
    lines.append("")
    lines.append(
        f"Flags: lemma_closed={board['lemma_closed']} prizes_solved={board['prizes_solved']} "
        f"independence_credit={board['independence_credit']}"
    )
    lines.append("Non-claims: " + board["non_claims"][0])
    if board.get("receipt_contradictions"):
        lines.append("Receipt contradictions (ignored; board stays OPEN):")
        for item in board["receipt_contradictions"]:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    try:
        v = float(x)
        if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.4g}"
        return f"{v:.6g}"
    except Exception:
        return str(x)[:80]


def fail_closed_ok(board: dict) -> bool:
    j = board.get("OBL_H5_JETMOD") or {}
    r = board.get("D3_LEMMA_RN_UNIF") or {}
    m = board.get("mesh_plan_toy") or {}
    return (
        board.get("lemma_closed") is False
        and board.get("prizes_solved") is False
        and board.get("original_prize_closed") is False
        and board.get("independence_credit") == 0
        and type(board.get("independence_credit")) is int
        and board.get("arithmetic") == "NON-CERTIFYING"
        and j.get("status") == "OPEN"
        and j.get("lemma_closed") is False
        and j.get("discharges_OBL") is False
        and r.get("status") == "OPEN"
        and r.get("lemma_closed") is False
        and r.get("discharges_lemma") is False
        and r.get("U_certified") is False
        and m.get("certified") is False
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print JSON board only")
    ap.add_argument("--budget", type=float, default=0.68)
    ap.add_argument("--margin", type=float, default=2e-4)
    args = ap.parse_args(argv)
    board = build(args.budget, args.margin)
    if not fail_closed_ok(board):
        print("CK-FAIL: console tried to claim discharge or a closed lemma")
        return 1
    if args.json:
        print(json.dumps(board, indent=2))
    else:
        print(render(board))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
