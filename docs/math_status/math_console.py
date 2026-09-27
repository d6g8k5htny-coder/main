#!/usr/bin/env python3
"""math_console.py — one-command fail-closed math board for Dylan.

Reads local JETMOD / RN-UNIF receipts under docs/math_status/ (prototypes are not mirrored here); prints an honest
OPEN/blocked status board, recommends mesh size from the whitened envelope,
and names the single next action. Never flips lemma_closed / never discharges
OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.

Usage:
  python3 docs/math_status/math_console.py
  python3 docs/math_status/math_console.py --json
  python3 docs/math_status/math_console.py --budget 0.68 --margin 2e-4
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]  # repository root
PROTO = Path(__file__).resolve().parent
OUT = PROTO / "math_console_snapshot.json"

# Prefer newer aliases; fall back to older names from the same push.
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
        "rnu_white_grad_box_envelope_receipt.json",
    ],
    "rnu_hess": [
        "rnu_white_hess_calibrate_v1_receipt.json",
        "rnu_white_hess_calibrate_v1_receipt.json",
    ],
    "piece2": [
        "piece2_first_cell_failclosed_receipt.json",
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


def mesh_plan(envelope: Optional[float], kap: Optional[float], budget: float, margin: float, source: str, certified: bool) -> MeshPlan:
    if envelope is None or kap is None or envelope <= 0:
        return MeshPlan(budget, margin, kap, envelope, source, None, None,
                        "Need kap_far + |∇κ| envelope to size the first polar ring.", certified)
    room = budget - margin - kap
    if room <= 0:
        return MeshPlan(budget, margin, kap, envelope, source, None, None,
                        f"No room under budget: kap={kap:.6g} already uses ≥ budget−margin.", certified)
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
            "Engine Δd=0.5 is too coarse if dd_max_if_theta_zero < 0.5. "
            + ("Envelope is CERTIFIED-ish box majorant path." if certified else "Envelope is EMPIRICAL / provisional — not a certificate.")
        ),
        certified=certified,
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

    # --- JETMOD ---
    jets_named = None
    max_w = None
    struct_hw = None
    jetmod_obl = "OPEN"
    jetmod_blocked = []
    if jet_ext:
        jets_named = jet_ext.get("jets_named_total_now") or jet_ext.get("jets_after")
        max_w = _f(jet_ext.get("combined_max_enclosure_width"))
        struct_hw = _f(jet_ext.get("struct_model_halfwidth") or jet_ext.get("struct_model_half_width"))
        if jet_ext.get("discharges_OBL_H5_JETMOD") is True:
            jetmod_obl = "CLAIMED_DISCHARGE_IN_RECEIPT — TREAT AS ERROR"
        stop = jet_ext.get("stop_reason") or jet_ext.get("halt_reason")
        if stop:
            jetmod_blocked.append(str(stop)[:240])
    if jet_multi and jets_named is None:
        jets_named = jet_multi.get("jets_after") or jet_multi.get("jets_done")
        max_w = _f(jet_multi.get("max_enclosure_width"))
        struct_hw = _f(jet_multi.get("struct_model_halfwidth"))
    jetmod_blocked.append("24-jet roster + p_J unenumerated in H5 sources")
    jetmod_blocked.append("PinFrame DER stops at order 2; no F(G12-band) cover runner")

    # --- RN-UNIF ---
    lemma = "OPEN"
    abs_grad = None
    if ex:
        abs_grad = _f(ex.get("abs_grad_kap_pair_from_white") or ex.get("abs_grad_kap_pair"))
    U = None
    U_certified = False
    U_source = "missing"
    kap = None
    if white:
        if white.get("D3_LEMMA_RN_UNIF"):
            lemma = str(white["D3_LEMMA_RN_UNIF"])
        if white.get("lemma_closed") is True:
            lemma = "CLAIMED_CLOSED_IN_RECEIPT — TREAT AS ERROR"
        U = _f(_dig(white, "provisional_box_majorant_grad_kap_pair", "U_abs_grad_kap_pair"))
        if U is not None:
            U_source = "white_box_provisional_U"
            U_certified = False
        kap = _f(_dig(white, "box", "kap_far_center")) or _f(_dig(white, "provisional_box_majorant_grad_kap_pair", "kap_far_center"))
        pt = _dig(white, "certified", "pointwise_exact_grad_kap_pair_at_center") or {}
        if isinstance(pt, dict) and abs_grad is None:
            abs_grad = _f(pt.get("abs_grad_kap_pair") or pt.get("abs_grad"))
        # some receipts nest kap under certified center
        if kap is None:
            kap = _f(pt.get("kap_far")) if isinstance(pt, dict) else None
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
            U_certified = False

    # Prefer tighter empirical envelope for mesh planning when both exist
    plan_U = env_upper if env_upper is not None else U
    plan_src = "empirical_envelope" if env_upper is not None else U_source
    plan_cert = False if env_upper is not None else U_certified

    # Hess calibration
    c_h_prior = None
    ratio_op = None
    if hess:
        c_h_prior = _f(hess.get("C_H_prior_provisional") or hess.get("C_H_prior"))
        ratio_op = _f(hess.get("ratio_hess_op_over_rss") or hess.get("ratio_hess_op_over_rss"))

    piece2 = "UNKNOWN"
    if p2:
        piece2 = str(p2.get("piece2_driver_status") or p2.get("piece2_annulus_driver") or "UNWRITTEN")
        if p2.get("lemma_closed") is True:
            lemma = "CLAIMED_CLOSED_IN_RECEIPT — TREAT AS ERROR"

    plan = mesh_plan(plan_U, kap, budget, margin, plan_src, plan_cert)

    # Next action (single)
    next_action = {
        "id": "rnu_tighten_CH",
        "why": "Whitening killed 1e17 slack; C_H still provisional (~20 vs measured ratio ~0.012). Tight certified Lip-of-Hess is the RN bottleneck.",
        "command": "python3 docs/math_status/math_console.py --json  # hess calibrator not mirrored; stay fail-closed",
        "do_not": "Do not flip lemma_closed; do not invent a 24-jet roster; do not treat CI green as discharge.",
    }
    if piece2.upper().startswith("UNWRITTEN") and plan.hw_max_toy and plan.hw_max_toy < 0.2:
        # still RN first — piece2 after bound
        pass
    if jets_named is not None and jets_named >= 8:
        # JETMOD is roster-blocked; don't recommend inventing names
        pass
    if white is None and env is None:
        next_action = {
            "id": "rnu_white_box",
            "why": "No whitened box / envelope receipt found.",
            "command": "python3 docs/math_status/math_console.py --json  # prototype receipts not mirrored; stay fail-closed",
            "do_not": "Do not use crude chi2_grad_bound as if whitened.",
        }

    board = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "discipline": "fail-closed; display ≠ certified enclosure; Drive SoT",
        "paths": {k: (str(v[0]) if v[0] else None) for k, v in loaded.items()},
        "OBL_H5_JETMOD": {
            "status": jetmod_obl,
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
            "status": lemma if lemma != "OPEN" and "CLAIMED" not in lemma else "OPEN",
            "lemma_closed": False,
            "abs_grad_kap_pair_white_center": abs_grad,
            "U_or_envelope": U,
            "U_source": U_source,
            "U_certified": U_certified,
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
        "non_claims": [
            "This console never discharges OBL-H5-JETMOD or D3-LEMMA-RN-UNIF.",
            "Toy mesh plan is not a certificate.",
            "Provisional U / empirical envelope ≠ FREEZE.",
        ],
    }
    return board


def render(board: dict) -> str:
    j = board["OBL_H5_JETMOD"]
    r = board["D3_LEMMA_RN_UNIF"]
    m = board["mesh_plan_toy"]
    n = board["next_action"]
    lines = []
    lines.append("══ Math console (fail-closed) ══")
    lines.append(f"generated {board['generated_at_utc']}")
    lines.append("")
    lines.append(f"OBL-H5-JETMOD …… {j['status']}")
    lines.append(f"  jets named {j['jets_named']}/24 · max width { _fmt(j['max_enclosure_width']) } · struct hw { _fmt(j['struct_model_halfwidth']) }")
    lines.append(f"  exceeds struct? {j['width_exceeds_struct']} · discharges? False")
    if j["blocked"]:
        lines.append(f"  blocked: {j['blocked'][0]}")
    lines.append("")
    lines.append(f"D3-LEMMA-RN-UNIF … {r['status']}")
    lines.append(f"  |∇κ|_white center { _fmt(r['abs_grad_kap_pair_white_center']) }")
    lines.append(f"  U/envelope { _fmt(r['U_or_envelope']) } ({r['U_source']}; certified={r['U_certified']})")
    if r["empirical_envelope_upper"] is not None:
        lines.append(f"  empirical envelope { _fmt(r['empirical_envelope_upper']) }")
    if r["C_H_prior_provisional"] is not None:
        lines.append(
            f"  C_H prior { _fmt(r['C_H_prior_provisional']) } · measured ‖H‖/RSS { _fmt(r['measured_hess_op_over_rss']) }"
            + (f" · loose ×{ _fmt(r['C_H_loose_factor_vs_center']) }" if r['C_H_loose_factor_vs_center'] else "")
        )
    lines.append(f"  Piece-2 annulus driver: {r['piece2_annulus_driver']}")
    lines.append("")
    lines.append("Mesh plan (toy, not a certificate)")
    if m["hw_max_toy"] is not None:
        lines.append(
            f"  hw_max ≲ {m['hw_max_toy']:.4g} · Δd_max(θ→0) ≲ {m['dd_max_if_theta_zero']:.4g}"
            f" · via {m['source']}"
        )
        if m["dd_max_if_theta_zero"] < 0.5:
            lines.append("  ⇒ engine first ring Δd=0.5 is too coarse; shrink radial step first.")
    else:
        lines.append(f"  {m['note']}")
    lines.append("")
    lines.append("Next action")
    lines.append(f"  {n['id']}: {n['why']}")
    lines.append(f"  $ {n['command']}")
    lines.append(f"  do not: {n['do_not']}")
    lines.append("")
    lines.append("Non-claims: " + board["non_claims"][0])
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print JSON snapshot only")
    ap.add_argument("--budget", type=float, default=0.68)
    ap.add_argument("--margin", type=float, default=2e-4)
    args = ap.parse_args()
    board = build(args.budget, args.margin)
    OUT.write_text(json.dumps(board, indent=2) + "\n")
    if args.json:
        print(json.dumps(board, indent=2))
    else:
        print(render(board))
        print(f"\n[wrote {OUT}]")
    # Fail-closed integrity: never exit claiming discharge
    if board["OBL_H5_JETMOD"].get("discharges_OBL") or board["D3_LEMMA_RN_UNIF"].get("discharges_lemma"):
        print("CK-FAIL: console tried to claim discharge")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
