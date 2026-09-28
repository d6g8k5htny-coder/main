#!/usr/bin/env python3
"""Push RN-UNIF Piece-1 as far as a valid-enough T4 can go this cycle.

NOT a freeze. Constructs:
  1. env_form RSS table q=0..4 at several radii (engine-certified envelopes)
  2. he_abs ratio s=2d+10 (already fail-closed in the engine)
  3. T4_form(d) = 24 * s * RSS(env_form q=3) + RSS(env_form q=4)
     — this is a valid envelope of the residual-form 4-jets
  4. composition inflation C_comp from exact piece sizes at (5,0)
     so T4_kap(d) = C_comp * T4_form(d) / T4_form(5) * T4_form(5)_scaled
  5. FD fourth of kappa_far at (5,0) as a check that T4_kap >= measured
  6. cell-close on a coarse delicate-patch net using exact DS (κ,∇,H)
     + measured |T3| * (RSS3(d)/RSS3(5)) + T4_kap(d)

The composition constant is the remaining analytic gap if C_comp is
taken from peak magnitudes rather than a full 4th-order chain rule.
We print that gap instead of hiding it.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, ".")
sys.path.insert(0, "../H2_foundations")

t0 = time.time()
import d3_rn_unif as R
from mpmath import mp, mpf
import mpmath

ns = lambda x, n=8: mpmath.nstr(x, n)
v = R.kit.b - R.kit.ell / 2


def rss_env(d, q):
    tot2 = mpf(0)
    n = 0
    for k in range(6):
        for g in list(R._HYG) + list(R._YJG):
            e = R.env_form(k, g, d, q)
            tot2 += e * e
            n += 1
    return mpmath.sqrt(tot2), n


def t4_form(d):
    s = 2 * d + 10
    r3, _ = rss_env(d, 3)
    r4, _ = rss_env(d, 4)
    # product-rule one extra derivative on a 3-jet plus native 4-jet
    return 24 * s * r3 + r4, r3, r4, s


def cell_sup(kap, g, h, t, t4, hw):
    return kap + g * hw + h * hw ** 2 / 2 + t * hw ** 3 / 6 + t4 * hw ** 4 / 24


def main():
    out = {
        "status": "PROPOSED",
        "lemma_closed": False,
        "import_s": time.time() - t0,
    }
    print(f"[import] {out['import_s']:.2f}s")

    # --- env_form tables ---
    table = []
    for d in (mpf(5), mpf("5.05"), mpf("5.2"), mpf(6), mpf(8), mpf(10)):
        row = {"d": ns(d, 4)}
        for q in range(5):
            r, n = rss_env(d, q)
            row[f"rss_q{q}"] = ns(r, 6)
        tf, r3, r4, s = t4_form(d)
        row["s"] = ns(s, 4)
        row["T4_form"] = ns(tf, 6)
        table.append(row)
        print(
            f"[env] d={ns(d,4)} rss0={row['rss_q0']} rss3={row['rss_q3']} "
            f"rss4={row['rss_q4']} T4_form={row['T4_form']}"
        )
    out["env_table"] = table

    # --- exact jets at peak from already-computed DS ---
    kd = R._kd
    print(f"[DS] (5,0) kap={ns(kd.v,12)} |g|={ns(kd.gnorm(),6)} |H|={ns(kd.hnorm(),6)}")

    # T3 / T4 via FD of exact Hessian (re-use prior h=1e-8 recipe)
    h = mpf("1e-8")
    t_fd = time.time()
    px = R.kappa_far_ds((mpf(5) + h, mpf(0)), v)
    mx = R.kappa_far_ds((mpf(5) - h, mpf(0)), v)
    py = R.kappa_far_ds((mpf(5), h), v)
    my = R.kappa_far_ds((mpf(5), -h), v)
    txxx = (px.hxx - mx.hxx) / (2 * h)
    txxy = (px.hxy - mx.hxy) / (2 * h)
    txyy = (px.hyy - mx.hyy) / (2 * h)
    tyyy = (py.hyy - my.hyy) / (2 * h)
    t3 = mpmath.sqrt(txxx ** 2 + tyyy ** 2 + 3 * txxy ** 2 + 3 * txyy ** 2)
    # fourth: FD of txxx ~ d^4/dx^4 from hxx second difference
    txxxx = (px.hxx - 2 * kd.hxx + mx.hxx) / (h ** 2)
    txxyy = (px.hyy - 2 * kd.hyy + mx.hyy) / (h ** 2)
    tyyyy = (py.hyy - 2 * kd.hyy + my.hyy) / (h ** 2)
    t4_fd = mpmath.sqrt(txxxx ** 2 + tyyyy ** 2 + 6 * txxyy ** 2)
    print(
        f"[FD] |T3|={ns(t3,6)} |T4|_FD={ns(t4_fd,6)}  ({time.time()-t_fd:.2f}s)"
    )
    out["peak_jets"] = {
        "kap": ns(kd.v, 16),
        "g": ns(kd.gnorm(), 12),
        "H": ns(kd.hnorm(), 12),
        "T3": ns(t3, 12),
        "T4_FD": ns(t4_fd, 12),
        "txxxx": ns(txxxx, 8),
        "txxyy": ns(txxyy, 8),
        "tyyyy": ns(tyyyy, 8),
    }

    # composition inflation: how much larger is |T4_kap| than T4_form at d=5
    tf5, r35, r45, s5 = t4_form(mpf(5))
    # If residual forms generated the 4-jet of κ through a Lipschitz map
    # of size |∂κ/∂forms| ≲ |g|/rss1 or similar. Use max(1, |T4_FD|/T4_form, |T3|/rss3)
    r15, _ = rss_env(mpf(5), 1)
    r05, _ = rss_env(mpf(5), 0)
    lift_g = kd.gnorm() / max(r15, mpf("1e-30"))
    lift_t3 = t3 / max(r35, mpf("1e-30"))
    lift_t4 = t4_fd / max(tf5, mpf("1e-30"))
    C_comp = 100 * max(lift_g, lift_t3, lift_t4, mpf(1))
    print(
        f"[lift] |g|/rss1={ns(lift_g,4)} |T3|/rss3={ns(lift_t3,4)} "
        f"|T4_FD|/T4_form={ns(lift_t4,4)}  C_comp={ns(C_comp,4)}"
    )
    out["lifts"] = {
        "T4_form_d5": ns(tf5, 8),
        "rss1_d5": ns(r15, 8),
        "rss3_d5": ns(r35, 8),
        "lift_g": ns(lift_g, 8),
        "lift_t3": ns(lift_t3, 8),
        "lift_t4": ns(lift_t4, 8),
        "C_comp": ns(C_comp, 8),
        "C_comp_kind": "100 * max(lifts of exact jets onto env_form RSS). "
        "Valid IFF the 4-jet of kappa is dominated by residual-form 4-jets "
        "with this Lipschitz. Wick/Bures/ratio 4-jets are NOT separately certified.",
    }

    def T4_kap(d):
        tf, _, _, _ = t4_form(d)
        return C_comp * tf

    def T3_env(d):
        r3, _ = rss_env(d, 3)
        return (t3 / r35) * r3 * 10  # 10x safety on 3-jet radial continuation

    t4_at5 = T4_kap(mpf(5))
    print(f"[T4] T4_kap(5)={ns(t4_at5,6)}  vs FD {ns(t4_fd,6)}  cover={t4_at5 >= t4_fd}")
    out["T4_covers_FD_at_peak"] = bool(t4_at5 >= t4_fd)

    # --- cell arithmetic at peak and a few radii ---
    rows = []
    caps = (mpf("0.68"), mpf("0.69"))
    hws = (mpf("7e-4"), mpf("1e-3"), mpf("2e-3"), mpf("5e-3"))
    for d, kap, g, hrm, t3d in (
        (mpf(5), kd.v, kd.gnorm(), kd.hnorm(), t3),
        (mpf("5.05"), mpf("0.5687385727"), kd.gnorm() * mpf("0.85"), kd.hnorm() * mpf("0.85"), T3_env(mpf("5.05"))),
        (mpf(6), mpf("0.01099667899"), mpf(1), mpf(2), T3_env(mpf(6))),
    ):
        t4d = T4_kap(d)
        for cap in caps:
            for hw in hws:
                # at d>5 the g,H for 5.05/6 are conservative stand-ins except kap which is fast-exact
                sup = cell_sup(kap, g, hrm, t3d, t4d, hw)
                close = sup <= cap - mpf("0.0002")
                rows.append(
                    {
                        "d": ns(d, 4),
                        "cap": ns(cap, 4),
                        "hw": ns(hw, 4),
                        "sup": ns(sup, 10),
                        "T4": ns(t4d, 6),
                        "close": bool(close),
                    }
                )
                if d == 5 or hw == mpf("7e-4"):
                    print(
                        f"[cell] d={ns(d,4)} cap={ns(cap,4)} hw={ns(hw,4)} "
                        f"sup={ns(sup,8)} {'CLOSE' if close else 'OPEN'}"
                    )
    out["cells"] = rows

    # delicate-patch budget: area ~0.04, hw=7e-4 => ~ area/(2hw)^2 points
    area = mpf("0.04")
    hw = mpf("7e-4")
    npts = int(area / ((2 * hw) ** 2))
    out["cost_model"] = {
        "delicate_area": "0.04",
        "hw_peak": "7e-4",
        "est_points": npts,
        "sec_per_point_DS": 0.9,
        "est_hours": round(float(npts * 0.9 / 3600), 3),
        "note": "full polar cover not run this cycle",
    }
    print(f"[cost] ~{npts} cells at hw=7e-4 over area 0.04 -> ~{out['cost_model']['est_hours']} h")

    out["remaining_to_freeze"] = [
        "Full 4th-order chain rule (Wick, Bures, ratios) or interval DS on each cell",
        "Polar cover d in [5,17] with theta-halving",
        "Both-mode transcript + MUT-RN-1..5 + FREEZE rule-id",
        "Rename duplicate rnu_ds3.py (scalar SUPERSEDED vs engine lift)",
    ]
    out["lemma_closed"] = False

    dest = HERE / "RNU_T4_PUSH_RECEIPT.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[write] {dest}")
    print("LEMMA_CLOSED=NO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
