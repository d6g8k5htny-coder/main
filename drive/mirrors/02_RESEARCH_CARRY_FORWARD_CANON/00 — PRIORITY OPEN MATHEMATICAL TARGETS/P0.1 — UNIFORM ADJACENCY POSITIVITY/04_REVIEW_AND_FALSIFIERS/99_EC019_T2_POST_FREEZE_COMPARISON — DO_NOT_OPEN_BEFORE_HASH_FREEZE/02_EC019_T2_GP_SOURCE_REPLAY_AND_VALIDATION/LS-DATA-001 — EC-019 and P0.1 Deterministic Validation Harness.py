#!/usr/bin/env python3
"""Lead-scientist deterministic checks for EC-019 and P0.1 Domains G/D/T.

This script does not prove Gaussian-process theorems. It checks the exact
finite algebra, content identity, source replay identity, determinant scaling,
and the counterexample/repair trigger used by the accompanying audits.
"""
from pathlib import Path
import hashlib
import json
import sympy as sp
import subprocess
import sys

ROOT = Path("/mnt/data")
FROZEN = ROOT / "GP-DATA-123-v1.0 — Frozen Content-Addressed Snapshot of GP-DER-118-v1.3.txt"
EC_SOURCE = ROOT / "gp_data_168_v1_1.py"
EC_RESULT = ROOT / "gp_data_168_v1_1_result.json"
EC_REPLAY = ROOT / "gp_data_168_v1_1_replay_audit.json"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frozen_body_identity():
    text = FROZEN.read_bytes().decode("utf-8-sig").replace("\r\n","\n").replace("\r","\n")
    sm = "BEGIN_FROZEN_THEOREM_BODY\n"
    em = "\nEND_FROZEN_THEOREM_BODY"
    i = text.index(sm) + len(sm)
    j = text.index(em, i)
    body = (text[i:j].rstrip("\n") + "\n").encode("utf-8")
    return len(body), hashlib.sha256(body).hexdigest()

def ec019_replay():
    with EC_REPLAY.open("wb") as out:
        cp = subprocess.run([sys.executable, str(EC_SOURCE)], stdout=out, stderr=subprocess.PIPE)
    return {
        "returncode": cp.returncode,
        "stderr": cp.stderr.decode("utf-8", errors="replace"),
        "source_bytes": EC_SOURCE.stat().st_size,
        "source_sha256": sha256(EC_SOURCE),
        "stored_result_bytes": EC_RESULT.stat().st_size,
        "stored_result_sha256": sha256(EC_RESULT),
        "replay_result_bytes": EC_REPLAY.stat().st_size,
        "replay_result_sha256": sha256(EC_REPLAY),
        "byte_identical": EC_REPLAY.read_bytes() == EC_RESULT.read_bytes(),
    }

def symbolic_checks():
    h,b=sp.symbols("h b", positive=True)
    fminus=b
    fplus=b-(2*h)**3/sp.Integer(6)
    L1=sp.simplify((fplus-fminus)/(2*h))
    L5=sp.simplify(sp.Rational(3,1)/h**2*(0-L1))

    x=sp.symbols("x")
    g0,g1,g2,g3,g4,g5=sp.symbols("g0 g1 g2 g3 g4 g5")
    g=g0+g1*x+g2*x**2/sp.Integer(2)+g3*x**3/sp.Integer(6)+g4*x**4/sp.Integer(24)+g5*x**5/sp.Integer(120)
    sol=sp.solve([sp.expand(g.subs(x,h)),sp.expand(g.subs(x,-h))],[g0,g1], dict=True)[0]
    gp=sp.diff(g,x).subs(sol)
    AM=sp.simplify(gp.subs(x,-h)/(2*h))
    AS=sp.simplify(gp.subs(x,h)/(2*h))

    H11,H12,H22,E11,E12,E22,r=sp.symbols("H11 H12 H22 E11 E12 E22 r")
    H=sp.Matrix([[H11,H12],[H12,H22]])
    E=sp.Matrix([[E11,E12],[E12,E22]])
    det_residual=sp.expand((H+r*E).det()-H.det()-r*(H22*E11+H11*E22-2*H12*E12)-r**2*E.det())

    # Counterexample to the invalid inequality sup|Z| <= sup Z + sup(-Z):
    singleton_value=-2
    lhs=abs(singleton_value)
    rhs=singleton_value+(-singleton_value)

    return {
        "pin_L1": str(L1),
        "pin_L5": str(L5),
        "AM_series": str(sp.series(AM,h,0,4)),
        "AS_series": str(sp.series(AS,h,0,4)),
        "AM_limit_if_g2_equals_2": str(sp.limit(AM.subs(g2,2),h,0)),
        "AS_limit_if_g2_equals_2": str(sp.limit(AS.subs(g2,2),h,0)),
        "determinant_perturbation_residual": str(det_residual),
        "invalid_sup_inequality_counterexample": {
            "index_set":"singleton",
            "Z":singleton_value,
            "sup_abs_Z":lhs,
            "sup_Z_plus_sup_minus_Z":rhs,
            "inequality_holds": lhs <= rhs,
        },
        "repair":"Apply Dudley directly to Y(sign,x)=sign*Z(x) on {+1,-1}xK; its supremum is sup|Z|.",
    }

def main():
    body_bytes, body_sha = frozen_body_identity()
    report = {
        "frozen_theorem": {
            "body_bytes": body_bytes,
            "body_sha256": body_sha,
            "expected_bytes":14591,
            "expected_sha256":"666f582cb1374e3191b04053d2f7b75b129006fc83f3c6bbb237055721c4eb28",
            "identity_pass": body_bytes==14591 and body_sha=="666f582cb1374e3191b04053d2f7b75b129006fc83f3c6bbb237055721c4eb28",
        },
        "ec019":ec019_replay(),
        "symbolic":symbolic_checks(),
    }
    report["overall_deterministic_checks_pass"] = (
        report["frozen_theorem"]["identity_pass"]
        and report["ec019"]["returncode"]==0
        and report["ec019"]["byte_identical"]
        and report["symbolic"]["pin_L5"]=="2"
        and report["symbolic"]["AM_limit_if_g2_equals_2"]=="-1"
        and report["symbolic"]["AS_limit_if_g2_equals_2"]=="1"
        and report["symbolic"]["determinant_perturbation_residual"]=="0"
        and report["symbolic"]["invalid_sup_inequality_counterexample"]["inequality_holds"] is False
    )
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__=="__main__":
    main()
