#!/usr/bin/env python3
"""
verifier/v8/verify.py -- V5 criteria (Palm-transfer round).

Extends v7: re-executes all TWELVE scripts (nine v2/V3/V4 + three new Part-G
scripts) in BOTH normal and `python3 -O` modes (24 runs; exit 0 +
ALL_ASSERTIONS_PASS required), AST-scans for bare asserts, checks the
Part-G document content (Palm-transfer closure theorems, RP-F audit
verdict, and the honestly retained OPEN status of RP-C/RP-S), and repeats
the pinned-determinant symbolic spot check of v7.

Checks:
  1-12  normal-mode re-execution of each script
 13-24  python3 -O re-execution of each script
 25     no bare `assert` anywhere (ast scan)
 26     ck() helper present in every script
 27     Part G present with Theorem G.7.1 (RP-A/RP-L closed) and the RP-F
        audit verdict
 28     Part G.9 retains RP-C/RP-S as OPEN with the localized residual
        step; candidate theorem still HOLD
 29     the two numeric-corroboration scripts print their NUMERIC scope
        limits in their transcripts
 30     symbolic: (2.3) pin equivalence and pinned -det H_M (as v7)
Prints ALL_ASSERTIONS_PASS and exits 0 iff all checks pass.
"""
import ast, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(ROOT, "scripts")

NAMES = [
    "arithmetic_ledgers.py",
    "audit_density_vs_cumulative.py",
    "contact_covariance_verification.py",
    "d2_closed_form.py",
    "first_variation_derivation.py",
    "goe_cone_coefficient.py",
    "triple_contact_elimination.py",
    "verify_capture_escape_budgets.py",
    "verify_constrained_triple_contact.py",
    "verify_palm_soft_factors.py",
    "verify_corrected_pin_floor.py",
    "verify_collar_factorization.py",
]

results = []

def report(label, ok, detail=""):
    results.append((label, ok))
    print(f"[{'ok' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))

def run_script(name, opt):
    cmd = [sys.executable] + ([opt] if opt else []) + [os.path.join(SCRIPTS, name)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and "ALL_ASSERTIONS_PASS" in r.stdout, r

for i, name in enumerate(NAMES):
    ok, r = run_script(name, None)
    report(f"{i+1:2d}. normal run {name}", ok, (r.stderr.strip().splitlines() or [""])[-1][:120])
for i, name in enumerate(NAMES):
    ok, r = run_script(name, "-O")
    report(f"{i+13:2d}. python -O run {name}", ok, (r.stderr.strip().splitlines() or [""])[-1][:120])

bare = []
for name in NAMES:
    tree = ast.parse(open(os.path.join(SCRIPTS, name)).read())
    bare += [(name, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Assert)]
report("25. no bare `assert` statement in any script (ast scan)", not bare, str(bare[:5]))

missing = [n for n in NAMES if "def ck(" not in open(os.path.join(SCRIPTS, n)).read()]
report("26. ck() raising check helper present in every script", not missing, str(missing))

supp = open(os.path.join(ROOT, "SIDE24_GAP_FILL_SUPPLEMENT.md")).read()
report("27. Part G: Theorem G.7.1 (RP-A/RP-L closed) and RP-F audit verdict present",
       "Theorem G.7.1" in supp and "RP-F is CLOSED" in supp and "Audit verdict" in supp)
report("28. Part G.9: RP-C/RP-S retained OPEN with localized residual step; theorem HOLD",
       "residual step (OPEN, now sharply localized)" in supp
       and "HOLD" in supp and "conditional on exactly two premises" in supp)

t1 = open(os.path.join(SCRIPTS, "verify_corrected_pin_floor.out.txt")).read()
t2 = open(os.path.join(SCRIPTS, "verify_collar_factorization.out.txt")).read()
report("29. numeric-corroboration transcripts carry NUMERIC scope limits",
       "NUMERIC" in t1 and "SCOPE LIMIT" in t1 and "NUMERIC" in t2 and "SCOPE LIMIT" in t2)

try:
    import sympy as sp
    X, Y, kap, lam, m, d, gam, c, s, eta, alpha = sp.symbols(
        'X Y kap lam m d gam c s eta alpha', nonzero=True)
    p = (kap*(X**3/3 - X/4 - sp.Rational(1, 12)) + lam*(X**2 - sp.Rational(1, 4))*Y
         + (m/2 - d*X)*Y**2/sp.Integer(2) + gam*Y**3/sp.Integer(6))
    lam_sol = (-kap*(4*c**2 - eta**2) + 2*d*s**2)/(8*c*s)
    Pc = (c/eta, s/eta); Mp = (-sp.Rational(1, 2), 0)
    ps = p.subs(lam, lam_sol)
    gam_sub = sp.solve(sp.Eq(sp.diff(ps, Y).subs({X: Pc[0], Y: Pc[1]}), 0), gam)[0]
    ps = ps.subs(gam, gam_sub)
    pP = sp.simplify(ps.subs({X: Pc[0], Y: Pc[1]}))
    R_al = (2*c + eta)**2 - 8*alpha*c*eta
    lhs = 2*s**2*(2*c*m - eta*d); rhs = kap*eta*R_al
    diff2 = sp.simplify((pP + alpha*kap/6)*48*c*eta**3 - eta*(lhs - rhs))
    u = (2*s**2*d + kap*(2*c + eta)**2)/(8*kap*c*eta)
    H = sp.Matrix([[sp.diff(ps, X, 2), sp.diff(ps, X, Y)],
                   [sp.diff(ps, X, Y), sp.diff(ps, Y, 2)]])
    m_sub = sp.solve(sp.Eq(lhs, rhs), m)[0]
    d_sub = (8*kap*c*eta*u - kap*(2*c + eta)**2)/(2*s**2)
    ndM = sp.simplify(-H.det().subs({X: Mp[0], Y: Mp[1]}).subs(m, m_sub).subs(d, d_sub))
    target = kap**2*eta**2/s**2*(u**2 - alpha)
    ok30 = diff2 == 0 and sp.simplify(ndM - target) == 0
    report("30. symbolic: (2.3) pin equivalence and pinned -det H_M", ok30)
except Exception as e:
    report("30. symbolic spot check", False, repr(e)[:120])

n_ok = sum(1 for _, ok in results if ok)
print(f"v8: {n_ok}/{len(results)} checks passed")
if n_ok == len(results):
    print("ALL_ASSERTIONS_PASS")
    sys.exit(0)
sys.exit(1)
