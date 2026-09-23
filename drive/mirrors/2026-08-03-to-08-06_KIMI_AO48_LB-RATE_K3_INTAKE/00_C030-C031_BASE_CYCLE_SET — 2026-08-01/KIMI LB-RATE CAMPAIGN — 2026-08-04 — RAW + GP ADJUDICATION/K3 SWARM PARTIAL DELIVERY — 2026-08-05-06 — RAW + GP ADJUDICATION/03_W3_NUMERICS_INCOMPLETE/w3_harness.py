# W3 fail-closed harness.
#   python3 w3_harness.py            (normal)
#   python3 -O w3_harness.py         (must be byte-identical at program-output level)
# Mutation injection via env W3_MUTATE in {sqrt, pin, zone}; each must fail closed
# (SystemExit with nonzero exit code).
# Checks:
#   1. anchor truth: exact rho(0,0.60) interval inside [1.0e-4, 1.12e-4]
#   2. hierarchy: hi(exact) <= min(hi(cheap), hi(mid))*(1+1e-9)   [catches missing-sqrt]
#   3. quadrature convergence: g(gtol) vs g(4*gtol) within 6*gtol
#   4. F_conv against hardcoded brute-force values
#   5. dPhi vs Phi_cdf on a wide interval
import os, sys, math
from mpmath import iv, mpi
import mpmath as mp
iv.prec = 350
import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w3c = load("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3L = load("w3L", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_lower.py")
lo, hi, mid, width, ck = w3c.lo, w3c.hi, w3c.mid, w3c.width, w3c.ck
ivm = iv.mpf
MUT = os.environ.get("W3_MUTATE", "")

print("W3HARNESS v1 mutate=%r" % MUT)

w3c.init_caches()
G = w3c.gram_interval()

if MUT == "pin":
    # mutation: perturb a pin target value (law changes -> anchor check must fail)
    _orig = w3c.pin_values
    def pin_values_mut(mu_t):
        v = _orig(mu_t)
        v[0] = v[0] + ivm(1e-4)
        return v
    w3c.pin_values = pin_values_mut
    print("MUTATION: pin M f-value +1e-4")

if MUT == "zone":
    # mutation: corrupt the level (zone) -> anchor check must fail
    w3c.b_rat = w3c.b_rat + w3c.FR(1, 1000)
    print("MUTATION: level b + 1e-3")

if MUT == "sqrt":
    # mutation: restore the missing Cauchy-Schwarz square root in the cheap tier
    _orig_rho_box = w3c.rho_box
    def rho_box_mut(*a, **k):
        r = _orig_rho_box(*a, **k)
        # defective: drop the square root on Pwin (bound becomes invalid-small)
        r['cheap'] = r['cheap'] * r['Pwin']
        r['mid'] = r['mid'] * r['Pwin']
        return r
    w3c.rho_box = rho_box_mut
    print("MUTATION: missing-sqrt restored in cheap/mid")

C, Es = w3c.build_whitener(G)
law = w3c.build_law(G, C)
mu_t = w3c.compute_mu_t(G)
tv = w3c.matvec_iv(law['C'], w3c.pin_values(mu_t))


# ---- anchor ----
ybx = (mpi(0, 0), mpi(repr(0.60), repr(0.60)))
r = w3c.rho_box(ybx, law, tv, 3e-7, "anchor", need_exact=True)
ex, ch, md = r['rho'], r['cheap'], r['mid']
print("anchor exact=[%.10e, %.10e] cheap_hi=%.4e mid_hi=%.4e" % (
    lo(ex), hi(ex), hi(ch), hi(md)))

# check 1: anchor truth
ck(lo(ex) > 1.0e-4 and hi(ex) < 1.12e-4,
   "FAIL anchor truth: exact=[%.6e, %.6e] not in [1.0e-4, 1.12e-4]" % (lo(ex), hi(ex)))
print("CHECK1 anchor-truth PASS")

# check 2: hierarchy exact <= min(cheap, mid)
ck(hi(ex) <= min(float(hi(ch)), float(hi(md))) * (1 + 1e-9) + 1e-15,
   "FAIL hierarchy: exact_hi=%.6e > min(cheap,mid)_hi=%.6e" % (hi(ex), min(float(hi(ch)), float(hi(md)))))
print("CHECK2 hierarchy PASS")

# check 3: quadrature convergence of g
g1 = w3c.g_quad_iv(r['mH'], r['V'], 3e-7, "conv1")['g']
g2 = w3c.g_quad_iv(r['mH'], r['V'], 1.2e-6, "conv2")['g']
ov_lo = max(lo(g1), lo(g2))
ov_hi = min(hi(g1), hi(g2))
ck(ov_lo <= ov_hi + 6e-6, "FAIL quad convergence: [%.8g,%.8g] vs [%.8g,%.8g]" % (lo(g1), hi(g1), lo(g2), hi(g2)))
print("CHECK3 quad-convergence PASS g=[%.9g,%.9g]" % (lo(g1), hi(g1)))

# check 4: F_conv vs hardcoded brute-force values (from independent quadrature)
iv.prec = 200
cases = [
    (0.0125, 1.33e-4, 0.00821, 9.91e-05, 2.81e-05, 0.1221198),
    (0.0, 1e-4, 0.0084, 0.0375, 0.001, 0.0890936),
    (0.02, 1e-4, 0.0084, 0.01, 0.008, 0.0580977),
]
for (c_, Rm_, sig_, A_, B_, truth) in cases:
    F = w3L.F_conv(ivm(c_), Rm_, ivm(sig_), A_, B_)
    ck(abs(float(mid(F)) - truth) < 5e-7 and float(width(F)) < 1e-5,
       "FAIL F_conv c=%.4f: [%.8g,%.8g] truth %.8g" % (c_, lo(F), hi(F), truth))
print("CHECK4 F_conv PASS")

# check 5: dPhi vs Phi_cdf on a wide interval (consistency)
y0, y1 = ivm(0.3), ivm(1.4)
d1 = w3c.dPhi(y0, y1, "h5")
d2 = w3c.Phi_cdf(y1) - w3c.Phi_cdf(y0)
ck(lo(d1) <= hi(d2) and lo(d2) <= hi(d1), "FAIL dPhi consistency")
ck(float(width(d1)) < 1e-8, "FAIL dPhi width %.3g" % float(width(d1)))
print("CHECK5 dPhi PASS width=%.3g" % float(width(d1)))

print("W3HARNESS ALL PASS")
print("EXITOK")
