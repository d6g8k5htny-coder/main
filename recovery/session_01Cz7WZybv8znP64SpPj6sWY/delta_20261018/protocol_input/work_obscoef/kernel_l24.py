"""Run the frozen moment kernel (copied, hash-checked) at point L=24/1 for several
cutoffs/bits in UNASSIGNED/TEST role, verify each certificate with the standalone
checker, and report widths of z, mu2, mu4, mu6 plus interval-subtraction widths
of beta, nu3, kappa4, kappa6 and the implied first-order C_alg(2,24) width.
Moment arithmetic only; no field, sampler, driver or experiment."""
import importlib.util, os, sys, json, hashlib, time
from fractions import Fraction as Q

here = os.path.dirname(os.path.abspath(__file__))
kd = os.path.join(here, "kern")
want = {"moment_kernel.py": "68c3caf28190e8be58464cabfc39ab565449d9bb64716ab139ac0dd0a181193e",
        "verify_certificate.py": "a0afbfdf9ca323db38e06171c138f683669c42fe99f6892d6233ab03802d4715"}
for f, h in want.items():
    assert hashlib.sha256(open(os.path.join(kd, f), "rb").read()).hexdigest() == h, f

def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(kd, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

mk = load("moment_kernel")
vc = load("verify_certificate")

def iv(p):
    return Q(p[0]), Q(p[1])

def sub(x, y): return (x[0]-y[1], x[1]-y[0])
def mul(x, y):
    c = [x[0]*y[0], x[0]*y[1], x[1]*y[0], x[1]*y[1]]; return (min(c), max(c))
def div(x, y):
    assert y[0] > 0
    c = [x[0]/y[0], x[0]/y[1], x[1]/y[0], x[1]/y[1]]; return (min(c), max(c))
def sc(k, x): return (k*x[0], k*x[1]) if k >= 0 else (k*x[1], k*x[0])

for M in (10, 24, 40, 64):
    for bits in (80, 128):
        req = {"L_lo": "24/1", "L_hi": "24/1", "cutoff": M, "bits": bits, "pi_terms": 32, "exp_pairs": 24,
               "input_role": "UNASSIGNED_PARAMETER_INPUT"}
        t0 = time.time()
        try:
            cert = mk.certify(req)
        except mk.KernelHold as e:
            print("M=%d bits=%d HOLD %s" % (M, bits, e)); continue
        dt = time.time() - t0
        ok = True
        try:
            vc.verify(cert)
        except Exception as e:
            ok = False; print("verify failed", e)
        B = {k: iv(v) for k, v in cert["bounds"].items()}
        a, b, c = B["mu2"], B["mu4"], B["mu6"]
        beta = sub(b, mul(a, a))
        nu3 = sub(c, div(mul(b, b), a))
        k4 = sub(b, sc(3, mul(a, a)))
        k6 = (lambda x: x)(sub(sub(c, sc(15, mul(a, b))), sc(-30, mul(mul(a, a), a))))
        w = {k: float(v[1]-v[0]) for k, v in B.items()}
        tails = {k: float(Q(v)) for k, v in cert["tail_upper"].items()}
        # first-order relative C width: dlnC/dlna=1, dlnC/dk4=0.75, dlnC/dk6=1/14.4 (from calg_l24.py)
        relC = float((a[1]-a[0]) + Q(3, 4)*(k4[1]-k4[0]) + Q(5, 72)*(k6[1]-k6[0]))
        print("M=%d bits=%d verify=%s %.2fs widths z=%.3g mu2=%.3g mu4=%.3g mu6=%.3g | tail6=%.3g | beta=[%.6g,%.6g] nu3=[%.6g,%.6g] k4w=%.3g k6w=%.3g | ~relC width %.3g"
              % (M, bits, ok, dt, w["z"], w["mu2"], w["mu4"], w["mu6"], tails["6"],
                 float(beta[0]), float(beta[1]), float(nu3[0]), float(nu3[1]),
                 float(k4[1]-k4[0]), float(k6[1]-k6[0]), relC))
