"""Targeted soundness mutations: each must be REJECTED unless noted."""
import sys, json, copy
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
from moment_kernel import certify, KernelHold, Arithmetic, _exp_negative
from verify_certificate import verify
def qs(q): return f"{q.numerator}/{q.denominator}"
def J(c): return json.loads(json.dumps(c))
P = J(certify(dict(L_lo="1/1", L_hi="1/1", cutoff=2, input_role="TEST_ONLY", width_goal="1/1000000000000")))
I = J(certify(dict(L_lo="1/1", L_hi="2/1", cutoff=2, input_role="TEST_ONLY", width_goal="1/1000000000000")))
res = []
def t(name, c, expect_reject=True):
    try:
        verify(c); r = "ACCEPT"
    except ValueError as e:
        r = "REJECT:" + str(e)
    ok = r.startswith("REJECT") == expect_reject
    res.append(ok); print(("OK  " if ok else "BAD ") + name + " -> " + r)
c = copy.deepcopy(P); c["request"]["L_hi"] = "2/1"; t("point cert relabelled as [1,2] (claim widening)", c)
c = copy.deepcopy(I); c["request"]["L_hi"] = "3/1"; t("interval cert relabelled [1,3]", c)
c = copy.deepcopy(I); c["request"]["L_lo"] = "1/2"; t("interval cert relabelled [1/2,2]", c)
# mu upper computed as raw_hi/z_hi (wrong pairing)
c = copy.deepcopy(I)
for k in ("2","4","6"):
    rhi = Q(c["raw"][k][1]); zhi = Q(c["raw"]["0"][1]); D = 2**80
    c["bounds"]["mu"+k][1] = qs(Q(-((-rhi*D).numerator // (rhi*D).denominator) if False else -(-(rhi/zhi*D).numerator // (rhi/zhi*D).denominator), D))
t("mu upper = raw_hi/z_hi (wrong pairing)", c)
c = copy.deepcopy(I)
for k in ("2","4","6"):
    rlo = Q(c["raw"][k][0]); zlo = Q(c["raw"]["0"][0]); D = 2**80
    c["bounds"]["mu"+k][0] = qs(Q((rlo/zlo*D).numerator // (rlo/zlo*D).denominator, D))
t("mu lower = raw_lo/z_lo (wrong pairing)", c)
# drop symmetric factor 2 in mode contributions and finite totals
c = copy.deepcopy(P)
for mode in c["modes"]:
    for k in mode["contribution"]:
        mode["contribution"][k] = [qs(Q(x)/2) for x in mode["contribution"][k]]
t("contributions without +/- factor 2", c)
# z normalizer from finite only (no tail) in interval case
c = copy.deepcopy(I); c["raw"]["0"] = list(c["finite"]["0"]); c["bounds"]["z"] = list(c["finite"]["0"])
t("raw z without tail", c)
# outcome lie
c = copy.deepcopy(I); c["outcome"] = "TOLERANCE_MET"; t("interval outcome forged TOLERANCE_MET", c)
c = copy.deepcopy(I); c["outcome"] = "HOLD_TARGET_UNMET"; t("interval outcome downgraded to TARGET_UNMET with proven spread", c)
c = copy.deepcopy(I); c["normalizer_input_spread_lower"] = "0/1"; c["outcome"] = "HOLD_TARGET_UNMET"
t("spread understated to 0 + TARGET_UNMET (sound, spec permits)", c, expect_reject=False)
c = copy.deepcopy(I); c["normalizer_input_spread_lower"] = qs(Q(c["bounds"]["z"][1]) - Q(c["bounds"]["z"][0]) + 1)
t("spread overstated", c)
# exp witness using the WRONG endpoint (alpha_lo for the lower weight)
c = copy.deepcopy(I); c["modes"][0]["exp_at_hi"] = copy.deepcopy(c["modes"][0]["exp_at_lo"]); t("lower weight from alpha_lo witness", c)
# tail exp at alpha_hi*M^2 (smaller E -> understated tail)
c = copy.deepcopy(I)
ar = Arithmetic(80); al_hi = Q(c["alpha"][1])
c["tail_exp"] = J(_exp_negative(al_hi*4, 16, ar)); t("tail exp at alpha_hi*M^2", c)
# tail multiplier using h_lo instead of h_hi
c = copy.deepcopy(I)
hlo = Q(c["h"][0]); ints = [Q(x) for x in c["tail_integral_upper"]]
for k, key in enumerate(("0","2","4","6")):
    c["tail_upper"][key] = qs(Q(-(-(2*hlo**(2*k)*ints[k]*2**80).numerator // (2*hlo**(2*k)*ints[k]*2**80).denominator), 2**80))
t("tail multiplier with h_lo", c)
# pi_terms mismatch
c = copy.deepcopy(P); c["pi"]["terms"] = 19; t("pi terms mismatch", c)
c = copy.deepcopy(P); c["request"]["exp_pairs"] = 15; t("exp_pairs claim mismatch", c)
c = copy.deepcopy(P); c["request"]["cutoff"] = 1; c["modes"] = c["modes"][:1]; t("cutoff relabelled 1 dropping mode", c)
# tail threshold: 2*alpha*M^2 claim with k recurrences ok, but tamper alpha_lo to small -> threshold
c = copy.deepcopy(P); c["alpha"][0] = "0/1"; t("alpha_lo=0", c)
print("ALL_AS_EXPECTED", all(res))
