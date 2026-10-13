import math, importlib.util, sys, io, contextlib
# reuse the definitions of sliceA185.py without re-printing its report
src = open("sliceA185.py").read().split("print(\"[1]")[0]  # definitions only (up to the first report line)
ns = {}
exec(src, ns)
# pull the numeric helpers defined later in the file by executing the remainder silently
rest = open("sliceA185.py").read()
with contextlib.redirect_stdout(io.StringIO()):
    exec(rest, ns)
root_b, root_c, a_of, A_eps, S_eps, fe, S, tv_pair, integ, Kval = (ns[k] for k in ("root_b", "root_c", "a_of", "A_eps", "S_eps", "fe", "S", "tv_pair", "integ", "Kval"))
F = ns["F"]
Cs = float(F(486, 169)) * 2 ** (-2 / 3); kap = float(F(2816, 819)) / math.sqrt(3)
print("[11] sharper expansion check: (P(n=1|eps)/(C_s eps^(1/3)) - 1)/sqrt(eps)  -> -kappa = %.5f ?" % (-kap))
for eps in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    P = S_eps(eps) / A_eps(eps)
    print("     eps=%.0e  %+.5f   (S(eps)/lead - 1)/eps^(2/3) = %+.5f" % (eps, (P / (Cs * eps ** (1 / 3)) - 1) / math.sqrt(eps), (S(eps) / (10368 * 2 ** (-2 / 3) * eps ** (10 / 3)) - 1) / eps ** (2 / 3)))
print("[12] tagged TV at smaller eps: (TV_tag - P1)/sqrt(eps) stays within [0, TV/sqrt(eps)] (triangle inequality bound)")
for eps in (1e-5, 1e-6):
    tv = tv_pair(eps, False); tvt = tv_pair(eps, True); P1 = S_eps(eps) / A_eps(eps)
    print("     eps=%.0e  TV/sqrt(eps)=%.5f  (TV_tag-P1)/sqrt(eps)=%.5f  TV_tag/eps^(1/3)=%.5f  P1/eps^(1/3)=%.5f" % (eps, tv / math.sqrt(eps), (tvt - P1) / math.sqrt(eps), tvt / eps ** (1 / 3), P1 / eps ** (1 / 3)))
print("[13] 0 < c(e) < b(e) < 1/2 on a grid of e in (0,1), and K > 0 at the midpoint x of every slice:")
ok = True; kmin = float("inf")
for k in range(1, 1000):
    ev = k / 1000
    b, c = root_b(ev), root_c(ev)
    ok &= 0 < c < b < 0.5
    for xv in (a_of(ev) * 0.5, c, 0.5 * (c + b)):
        kmin = min(kmin, Kval(ev, xv))
print("     ordering holds:", ok, "| min K sampled: %.3e" % kmin)
