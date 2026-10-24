"""Cancellation quantification at the ORIGINAL weights w_m = exp(-alpha m^2), alpha=(2pi/L)^2/2.
Reads kernel certificates (TEST_ONLY point inputs) and compares:
  (i) interval subtraction from the kernel's own a=mu2,b=mu4,c=mu6 enclosures,
  (ii) cancellation-free double sums built from the same certificate's per-mode weight/h/z/tail enclosures.
All interval work on certificate data is exact Fraction arithmetic; mpmath only supplies reference values."""
import sys, json, glob, os
from fractions import Fraction as F
import mpmath as mp
mp.mp.dps = 700

def ref(L):
    L = mp.mpf(L.numerator) / L.denominator
    h = 2 * mp.pi / L; al = h * h / 2
    K = int(mp.sqrt(mp.mpf(1700) / al)) + 3   # exp(-al K^2) < 1e-738
    S = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    xs = [(mp.mpf(0), mp.mpf(1))]
    for m in range(1, K + 1):
        w = 2 * mp.e ** (-al * m * m); x = (h * m) ** 2
        xs.append((x, w))
        for k in range(4): S[k] += w * x ** k
    z = S[0]; a, b, c = S[1] / z, S[2] / z, S[3] / z
    # cancellation-free double sums over unordered distinct pairs (multiplicity folded into w)
    beta_cf = mp.mpf(0); D_cf = mp.mpf(0)
    for i in range(len(xs)):
        xi, wi = xs[i]
        for j in range(i + 1, len(xs)):
            xj, wj = xs[j]
            d2 = (xi - xj) ** 2
            beta_cf += wi * wj * d2; D_cf += wi * wj * xi * xj * d2
    # same-|m| pairs (+m,-m) have xi==xj -> contribute 0; zero-mode handled via xs[0]
    beta_cf /= z * z; D_cf /= z * z
    return dict(z=z, a=a, b=b, c=c, beta=b - a * a, D=a * c - b * b, nu3=c - b * b / a,
                beta_cf=beta_cf, D_cf=D_cf, nu3_cf=D_cf / a, alpha=al, h=h)

def iv(c, k): return F(c["bounds"][k][0]), F(c["bounds"][k][1])

def sub_forms(A, B, C):
    beta = (B[0] - A[1] ** 2, B[1] - A[0] ** 2)
    D = (A[0] * C[0] - B[1] ** 2, A[1] * C[1] - B[0] ** 2)
    if A[0] > 0:
        nu = (C[0] - B[1] ** 2 / A[0], C[1] - B[0] ** 2 / A[1])
    else:
        nu = (None, C[1] - (B[0] ** 2 / A[1] if A[1] > 0 else 0))  # lower = -inf
    return beta, D, nu

def cf_forms(c):
    """Rigorous enclosures of beta, ac-b^2, nu3 by nonnegative double sums from certificate data."""
    hlo, hhi = F(c["h"][0]), F(c["h"][1])
    zlo, zhi = iv(c, "z")
    A = iv(c, "mu2")
    modes = [(0, F(1), F(1))] + [(md["m"], F(md["weight"][0]), F(md["weight"][1])) for md in c["modes"]]
    blo = bhi = dlo = dhi = F(0)
    for i in range(len(modes)):
        mi, wil, wih = modes[i]
        for j in range(i + 1, len(modes)):
            mj, wjl, wjh = modes[j]
            # unordered pairs of |m| classes; signed multiplicity: 1 for m=0, 2 for m>0
            mult = (1 if mi == 0 else 2) * (1 if mj == 0 else 2)
            d2 = (mi * mi - mj * mj) ** 2           # (x_i-x_j)^2 = h^4 * d2
            p = mi * mi * mj * mj                   # x_i x_j = h^4 * p
            blo += mult * wil * wjl * d2 * hlo ** 4; bhi += mult * wih * wjh * d2 * hhi ** 4
            dlo += mult * wil * wjl * p * d2 * hlo ** 8; dhi += mult * wih * wjh * p * d2 * hhi ** 8
    T = {k: F(c["tail_upper"][str(2 * k)]) for k in range(4)}
    S = {k: F(c["raw"][str(2 * k)][1]) for k in range(4)}
    bhi += T[2] * S[0] + T[0] * S[2]          # omitted pairs, (x-y)^2 <= x^2+y^2
    dhi += T[3] * S[1] + T[1] * S[3]          # x y (x-y)^2 <= x^3 y + x y^3
    beta = (blo / zhi ** 2, bhi / zlo ** 2)
    D = (dlo / zhi ** 2, dhi / zlo ** 2)
    nu = (D[0] / A[1], D[1] / A[0]) if A[0] > 0 else (F(0), None)
    return beta, D, nu

def mpf(q): return None if q is None else mp.mpf(q.numerator) / q.denominator

def digits(lo, hi, true):
    """relative digits retained: -log10(width/|true|); None if unbounded."""
    if lo is None or hi is None: return None
    w = mpf(hi) - mpf(lo)
    if w == 0: return mp.inf
    return float(-mp.log10(w / abs(true)))

def fmt(x): return "  n/a " if x is None else ("%6.1f" % x)

rows = []
for fn in sorted(glob.glob(os.path.join(sys.argv[1], "cert_L*.json"))):
    c = json.load(open(fn))
    L = F(c["request"]["L_lo"]); setting = "max" if c["request"]["bits"] == 128 else "default"
    r = ref(L)
    A, B, C, Z = iv(c, "mu2"), iv(c, "mu4"), iv(c, "mu6"), iv(c, "z")
    for nm, I, t in (("z", Z, r["z"]), ("a", A, r["a"]), ("b", B, r["b"]), ("c", C, r["c"])):
        assert mpf(I[0]) <= t <= mpf(I[1]), ("CONTAINMENT FAIL", fn, nm)
    assert abs(r["beta"] - r["beta_cf"]) <= mp.mpf(10) ** -550 * r["beta"]
    assert abs(r["D"] - r["D_cf"]) <= mp.mpf(10) ** -450 * r["D"], (r["D"], r["D_cf"])
    sb, sD, sn = sub_forms(A, B, C)
    cb, cD, cn = cf_forms(c)
    for nm, I, t in (("beta_sub", sb, r["beta"]), ("D_sub", sD, r["D"]), ("beta_cf", cb, r["beta"]), ("D_cf", cD, r["D"]), ("nu_cf", cn, r["nu3"])):
        lo, hi = mpf(I[0]), mpf(I[1])
        assert (lo is None or lo <= t) and (hi is None or t <= hi), ("ENCLOSURE FAIL", nm, fn)
    if sn[0] is not None: assert mpf(sn[0]) <= r["nu3"] <= mpf(sn[1])
    in_d = min(digits(*A, r["a"]), digits(*B, r["b"]), digits(*C, r["c"]))
    kb = (r["b"] + 2 * r["a"] ** 2) / r["beta"]
    kn = (r["c"] + 3 * r["b"] ** 2 / r["a"]) / r["nu3"]
    kD = (2 * r["a"] * r["c"] + 2 * r["b"] ** 2) / r["D"]
    rows.append(dict(L=str(L), setting=setting, M=c["request"]["cutoff"], bits=c["request"]["bits"],
        alpha=float(r["alpha"]), a=mp.nstr(r["a"], 8), b=mp.nstr(r["b"], 8), c=mp.nstr(r["c"], 8),
        beta=mp.nstr(r["beta"], 8), nu3=mp.nstr(r["nu3"], 8), D=mp.nstr(r["D"], 8),
        nu3_over_c=mp.nstr(r["nu3"] / r["c"], 6), beta_over_b=mp.nstr(r["beta"] / r["b"], 6),
        log10_kappa_beta=float(mp.log10(kb)), log10_kappa_nu3=float(mp.log10(kn)), log10_kappa_D=float(mp.log10(kD)),
        digits_a=digits(*A, r["a"]), digits_b=digits(*B, r["b"]), digits_c=digits(*C, r["c"]), digits_in=in_d,
        A_lo_positive=A[0] > 0,
        beta_sub_digits=digits(*sb, r["beta"]), beta_sub_lo_pos=sb[0] > 0,
        nu_sub_digits=digits(*sn, r["nu3"]), nu_sub_lo_pos=(sn[0] is not None and sn[0] > 0),
        nu_sub_lo=None if sn[0] is None else mp.nstr(mpf(sn[0]), 5), nu_sub_hi=mp.nstr(mpf(sn[1]), 5),
        D_sub_digits=digits(*sD, r["D"]), D_sub_lo_pos=sD[0] > 0, D_sub_lo=mp.nstr(mpf(sD[0]), 5), D_sub_hi=mp.nstr(mpf(sD[1]), 5),
        beta_cf_digits=digits(*cb, r["beta"]), beta_cf_lo_pos=cb[0] > 0,
        D_cf_digits=digits(*cD, r["D"]), D_cf_lo_pos=cD[0] > 0,
        nu_cf_digits=digits(*cn, r["nu3"]), nu_cf_lo_pos=(cn[0] is not None and cn[0] > 0),
        nu_cf_lo=mp.nstr(mpf(cn[0]), 5), nu_cf_hi=None if cn[1] is None else mp.nstr(mpf(cn[1]), 5)))
json.dump(rows, open(os.path.join(sys.argv[1], "cancel_rows.json"), "w"), indent=1, default=str)
order = {"1/2": 0, "1": 1, "2": 2, "3": 3}
rows.sort(key=lambda x: (order[x["L"]], x["setting"]))
print("L    set     M bits | true a          b               c               | nu3/c       beta/b   | log10 kappa beta nu3  D")
for x in rows:
    print("%-4s %-7s %2d %3d | %s %s %s | %s %s | %5.2f %6.1f %6.1f" % (x["L"], x["setting"], x["M"], x["bits"], x["a"], x["b"], x["c"], x["nu3_over_c"], x["beta_over_b"], x["log10_kappa_beta"], x["log10_kappa_nu3"], x["log10_kappa_D"]))
print()
print("relative digits retained (-log10(width/true)); lo>0 = certified positivity")
print("L    set     | a      b      c      | beta_sub lo>0 | nu3_sub lo>0 | D_sub lo>0 | beta_cf lo>0 | nu3_cf lo>0 | D_cf lo>0")
for x in rows:
    print("%-4s %-7s | %s %s %s | %s %-5s | %s %-5s | %s %-5s | %s %-5s | %s %-5s | %s %-5s" % (x["L"], x["setting"], fmt(x["digits_a"]), fmt(x["digits_b"]), fmt(x["digits_c"]),
        fmt(x["beta_sub_digits"]), x["beta_sub_lo_pos"], fmt(x["nu_sub_digits"]), x["nu_sub_lo_pos"], fmt(x["D_sub_digits"]), x["D_sub_lo_pos"],
        fmt(x["beta_cf_digits"]), x["beta_cf_lo_pos"], fmt(x["nu_cf_digits"]), x["nu_cf_lo_pos"], fmt(x["D_cf_digits"]), x["D_cf_lo_pos"]))
print()
for x in rows:
    print(x["L"], x["setting"], "true nu3", x["nu3"], "| nu3_sub [", x["nu_sub_lo"], ",", x["nu_sub_hi"], "] | nu3_cf [", x["nu_cf_lo"], ",", x["nu_cf_hi"], "]", "| true D", x["D"], "D_sub [", x["D_sub_lo"], ",", x["D_sub_hi"], "]")
