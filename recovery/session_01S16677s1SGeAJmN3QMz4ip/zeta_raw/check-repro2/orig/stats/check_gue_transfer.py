"""Section 5 checks: small-r expansion of the GUE pair correlation; the
pushforward of a separation law r^beta dr under ell = k r^3, including the
admission (integrability) condition on the mark k.

Notation: in this script "beta" is the exponent of the separation law r^beta dr (the pair or
gap exponent, written alpha in the README). The README reserves beta = (alpha+1)/m for the
Math- collision_mechanism sense. Output: results/stats/check_gue_transfer.out
"""
from mpmath import mp

from stats_paths import out_path

mp.dps = 30
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

R2 = lambda r: 1 - (mp.sin(mp.pi * r) / (mp.pi * r)) ** 2 if r != 0 else mp.mpf(0)
c = mp.taylor(lambda r: 1 - mp.sinc(mp.pi * r) ** 2, 0, 6)
rep("Taylor coefficients of 1-(sin pi r/(pi r))^2 at r=0:", [mp.nstr(x, 15) for x in c])
rep("pi^2/3 =", mp.nstr(mp.pi**2 / 3, 15), "; -2 pi^4/45 =", mp.nstr(-2 * mp.pi**4 / 45, 15))
# GUE nearest-neighbour spacing density p(s) also ~ (pi^2/3) s^2; Wigner surmise (32/pi^2) s^2 e^{-4 s^2/pi}
rep("Wigner-surmise (beta=2) small-s coefficient 32/pi^2 =", mp.nstr(32 / mp.pi**2, 10), "(surmise only; exact GUE gap density has pi^2/3)")

# exact pushforward: separation density c r^beta on (0, r_max), ell = k r^3, k fixed
# => nu(ell) = (c/3) k^{-(beta+1)/3} ell^{(beta-2)/3}
for beta in [0, 1, 2]:
    rep(f"beta={beta}: nu(ell) ∝ ell^({beta}-2)/3 = ell^{mp.nstr(mp.mpf(beta-2)/3,4)}; mark moment needed: E[k^(-{beta+1}/3)] < inf")

# Deterministic check (quadrature, no sampling noise): separation density (beta+1) r^beta on (0,1),
# mark law mu(dk), ell = k r^3.  nu(ell) = int mu(dk) (beta+1) (ell/k)^{beta/3} (1/3) k^{-1/3} ell^{-2/3} 1{ell<k}
def nu(ell, beta, mark):
    ell = mp.mpf(ell)
    f = lambda k: (beta + 1) * (ell / k) ** (mp.mpf(beta) / 3) * k ** (-mp.mpf(1) / 3) * ell ** (-mp.mpf(2) / 3) / 3
    if mark == "k=1":
        return f(mp.mpf(1)) if ell < 1 else 0
    if mark == "U[0.5,2]":
        return mp.quad(lambda k: f(k) / mp.mpf("1.5"), [max(ell, mp.mpf("0.5")), 2])
    if mark == "U(0,1)":
        return mp.quad(f, [ell, 1])
for beta in [1, 2]:
    for mark in ["k=1", "U[0.5,2]", "U(0,1)"]:
        vals = []
        for ell in ["1e-4", "1e-8", "1e-12"]:
            e = mp.mpf(ell)
            sl = (mp.log(nu(e * mp.mpf("1.01"), beta, mark)) - mp.log(nu(e / mp.mpf("1.01"), beta, mark))) / (2 * mp.log(mp.mpf("1.01")))
            vals.append(f"ell={ell}: nu={mp.nstr(nu(e,beta,mark),8)}, local slope={mp.nstr(sl,5)}")
        rep(f"beta={beta}, mark {mark}: " + "; ".join(vals) + f"  [pure-power prediction {(beta-2)/3:+.4f}]")
rep("beta=2, k~U(0,1): exact nu(ell) = int_ell^1 dk/k = log(1/ell); check at ell=1e-8:", mp.nstr(nu(mp.mpf('1e-8'),2,'U(0,1)'),12), "vs", mp.nstr(mp.log(mp.mpf(10)**8),12))

out_path("check_gue_transfer.out").write_text("\n".join(out) + "\n")
