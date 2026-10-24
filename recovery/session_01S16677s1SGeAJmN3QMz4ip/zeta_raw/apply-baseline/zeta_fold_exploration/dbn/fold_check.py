"""Local fold test at an actual collision of de Bruijn--Newman zeros.

Usage: python3 dbn/fold_check.py <t_star_guess> <x0_guess> <tag>
Outputs: results/dbn/fold_<tag>.csv and results/dbn/fold_<tag>.json (stdout saved as fold_<tag>.out).

1. Refine the double zero (x0, t*) by 2-D Newton on (H, H_z) = 0 (Jacobian det = H_zz^2 != 0).
2. Heat-polynomial Taylor coefficients at (x0, t*):  H = A p2 + B p3 + C p4 + ...,
   A = H''/2, B = H'''/6, C = H''''/24, p2 = x^2-2tau, p3 = x^3-6x tau, p4 = x^4-12x^2 tau+12tau^2.
3. For tau = t - t* > 0 locate the two real zeros z_-(t) < z_+(t) near x0 and test
   (i)  r^2 = (z_+ - z_-)^2  vs  8 tau                  [note's x_pm ~ +-sqrt(2 tau)]
   (ii) ell = |int_{z_-}^{z_+} H_t dz|  vs  |A| r^3/6   [note's cubic fold height]
   and the second-order predictions derived by hand from the heat-polynomial expansion:
        r^2/(8 tau) - 1         ~ kappa_r   * tau,  kappa_r   = 4C/A - 2B^2/A^2
        ell/(|A| r^3/6) - 1     ~ kappa_ell * tau,  kappa_ell = 6B^2/A^2 - 48C/(5A)
        midpoint (z_+ + z_-)/2 - x0 ~ 2(B/A) tau
4. For tau < 0 confirm there is no real zero near x0 (the pair is complex) and locate the
   complex-conjugate pair z = x0 + iy by Newton on the analytic continuation of the quadrature sum
   (expected y ~ +-sqrt(2|tau|)).
"""
import sys, csv, json, mpmath as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "dbn"

from hflow import HFlow
from zeros import refine_root

tg, xg, tag = mp.mpf(sys.argv[1]), mp.mpf(sys.argv[2]), sys.argv[3]
hf = HFlow(dps=90)
mp.mp.dps = 90
tg, xg = mp.mpf(sys.argv[1]), mp.mpf(sys.argv[2])

# --- 1. refine double zero
x, t = xg, tg
for it in range(40):
    d = hf.derivs(x, t, 3)
    H, H1, H2, H3 = d
    # dH/dt = -H2, dH1/dt = -H3
    J = mp.matrix([[H1, -H2], [H2, -H3]])
    dx = mp.lu_solve(J, mp.matrix([-H, -H1]))
    x += dx[0]; t += dx[1]
    if abs(dx[0]) + abs(dx[1]) < mp.mpf(10) ** -70:
        break
x0, ts = x, t
d = hf.derivs(x0, ts, 4)
A, B, C = d[2] / 2, d[3] / 6, d[4] / 24
scale = abs(A)
print('t* =', mp.nstr(ts, 50))
print('x0 =', mp.nstr(x0, 50), ' (gamma-units x0/2 =', mp.nstr(x0 / 2, 30), ')')
print('residuals H/|A| =', mp.nstr(d[0] / scale, 3), ' H_z/|A| =', mp.nstr(d[1] / scale, 3))
print('A =', mp.nstr(A, 30)); print('B/A =', mp.nstr(B / A, 30)); print('C/A =', mp.nstr(C / A, 30))
# check heat equation directly at the double zero: H_t = -H_zz
print('H_t/(-2A) =', mp.nstr(hf.dt(x0, ts) / (-2 * A), 30))
kr = 4 * C / A - 2 * (B / A) ** 2
kl = 6 * (B / A) ** 2 - mp.mpf(48) / 5 * C / A
print('kappa_r =', mp.nstr(kr, 20), ' kappa_ell =', mp.nstr(kl, 20), ' midpoint slope 2B/A =', mp.nstr(2 * B / A, 20))

# --- 3. tau > 0 : two real zeros
rows = []
exps = [mp.mpf(k) / 4 for k in range(-4, 49)]      # tau = 10^(-k/4), k=-4..48: 10 .. 1e-12
taus = [mp.mpf(10) ** (-e) for e in exps]
taus = [tau for tau in taus if tau <= abs(ts)] + [abs(ts)]   # also tau = |t*|, i.e. t = 0
taus = sorted(set(taus), reverse=True)
for tau in taus:
    t = ts + tau
    # critical point between the pair (continuation from x0 by Newton on H_z)
    c = x0 + 2 * B / A * tau
    for _ in range(60):
        dd = hf.derivs(c, t, 2); st = dd[1] / dd[2]; c -= st
        if abs(st) < mp.mpf(10) ** -75: break
    Hc = hf.H(c, t)
    w = mp.sqrt(2 * tau)
    f = lambda z: hf.H(z, t)
    zs = []
    for sgn in (-1, 1):
        lo, hi = c, c + sgn * 0.5 * w
        while f(hi) * Hc > 0:
            hi = c + (hi - c) * mp.mpf('1.25')
            if abs(hi - c) > 6 * w + 3:
                raise RuntimeError('no bracket at tau=%s' % tau)
        zs.append(refine_root(f, lo, hi, Hc, f(hi), tol=mp.mpf(10) ** -70))
    zm, zp = min(zs), max(zs)
    r = zp - zm
    ell = abs(hf.F(zp, t) - hf.F(zm, t))
    pred_ell = abs(A) * r ** 3 / 6
    mid = (zp + zm) / 2 - x0
    row = dict(tag=tag, tau=tau, t=t, z_minus=zm, z_plus=zp, r=r, r2=r ** 2, pred_r2=8 * tau,
               relerr_r2=r ** 2 / (8 * tau) - 1, pred2_relerr_r2=kr * tau,
               ell=ell, pred_ell=pred_ell, relerr_ell=ell / pred_ell - 1, pred2_relerr_ell=kl * tau,
               ell_over_A=ell / abs(A), midpoint_shift=mid, pred_midpoint_shift=2 * B / A * tau)
    rows.append(row)
    print('tau=%-10s r=%-12s r2/8tau-1=%-11s (pred %-11s) ell/pred-1=%-11s (pred %-11s) mid=%-11s (pred %s)' % (
        mp.nstr(tau, 4), mp.nstr(r, 8), mp.nstr(row['relerr_r2'], 4), mp.nstr(kr * tau, 4),
        mp.nstr(row['relerr_ell'], 4), mp.nstr(kl * tau, 4), mp.nstr(mid, 4), mp.nstr(2 * B / A * tau, 4)))
    sys.stdout.flush()

# --- 4. tau < 0 : the pair is a complex-conjugate pair x0 + i y
def Hc_complex(zc, t):
    # analytic continuation of the quadrature sum to complex z (mpmath, direct)
    s = mp.mpc(0)
    for k in range(hf.N):
        u = hf.u[k]
        s += hf.W[k] * mp.exp(t * u * u) * mp.cos(zc * u)
    return s
def Hc_complex_d(zc, t):
    s = mp.mpc(0)
    for k in range(hf.N):
        u = hf.u[k]
        s += -hf.W[k] * mp.exp(t * u * u) * u * mp.sin(zc * u)
    return s
cplx = []
for tau in [mp.mpf('1e-2'), mp.mpf('1e-4'), mp.mpf('1e-6')]:
    t = ts - tau
    zc = x0 + 2 * B / A * (-tau) + 1j * mp.sqrt(2 * tau)
    for _ in range(60):
        st = Hc_complex(zc, t) / Hc_complex_d(zc, t); zc -= st
        if abs(st) < mp.mpf(10) ** -60: break
    cplx.append((tau, zc))
    print('tau=-%s complex zero x0+iy: Re-x0=%s  Im=%s  Im/sqrt(2|tau|)=%s' % (
        mp.nstr(tau, 3), mp.nstr(zc.real - x0, 6), mp.nstr(zc.imag, 12), mp.nstr(zc.imag / mp.sqrt(2 * tau), 12)))

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / ('fold_%s.csv' % tag), 'w', newline='') as fh:
    keys = list(rows[0].keys())
    wr = csv.writer(fh)
    wr.writerow(keys)
    for row in rows:
        wr.writerow([row[k] if isinstance(row[k], str) else mp.nstr(row[k], 25) for k in keys])
json.dump(dict(tag=tag, t_star=mp.nstr(ts, 45), x0=mp.nstr(x0, 45), A=mp.nstr(A, 25), B_over_A=mp.nstr(B / A, 25),
               C_over_A=mp.nstr(C / A, 25), kappa_r=mp.nstr(kr, 15), kappa_ell=mp.nstr(kl, 15),
               complex_check=[(mp.nstr(a, 3), mp.nstr(b.real - x0, 10), mp.nstr(b.imag, 15)) for a, b in cplx]),
          open(OUT / ('fold_%s.json' % tag), 'w'), indent=1)
