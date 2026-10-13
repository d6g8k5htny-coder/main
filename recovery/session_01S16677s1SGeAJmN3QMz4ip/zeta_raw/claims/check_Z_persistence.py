"""What the H0 persistence of an antiderivative of Hardy's Z (or of Xi) measures,
on sampled windows.  Usage: python3 check_Z_persistence.py Z_grid_600_900.npz"""
import sys
import numpy as np
from pers1d import superlevel_h0

fn = sys.argv[1]
d = np.load(fn)
t, Z, logw = d["t"], d["Z"], d["logw"]
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

rep(f"window {t[0]:.0f}..{t[-1]:.0f}, step {t[1]-t[0]:.4f}, {len(t)} samples")
# zeros by sign change + linear interpolation
idx = np.nonzero(np.sign(Z[1:]) != np.sign(Z[:-1]))[0]
zer = t[idx] - Z[idx] * (t[idx + 1] - t[idx]) / (Z[idx + 1] - Z[idx])
rep(f"zeros found: {len(zer)}; Riemann-von Mangoldt main term N(T1)-N(T0) = "
    f"{(t[-1]/(2*np.pi))*np.log(t[-1]/(2*np.pi*np.e)) - (t[0]/(2*np.pi))*np.log(t[0]/(2*np.pi*np.e)):.1f}")

def antideriv(f):
    return np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(t))])

FZ = antideriv(Z)
Fz = np.interp(zer, t, FZ)
gaps = np.diff(zer)
lobes = np.abs(np.diff(Fz))  # area of |Z| between consecutive zeros
mid = (zer[1:] + zer[:-1]) / 2
sgap = gaps * np.log(mid / (2 * np.pi)) / (2 * np.pi)  # unit mean spacing
k_eff = lobes / gaps**3
q = np.quantile(k_eff, [0.05, 0.5, 0.95])
rep(f"lobe area / gap^3 ('k' if bars were single lobes): 5%={q[0]:.3g}, median={q[1]:.3g}, 95%={q[2]:.3g}, max/min={k_eff.max()/k_eff.min():.3g}")
amp = lobes / gaps
qa = np.quantile(amp, [0.05, 0.5, 0.95])
rep(f"mean |Z| on lobe (area/gap): 5%={qa[0]:.3g}, median={qa[1]:.3g}, 95%={qa[2]:.3g}")
sl = np.polyfit(np.log(gaps), np.log(lobes), 1)[0]
rep(f"regression slope of log(lobe area) on log(gap): {sl:.3f}  (cubic fold law would give 3 only for isolated close pairs with controlled curvature)")
rep(f"normalized gaps: min={sgap.min():.3f}, fraction < 0.5: {(sgap<0.5).mean():.3f} (GUE ~0.11; Poisson ~0.39)")

def pers_report(F, label):
    Fc = np.interp(zer, t, F)  # values at critical points of F (zeros of integrand)
    bars, ess = superlevel_h0(list(Fc))
    adj = sum(1 for (b, dth, bv, dv) in bars if abs(b - dth) == 1)
    L = np.array([bv - dv for (b, dth, bv, dv) in bars])
    rep(f"[{label}] finite bars: {len(bars)}; bars between ADJACENT critical points (single lobe): {adj} ({adj/len(bars):.1%}); "
        f"bar length quantiles 5/50/95%: {np.quantile(L,[0.05,0.5,0.95])}")
    return {(b, dth) for (b, dth, bv, dv) in bars}, bars

pairsZ, barsZ = pers_report(FZ, "F = int Z (Hardy normalization)")
# Xi normalization: Xi(t) = -w(t) Z(t); scale w by its value at the left end (positive constant)
w = np.exp(logw - logw[0])
rep(f"Xi weight w(t)/w(t0) over the window: from 1 to {w[-1]:.3g}")
# reverse accumulation for accuracy where w is tiny: G(t_i) = -int_{t_i}^{T1} w Z, an antiderivative of w Z
wZ = w * Z
G = -np.concatenate([np.cumsum(((wZ[1:] + wZ[:-1]) / 2 * np.diff(t))[::-1])[::-1], [0.0]])
pairsW, barsW = pers_report(G, "F = int w Z (Xi weight, sign removed)")
rep(f"pairs shared by Hardy and Xi-weight (same sign) normalizations: {len(pairsZ & pairsW)} of {len(pairsZ)}")
pairsX, barsX = pers_report(-G, "F = int Xi = -int w Z (actual Xi sign)")
rep(f"pairs shared by Hardy and actual-Xi normalizations: {len(pairsZ & pairsX)} of {len(pairsZ)}  (sign flip exchanges superlevel and sublevel roles)")
# lobe statistics in the Xi-weight normalization
Gz = np.interp(zer, t, G)
lobesW = np.abs(np.diff(Gz))
kW = lobesW / gaps**3
rep(f"Xi-weight lobe area/gap^3: max/min = {kW.max()/kW.min():.3g}; regression slope log area vs log gap = {np.polyfit(np.log(gaps), np.log(lobesW), 1)[0]:.3f}")
# how much of the lobe-area spread does the gap explain (Hardy normalization)?
res = np.log(lobes) - np.polyval(np.polyfit(np.log(gaps), np.log(lobes), 1), np.log(gaps))
rep(f"Hardy normalization: sd of log(lobe area) = {np.log(lobes).std():.3f}; residual sd after regressing on log gap = {res.std():.3f}")
# short bars: are the shortest persistence bars the closest zero pairs?
Ls = sorted(((bv - dv), b, dth) for (b, dth, bv, dv) in barsZ)
short = Ls[:10]
rep("ten shortest Hardy bars: (length, gap between birth/death crit points if adjacent, normalized gap)")
for (Lb, b, dth) in short:
    if abs(b - dth) == 1:
        i = min(b, dth)
        rep(f"   {Lb:.4g}  gap={gaps[i]:.4f}  norm.gap={sgap[i]:.3f}  ell/gap^3={Lb/gaps[i]**3:.3f}")
    else:
        rep(f"   {Lb:.4g}  (non-adjacent pair, {abs(b-dth)} critical points apart)")
# a mild weight: w = exp(0.01 (t - t0))
wm = np.exp(0.01 * (t - t[0]))
pairsM, _ = pers_report(antideriv(wm * Z), "F = int e^{0.01(t-t0)} Z (mild positive weight)")
rep(f"pairs shared between Hardy and mild-weight normalizations: {len(pairsZ & pairsM)} of {len(pairsZ)}")
open(fn.replace(".npz", "_persistence.out"), "w").write("\n".join(out) + "\n")
