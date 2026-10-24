# Vectorized "no missed critical point" check: |grad F_lambda| on a dense log-polar grid of the chart ball, with Newton polish of every local minimum of |grad|.
import numpy as np, instance as I
from scipy.optimize import fsolve
def run(lam):
    rad = np.geomspace(1e-7, I.R, 3000); ang = np.linspace(0, 2*np.pi, 721)[:-1]
    RR, AA = np.meshgrid(rad, ang, indexing='ij'); S, T = RR*np.cos(AA), RR*np.sin(AA)
    G = I.gradFlam(S, T, lam); gn = np.sqrt((G**2).sum(0))
    nb = [np.roll(np.roll(gn, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    lm = np.ones_like(gn, bool)
    for q in nb: lm &= gn <= q
    lm[0, :] = lm[-1, :] = False
    cand = list(zip(S[lm], T[lm]))
    found = []
    for s0, t0 in cand:
        sol, info, ier, msg = fsolve(lambda x: I.gradFlam(x[0], x[1], lam), [s0, t0], full_output=True, xtol=1e-15)
        if np.linalg.norm(I.gradFlam(sol[0], sol[1], lam)) < 1e-13 and np.hypot(*sol) < I.R:
            if not any(np.hypot(sol[0]-p[0], sol[1]-p[1]) < 1e-9 for p in found): found.append((sol[0], sol[1]))
    found.sort()
    pred = [np.sqrt(lam)*zz for zz in (I.zm, I.z0, I.zp)]
    print(f"lambda={lam:.4e}: {len(cand)} grid local minima of |grad| -> {len(found)} distinct zeros in chart ball")
    for (s, t), ps, zz in zip(found, pred, (I.zm, I.z0, I.zp)):
        print(f"   s={s:+.10e} (pred {ps:+.10e}) t={t:+.1e}  F-b={I.Flam(s,t,lam)-I.b:+.10e} (pred {lam**2*I.Phi(zz):+.10e})")
    far = np.ones_like(gn, bool)
    for ps in pred: far &= np.hypot(S-ps, T) > 0.2*np.sqrt(lam)
    print(f"   min |grad| on grid outside 0.2*sqrt(lam) discs around the 3 predicted points: {gn[far].min():.3e}  (min over annulus A: {gn[(RR>=I.r1/2)&(RR<=I.r1)].min():.3e} vs g*/2={I.gstar/2:.3e})")
    if len(found) == 3:
        life = I.Flam(*found[0], lam) - I.Flam(*found[1], lam)
        print(f"   lifetime from located points {life:.10e} vs kappa lam^2 {I.kappa*lam**2:.10e} rel {(life-I.kappa*lam**2)/(I.kappa*lam**2):+.1e}")
for lam in [I.lam0, I.lam0/10, I.lam0/1000]: run(lam)
