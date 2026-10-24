# Complete critical list of F_lambda inside the chart ball at lambda=lambda_0 (and smaller), Newton from a dense log-radial seed grid.
import numpy as np, instance as I
from scipy.optimize import fsolve
def run(lam):
    seeds = []
    radii = np.concatenate([np.geomspace(1e-6, I.R, 400)])
    angs = np.linspace(0, 2*np.pi, 181)[:-1]
    for r in radii:
        for a in angs:
            seeds.append((r*np.cos(a), r*np.sin(a)))
    seeds.append((0.0, 0.0))
    seeds = np.array(seeds)
    found = []
    for s0, t0 in seeds:
        sol, info, ier, msg = fsolve(lambda x: I.gradFlam(x[0], x[1], lam), [s0, t0], full_output=True, xtol=1e-14)
        if ier == 1 and np.hypot(*sol) < I.R and np.linalg.norm(I.gradFlam(sol[0], sol[1], lam)) < 1e-12:
            if not any(np.hypot(sol[0]-p[0], sol[1]-p[1]) < 1e-9 for p in found):
                found.append(tuple(sol))
    found.sort()
    print(f"lambda={lam:.4e}: {len(found)} critical points in chart ball")
    pred = [(np.sqrt(lam)*zz, 0.0) for zz in (I.zm, I.z0, I.zp)]
    for (s, t), (ps, pt), zz in zip(found, pred, (I.zm, I.z0, I.zp)):
        Fv = I.Flam(s, t, lam); Fp = I.b + lam**2*I.Phi(zz)
        print(f"   found s={s:+.10e} t={t:+.2e} | predicted s={ps:+.10e} | F={Fv:.17g} predicted={Fp:.17g} | diff={Fv-Fp:+.2e}")
    if len(found) == 3:
        (sm, tm), (s0, t0), (sp, tp) = found
        life = I.Flam(sm, tm, lam) - I.Flam(s0, t0, lam)
        print(f"   lifetime F(M-)-F(S)={life:.10e}  kappa*lam^2={I.kappa*lam**2:.10e}  rel err={(life - I.kappa*lam**2)/(I.kappa*lam**2):+.2e}")
        print(f"   F(M+)-F(M-)={I.Flam(sp,tp,lam)-I.Flam(sm,tm,lam):.3e} >0: {I.Flam(sp,tp,lam) > I.Flam(sm,tm,lam)}")
    return found
for lam in [I.lam0, I.lam0/10, I.lam0/1000]:
    run(lam)
# also check lambda=0: only the origin
f0 = run(0.0)
