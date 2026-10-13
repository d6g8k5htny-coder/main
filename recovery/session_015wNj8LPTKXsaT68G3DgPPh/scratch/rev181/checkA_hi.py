import sys, time, math
sys.argv=['x']
exec(open('checkA.py').read().split('# ---- 5.')[0].split('# ---- 3.')[0].replace("print(", "(lambda *a, **k: None)("))
exec('# ---- 5.' + open('checkA.py').read().split('# ---- 5.')[1].split('for k in (0.5')[0])
import tail_constants as tc
for k in (1.0,):
    a1, a2 = tc.ALPHA_GH60[k]; m1, m2 = tc.ALPHA_MC[k]
    for (nu, nv) in ((64, 64), (96, 96)):
        t0 = time.time()
        mS = mass(k, 'S', nu, nv); mE = mass(k, 'E', nu, nv); mO = mass(k, 'O2', nu, nv); mI = mass(k, 'S2_inner', nu, nv)
        print(f'k={k} {nu}x{nv}: S={mS:.6f} (a1+2a2 {a1+2*a2:.6f} rel {abs(mS-(a1+2*a2))/(a1+2*a2):.1e}) E={mE:.6f} (a1+a2 {a1+a2:.6f} rel {abs(mE-(a1+a2))/(a1+a2):.1e}) O2={mO:.6f} inner={mI:.6f} (a2 {a2:.6f}, MC {m2:.6f}; rel {abs(mO-a2)/a2:.1e},{abs(mI-a2)/a2:.1e}; O2-inner {mO-mI:.1e}) E-inner={mE-mI:.6f} (a1 {a1:.6f} rel {abs(mE-mI-a1)/a1:.1e}) S-E-O2={mS-mE-mO:.1e} [{time.time()-t0:.0f}s]', flush=True)
