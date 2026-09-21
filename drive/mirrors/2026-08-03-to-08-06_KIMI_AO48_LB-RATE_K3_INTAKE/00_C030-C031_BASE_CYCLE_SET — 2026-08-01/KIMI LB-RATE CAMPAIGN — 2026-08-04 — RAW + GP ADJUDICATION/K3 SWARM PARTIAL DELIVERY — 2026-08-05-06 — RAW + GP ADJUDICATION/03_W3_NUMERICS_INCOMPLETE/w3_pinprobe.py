# Near-pin scaling probe (float): decides pin-chart design.
import mpmath as mp
import importlib.util

mp.mp.prec = 200
spec = importlib.util.spec_from_file_location("w3p", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_proto.py")
w3p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w3p)
spec2 = importlib.util.spec_from_file_location("w3s", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_scan.py")
w3s = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(w3s)

b_mp, ell = w3p.b_mp, w3p.ell
Mpt = (mp.mpf(-1)/80, mp.mpf(0))

# jet up to order 3 at a point, conditional law
JET3 = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
def jetlaw3(y):
    B = mp.matrix(10, 10)
    for i, ai in enumerate(JET3):
        for j, aj in enumerate(JET3):
            B[i, j] = w3p.cov2(ai, aj, y, y)
    A = mp.matrix(10, 9)
    for i, ai in enumerate(JET3):
        for j, (Pj, aj) in enumerate(w3p.LFUNS):
            A[i, j] = w3p.cov2(ai, aj, y, (w3p.mqf(Pj[0]), w3p.mqf(Pj[1])))
    At = A * mp.matrix(w3p.Cmp).T
    muJ = At * w3p.lam
    SigJ = B - At * w3p.Gi * At.T
    return muJ, SigJ

muM, SigM = jetlaw3(Mpt)
print("mean jet at M: f=%s grad=(%s,%s)" % tuple(mp.nstr(v,8) for v in (muM[0],muM[1],muM[2])))
print("mean Hessian at M: H11=%s H12=%s H22=%s" % tuple(mp.nstr(muM[k],10) for k in (3,4,5)))
print("  eigen-scale: tr=%s det=%s" % (mp.nstr(muM[3]+muM[5],10), mp.nstr(muM[3]*muM[5]-muM[4]**2,10)))
print("mean 3rd jet at M: m30=%s m21=%s m12=%s m03=%s" % tuple(mp.nstr(muM[k],10) for k in (6,7,8,9)))
print("cond sd of Hessian at M: (%s,%s,%s)" % tuple(mp.nstr(mp.sqrt(SigM[k,k]),10) for k in (3,4,5)))
print("cond sd of 3rd jet at M: (%s,%s,%s,%s)" % tuple(mp.nstr(mp.sqrt(SigM[k,k]),10) for k in (6,7,8,9)))

def rho_at(y):
    rv, info = w3s.rho(y)
    return rv, info

print("\nNear-pin rho probe (direction +x from M):")
for s in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
    y = (Mpt[0] + mp.mpf(s), Mpt[1])
    rv, info = rho_at(y)
    print("  s=%.4f rho=%s  s*rho=%s  pgrad=%s z=%s g=%s" %
          (s, mp.nstr(rv,6), mp.nstr(rv*s,6), mp.nstr(info['pgrad'],6),
           mp.nstr(info['z'],6), mp.nstr(info['g'],6)))
print("Near-pin rho probe (direction +y from M):")
for s in [0.05, 0.02, 0.01, 0.005]:
    y = (Mpt[0], Mpt[1] + mp.mpf(s))
    rv, info = rho_at(y)
    print("  s=%.4f rho=%s  s*rho=%s  pgrad=%s z=%s g=%s" %
          (s, mp.nstr(rv,6), mp.nstr(rv*s,6), mp.nstr(info['pgrad'],6),
           mp.nstr(info['z'],6), mp.nstr(info['g'],6)))
print("Near-pin rho probe (direction diag (1,1)/sqrt2 from M):")
for s in [0.02, 0.01, 0.005]:
    d = mp.mpf(s)/mp.sqrt(2)
    y = (Mpt[0] + d, Mpt[1] + d)
    rv, info = rho_at(y)
    print("  s=%.4f rho=%s  s*rho=%s" % (s, mp.nstr(rv,6), mp.nstr(rv*s,6)))
