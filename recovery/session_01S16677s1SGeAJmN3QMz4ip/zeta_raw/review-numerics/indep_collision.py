"""Independent (composite Gauss-Legendre, claims/dbn_common.py class, different nodes/precision)
evaluation of H, H_z, H_zz at the published double zeros; Newton-step size gives the
discrepancy in (t*, x0) between the two quadratures."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, '/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/claims')
import dbn_common as dc
from mpmath import mp
mp.dps = 90
pts = {'pair34_35': ('-0.384561116627981414864807288345216832815873886', '223.275780185615519913542719804846971188898645'),
       'pair9_10': ('-1.87790907297253526476683971812792711055587932', '99.3284693055757509480227688859994916526075262')}
for (pieces, deg) in [(80, 4), (120, 4)]:
    t0 = time.time()
    hf = dc.HFlow(U=1.4, pieces=pieces, degree=deg)
    for tag, (ts, xs) in pts.items():
        t, x = mp.mpf(ts), mp.mpf(xs)
        H, Hz, Hzz, Hzzz = hf.H(t, x), hf.Hz(t, x), hf.Hzz(t, x), hf.Hzzz(t, x)
        # 2-D Newton step on (H, H_z) with H_t = -H_zz, H_zt = -H_zzz
        J = mp.matrix([[Hz, -Hzz], [Hzz, -Hzzz]])
        d = mp.lu_solve(J, mp.matrix([-H, -Hz]))
        print('nodes=%d %s: H/|A|=%s Hz/|A|=%s  A=%s  -> dx=%s dt=%s' % (len(hf.u), tag, mp.nstr(H / abs(Hzz / 2), 3), mp.nstr(Hz / abs(Hzz / 2), 3),
              mp.nstr(Hzz / 2, 15), mp.nstr(d[0], 3), mp.nstr(d[1], 3)))
    print('  (%.0f s)' % (time.time() - t0))
