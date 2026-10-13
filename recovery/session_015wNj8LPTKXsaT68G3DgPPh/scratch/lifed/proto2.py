import time
from fractions import Fraction as Fr
import lifecore as LC
for d, M in ((2, LC.E2), (3, LC.E3)):
    for rs in (Fr(1,4096), Fr(1,512), Fr(1,64)):
        t=time.time(); ctx = LC.context(M, Fr(0), rs, d); tc=time.time()-t
        print('d=%d r_*=%s ctx %.1fs detS=[%.10f,%.10f] detfac=[%.8f,%.8f]'%(d, rs, tc, float(ctx.detS.lo), float(ctx.detS.hi), float(ctx.detfac.lo), float(ctx.detfac.hi)))
        worst_hi, worst_lo = 0, 9
        t=time.time()
        for b0 in (Fr(0), Fr(1,2)):
            for k0 in (Fr(1,2), Fr(1), Fr(3,2)):
                bb=(b0, b0+Fr(1,2)); kb=(k0, k0+Fr(1,2))
                lo, hi, pr, info = LC.rho_box(ctx, bb, kb)
                print('   b[%s,%s] k[%s,%s]  rho in [%.6f, %.6f]  pi[%.6f,%.6f] rat[%.6f,%.6f] cup %.2e clo %.2e'%(bb[0],bb[1],kb[0],kb[1],float(lo),float(hi),float(pr.lo),float(pr.hi),float(info['rat'][0]),float(info['rat'][1]),float(info['corr_up']),float(info['corr_lo'])))
        print('   sweep %.1fs'%(time.time()-t))
