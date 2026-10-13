from fractions import Fraction as Fr
import lifecore as LC
M=LC.E3
ctx = LC.context(M, Fr(0), Fr(1,512), 3)
law=ctx.law
bb=(Fr(0),Fr(1,2)); kb=(Fr(1,2),Fr(1))
b=M.IV(*bb); k=M.IV(*kb)
mu = {n: b * law.alpha[n] + k * law.beta[n] for n in M.TARGET_NAMES}
for n in ('T','tau','o11','o22','o12','v1','v2','nu1','nu2','a11','a22','a12'):
    print(n, 'mean [%.4f,%.4f]'%(float(mu[n].lo),float(mu[n].hi)), 'sdQ %.4f'%float(law.sdQ[n]), 'alpha [%.4f,%.4f] beta [%.4f,%.4f]'%(float(law.alpha[n].lo),float(law.alpha[n].hi),float(law.beta[n].lo),float(law.beta[n].hi)))
lo,hi,info=LC.zeta_ratio_d3(ctx,bb,kb)
print({k:(float(v) if not isinstance(v,tuple) else tuple(float(x) for x in v)) for k,v in info.items()})
