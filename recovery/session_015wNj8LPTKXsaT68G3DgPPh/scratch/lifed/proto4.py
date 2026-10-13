from fractions import Fraction as Fr
import lifecore as LC
M=LC.E2
for (r0,r1) in ((Fr(1,1024),Fr(1,512)),(Fr(7,4096),Fr(1,512)),(Fr(0),Fr(1,512))):
    law=M.band_law(r0,r1)
    f=float
    print('band',r0,r1)
    for n in ('w','om','v','nu','T','tau'):
        print('  %s alpha [%.6e,%.6e] beta [%.6e,%.6e] bw %s'%(n,f(law.alpha[n].lo),f(law.alpha[n].hi),f(law.beta[n].lo),f(law.beta[n].hi), ('[%.3e,%.3e]'%(f(law.bw[n].lo),f(law.bw[n].hi)) if n in law.bw else '-')))
    print('  var w [%.8f,%.8f]  Cg om %.4f v %.4f'%(f(law.C[('w','w')].lo),f(law.C[('w','w')].hi), f(law.Cg[('om','om')].hi), f(law.Cg[('v','v')].hi)))
