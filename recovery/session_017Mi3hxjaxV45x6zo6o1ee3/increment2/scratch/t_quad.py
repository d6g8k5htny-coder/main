import sys, time
sys.path.insert(0, '.')
import certified_d2 as c
from fractions import Fraction as Q
ref = {'3': Q('4.9879432666614182044e-2'), '4': Q('6.7568657600538789458e-2'), '2': Q('1.26175931306921059022772e-2')}
pref = c.prefactor()
for name in sys.argv[1:]:
    L = c.I(int(name))
    t = time.time()
    k = c.cumulants(*c.moments_image(L))
    J, info = c.quadrature(k)
    cc = pref * J
    print(name, 'c in [%s, %s]' % (format(float(cc.lo), '.20e'), format(float(cc.hi), '.20e')),
          'width %.2e' % float(cc.hi - cc.lo), 'a', info['a'], 'N', info['N'], 'M %.3e' % float(info['M']),
          'contains #297 value', cc.lo <= ref[name] <= cc.hi, 'diff %.2e' % float(abs((cc.lo+cc.hi)/2 - ref[name])), '%.1fs' % (time.time() - t), flush=True)
