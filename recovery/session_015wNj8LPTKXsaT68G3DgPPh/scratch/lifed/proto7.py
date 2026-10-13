import time, sys
from fractions import Fraction as Fr
import lifecore as LC
import tmfx as T
def show(name, t, k=6):
    print('  %-12s c=%s  e=%.3e' % (name, [('%.6g' % float(x)) for x in t.frac_coeffs()[:k]], float(t.frac_e())), flush=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for d, M, names in ((2, LC.E2, ['w']), (3, LC.E3, ['a11', 'a22', 'a12'])):
    t = time.time()
    law = T.tm_law(M, names, 6, N=N)
    print('d=%d  N=%d time %.1fs' % (d, N, time.time() - t), flush=True)
    show('det', law.det); show('e_bb', law.e_bb); show('e_bk', law.e_bk); show('e_kk', law.e_kk)
    for n in names:
        show('alpha_' + n, law.alpha[n]); show('beta_' + n, law.beta[n])
    for a in names:
        for b in names:
            if names.index(a) <= names.index(b):
                show('C_%s%s' % (a, b), law.Cu[(a, b)])
