import time, sys
from fractions import Fraction as Fr
import refint as RI
d = int(sys.argv[1]); nh = int(sys.argv[2])
M = RI.E2 if d == 2 else RI.E3
t = time.time()
cells = RI.b_cells(d, nh=nh, block=16)
tb = time.time() - t
Ib = RI.b_integral(M, cells, nh, 0, 1)
t = time.time()
J43 = RI.k_integral(M, 4, 3, Fr(1, 2), Fr(2))
J23 = RI.k_integral(M, 2, 3, Fr(1, 2), Fr(2))
J13 = RI.k_integral(M, 1, 3, Fr(1, 2), Fr(2))
tk = time.time() - t
K = RI.K_const(M, d)
c = K * Ib * J43
print('d=%d nh=%d  b-cells %.1fs k %.1fs' % (d, nh, tb, tk))
print('  I_b = [%.15f, %.15f] (rel width %.2e)' % (float(Ib.lo), float(Ib.hi), float((Ib.hi - Ib.lo) / Ib.lo)))
print('  J_4/3 = [%.15e, %.15e]' % (float(J43.lo), float(J43.hi)))
print('  c_ref = [%.16f, %.16f]' % (float(c.lo), float(c.hi)))
print('  c\' (k^-4/3 weight, a = 1) = [%.10e, %.10e]' % (float((K * Ib * J23).lo), float((K * Ib * J23).hi)))
print('  k^-5/3 weight integral = [%.10e, %.10e]' % (float((K * Ib * J13).lo), float((K * Ib * J13).hi)))
