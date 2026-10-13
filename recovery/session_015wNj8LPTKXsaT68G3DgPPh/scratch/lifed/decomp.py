import sys
from fractions import Fraction as Fr
sys.path.insert(0, '/home/user/wt_lifed/frontiers/c8_lifetime_difference_constant_20261001')
import theorem_b as TB
cap = {}
def prof(frame, event, arg):
    if event == 'return' and frame.f_code.co_name in ('bounds_d3', 'bounds_d2'):
        cap.update(frame.f_locals)
R = Fr(1, 4096)
for d, fn in ((3, TB.bounds_d3), (2, TB.bounds_d2)):
    tml = TB.tm_law(d)
    il = TB.ENG[d].band_law(Fr(0), R)
    for Q in ((Fr(0), Fr(1, 8), Fr(1, 2), Fr(5, 8)), (Fr(0), Fr(1, 8), Fr(15, 8), Fr(2))):
        cap.clear()
        sys.setprofile(prof)
        up, dn = fn(Q, R, tml, il)
        sys.setprofile(None)
        print('d', d, 'box', [str(x) for x in Q], 'up a=%.4f' % float(up[0]), 'dn a=%.4f' % float(dn[0]))
        n36 = cap.get('norm36')
        for k in ('U2_up', 'U2_dn', 'U3_up', 'U3_dn', 'U3r', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'bad', 'U4rel', 'U67rel', 'T3_up', 'T3_dn', 'X', 'z_up', 'z_dn', 'p_up', 'p_dn'):
            if k in cap:
                v = cap[k]
                if isinstance(v, tuple):
                    print('   %-7s (a=%.4f, e=%.3g)' % (k, float(v[0]), float(v[1])))
                else:
                    extra = ' /norm36=%.4f' % float(v / n36) if (n36 and k in ('U3r','U4','U5','U6','U7','U8','U9','U10','bad')) else ''
                    print('   %-7s %.5g%s' % (k, float(v), extra))
