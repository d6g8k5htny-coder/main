import json, sys
from fractions import Fraction as Fr
D = '/home/user/wt_c8/frontiers/c8_normalizer_floor_planar_20260930/'
res = json.load(open(D + 'RESULTS.json'))
tpl = open(D + 'NOTE.template.md').read()
def parse(k):
    p = k.replace('r=[', '').replace('],b=[', ',').replace('],k=[', ',').replace(']', '').split(',')
    return [Fr(x) for x in p]
tab = {k: (parse(k), Fr(v['floor'])) for k, v in res['table'].items()}
def dec(x, d=4):
    x = Fr(x); n = x.numerator * 10 ** d // x.denominator; s = str(n).rjust(d + 1, '0'); return s[:-d] + '.' + s[-d:]
def minr(rmax, bk=None):
    best = None
    for k, (p, fl) in tab.items():
        r0, r1, b0, b1, k0, k1 = p
        if r1 <= rmax and (bk is None or (b0 == bk[0] and k0 == bk[1])):
            if best is None or fl < best[0]: best = (fl, k)
    return best
zs = Fr(res['z_star'])
rep = {'__ZSTAR__': dec(zs), '__ARGMIN__': res['argmin'], '__ZR18__': dec(minr(Fr(3,16))[0]), '__ZR14__': dec(minr(Fr(1,4))[0]), '__ZR38__': dec(minr(Fr(3,8))[0]),
       '__EPS__': '%.2e' % res['parameters']['torus_epsilon_at_rmax'], '__EPS0__': '%.2e' % res['parameters']['torus_epsilon0'], '__RUNTIME__': sys.argv[1] if len(sys.argv) > 1 else 'an hour'}
rows = []
for lab, rmax in (('[1/8, 3/16]', Fr(3,16)), ('[1/8, 1/4]', Fr(1,4)), ('[1/8, 3/8]', Fr(3,8)), ('[1/8, 1/2]', Fr(1,2))):
    fl, k = minr(rmax); p = parse(k)
    rows.append('| `%s` | `%s` | `b in [%s, %s], k in [%s, %s]` |' % (lab, dec(fl), p[2], p[3], p[4], p[5]))
rep['__RTABLE__'] = '\n'.join(rows)
ratios = []
ctrl = res['float_controls']
for k, row in ctrl.items():
    c = row['r1,b0,k0']; ratios.append(Fr(res['table'][k]['floor']) / Fr(c['mc']))
rep['__RATIO_MIN__'] = dec(min(ratios), 2); rep['__RATIO_MAX__'] = dec(max(ratios), 2); rep['__NCTRL__'] = str(len(ctrl))
bkrows = []
bs = sorted(set(p[2] for p, _ in tab.values())); ks = sorted(set(p[4] for p, _ in tab.values()))
for b0 in (bs[0], bs[len(bs)//2], bs[-1]):
    for k0 in (ks[0], ks[len(ks)//2], ks[-1]):
        cells = []
        z0 = None
        for rmax in (Fr(1,4), Fr(1,2)):
            fl, key = minr(rmax, (b0, k0))
            mc = ctrl.get(key, {}).get('r1,b0,k0')
            if mc is None:
                import math
                # control not stored for this box: mark
                cells.append('`%s` / n.a.' % dec(fl))
            else:
                cells.append('`%s` / `%.3f` (`%.3f`)' % (dec(fl), mc['mc'], mc['se'])); z0 = mc['z0']
        if z0 is None:
            import math
            from math import erf, exp, sqrt, pi
            s = sqrt(2); m = -float(b0); a = -m / s; Ph = 0.5*(1+erf(a/sqrt(2))); ph = exp(-a*a/2)/sqrt(2*pi)
            z0 = 36 * float(k0) ** 2 * (m*m*Ph - 2*m*s*ph + s*s*(Ph - a*ph))
        bkrows.append('| `b in [%s, %s], k in [%s, %s]` | %s | %s | `%.3f` |' % (b0, b0 + (bs[1]-bs[0]), k0, k0 + (ks[1]-ks[0]), cells[0], cells[1], z0))
rep['__BKTABLE__'] = '\n'.join(bkrows)
for a, b in rep.items():
    assert a in tpl, a
    tpl = tpl.replace(a, b)
assert '__' not in tpl.replace('__init__', ''), [l for l in tpl.splitlines() if '__' in l]
open(D + 'NOTE.md', 'w').write(tpl)
print('filled; z_star', dec(zs), 'ratios', dec(min(ratios), 2), dec(max(ratios), 2))
