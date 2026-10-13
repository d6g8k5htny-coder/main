import sys, json
from fractions import Fraction as Fr
sys.path.insert(0, '/home/user/wt_lifed/frontiers/c8_lifetime_difference_constant_20261001')
import theorem_b as TB
res = json.load(open('/home/user/wt_lifed/frontiers/c8_lifetime_difference_constant_20261001/RESULTS.json'))
eng = {2: json.load(open('results_planar.json')), 3: json.load(open('results_d3.json'))}
for d in (2, 3):
    E = TB.ENG[d]
    cells = TB.b_cells(d)
    K = TB.K_const(d)
    ws = []
    for (b0, b1), (k0, k1) in E.BOXES:
        ws.append((K * TB.b_integral(E, cells, b0, b1) * TB.k_integral(E, 1, 3, k0, k1)).hi)
    tot_native = sum(ws)
    tot_ref = sum(Fr(x) for x in res['dims'][str(d)]['reference']['w_cap'])
    print('d', d, 'native boxes', len(ws), 'sum w native %.6e' % float(tot_native), 'sum w refined %.6e' % float(tot_ref))
    for R in TB.R_STARS:
        rr = res['dims'][str(d)]['rate'][str(R)]
        fac = 1 + Fr(rr['a_up']) * R * R + Fr(rr['eps_up'])
        Cb = [Fr(0)] * len(ws)
        for rec in eng[d]['bands']:
            if Fr(rec['r'][1]) <= R:
                for i, x in enumerate(rec['bound']):
                    Cb[i] = max(Cb[i], Fr(x))
        coarse = fac * sum(c * w for c, w in zip(Cb, ws))
        ref = Fr(res['dims'][str(d)]['table'][str(R)]['C_BK'])
        print('  R', R, 'native %.4f' % float(coarse), 'refined %.4f' % float(ref), 'factor %.3f' % float(coarse / ref))
