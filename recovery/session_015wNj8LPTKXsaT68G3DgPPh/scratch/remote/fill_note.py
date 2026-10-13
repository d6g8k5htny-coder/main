import json, math, sys
HERE = '/home/user/Math-/frontiers/c6_remote_coefficient_numerics_20260930'
res = json.load(open(HERE + '/RESULTS.json'))
alpha = {'k=0.5,b=0.0': (1.11821, 0.0347454), 'k=1.0,b=0.0': (1.32906, 0.0255091), 'k=2.0,b=0.0': (2.1621, 0.0169281),
         'k=0.5,b=1.0': (0.320155, 0.00994791), 'k=1.0,b=1.0': (0.380523, 0.00730347), 'k=2.0,b=1.0': (0.619027, 0.00484668)}
def g(v, n=6):
    return ('%.' + str(n) + 'g') % v
T = open(HERE + '/NOTE_template.md').read()
U = res['controls']['unconditional_hessian']
T = T.replace('{{absdet_err}}', '%.1e' % abs(U['E_absdet'] - 4 / math.sqrt(3)))
gc = res['grid_convergence']['k=1.0,b=0.0']
T = T.replace('{{absdet_err_hi}}', '%.1e' % abs(gc['lambda_inf_high_order'] - res['far_field']['0.0']['lambda_inf']) + ' relative on Lambda_inf(0) between orders')
T = T.replace('{{exact_count}}', str(res['exact_regression_points']))
far_rows = []
for b in ('0.0', '1.0'):
    f = res['far_field'][b]
    far_rows.append('| `%s` | %s | %s | %s | %s | %s |' % (b, g(f['lambda_inf']), g(f['lambda_inf_j'][0]), g(f['lambda_inf_j'][1]), g(f['lambda_inf_j'][2]), g(f['p_inf'])))
T = T.replace('{{far_rows}}', '\n'.join(far_rows))
T = T.replace('{{EdetH0}}', g(sum(res['far_field']['0.0']['E_F_j_given_level'])))
hole_rows = []
total_rows = []
for b in ('0.0', '1.0'):
    for k in ('0.5', '1.0', '2.0'):
        key = 'k=%s,b=%s' % (k, b)
        H = res['hole'][key]
        li = res['far_field'][b]['lambda_inf']
        hole_rows.append('| `%s` | `%s` | %s | %s | %s | %s | %s |' % (k, b, g(H['hole']), g(H['hole_j'][0]), g(H['hole_j'][1]), g(H['hole_j'][2]), g(H['hole'] / li, 4)))
        for L in ('6', '12', '24'):
            rt = res['remote_total'][key]['L=' + L]
            integ = rt['integral_X_Lambda_periodized_direct'] if L == '6' else rt['integral_X_Lambda_approx']
            R = float(k) * integ
            a1, a2 = alpha[key]
            nu1 = a1 + R
            total_rows.append('| `%s` | `%s` | `%s`%s | %s | %s | %s | %s | %s | %s |' % (k, b, L, ' (direct)' if L == '6' else '', g(integ), g(R), g(a1), g(nu1), g(a2), g(a2 / nu1, 3)))
T = T.replace('{{hole_rows}}', '\n'.join(hole_rows))
T = T.replace('{{total_rows}}', '\n'.join(total_rows))
nz = res['near_zero_profile']['k=1.0,b=0.0']
def nzv(r, ang):
    for row in nz:
        if row['r'] == r and row['angle_deg'] == ang:
            return '%.3g' % row['lambda']
T = T.replace('{{near05}}', nzv(0.05, 60)).replace('{{near04}}', nzv(0.4, 90))
T = T.replace('Lambda(0.05, 45 deg)', 'Lambda(0.05, 60 deg)')
devs = []
for key in res['remote_total']:
    rt = res['remote_total'][key]['L=6']
    devs.append(abs(rt['integral_X_Lambda_periodized_direct'] - rt['integral_X_Lambda_approx']) / rt['integral_X_Lambda_periodized_direct'])
T = T.replace('{{L6_dev}}', 'between %.1e and %.1e' % (min(devs), max(devs)))
T = T.replace('{{L12_direct}}', g(gc['torus_L=12_direct_h=0.2'])).replace('{{L12_approx}}', g(gc['torus_L=12_approx']))
T = T.replace('{{L12_rel}}', '%.1e' % (abs(gc['torus_L=12_direct_h=0.2'] - gc['torus_L=12_approx']) / gc['torus_L=12_direct_h=0.2']))
key = 'k=1.0,b=0.0'
R24 = res['remote_total'][key]['L=24']['k_integral']
a1, a2 = alpha[key]
T = T.replace('{{share24}}', '%.2e' % (a2 / (a1 + R24 + a2)))
T = T.replace('{{qdev}}', '%.1e' % res['quadrature']['max_rel_dev_vs_high_order_at_samples'])
T = T.replace('{{hole_h03}}', g(gc['hole_h=0.3,Rc=6'], 7)).replace('{{hole_h015}}', g(gc['hole_h=0.15,Rc=6'], 7)).replace('{{hole_h01}}', g(gc['hole_h=0.1,Rc=6'], 7)).replace('{{hole_h03_hi}}', g(gc['hole_h=0.3,Rc=6,high_order'], 7))
T = T.replace('{{tail_dev}}', '%.1e' % abs(gc['hole_h=0.3,Rc=5'] - gc['hole_h=0.3,Rc=6']))
import math as _m
def _fc(b):
    Phi=lambda t: 0.5*_m.erfc(-t/_m.sqrt(2)); phi=lambda t: _m.exp(-t*t/2)/_m.sqrt(2*_m.pi)
    e=_m.exp(-b*b/4); p=phi(b)/(2*_m.pi)
    return p*(b*b-1+2*_m.sqrt(2)*e)
fce = max(abs(res['far_field'][b]['lambda_inf'] - _fc(float(b))) for b in ('0.0','1.0'))
T = T.replace('{{far_closed_err}}', '%.0e' % fce)
T = T.replace('{{hole_order_dev}}', '%.0e' % abs(gc['hole_h=0.3,Rc=6'] - gc['hole_h=0.3,Rc=6,high_order']))
mc = res['controls']['mc_absdet_unconditional']
T = T.replace('{{mc}}', '%.4f +- %.4f' % (mc['mean'], mc['se'])).replace('{{mc_se}}', '%.1f' % ((4/_m.sqrt(3) - mc['mean'])/mc['se']))
T = T.replace('{{minutes}}', sys.argv[1] if len(sys.argv) > 1 else '45')
assert '{{' not in T, [l for l in T.splitlines() if '{{' in l]
open(HERE + '/NOTE.md', 'w').write(T)
print('NOTE.md written', len(T))
