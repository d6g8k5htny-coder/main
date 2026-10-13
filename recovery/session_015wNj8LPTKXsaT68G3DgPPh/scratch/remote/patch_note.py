"""Patch NOTE.md numbers from the regenerated RESULTS.json (only fields that the Codex fixes can change)."""
import json, re, pathlib
HERE = pathlib.Path('/home/user/Math-/frontiers/c6_remote_coefficient_numerics_20260930')
old = json.load(open('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/remote/RESULTS_ad7c488.json'))
new = json.load(open(HERE / 'RESULTS.json'))
alpha = {'k=0.5,b=0.0': (1.11821, 0.0347454), 'k=1.0,b=0.0': (1.32906, 0.0255091), 'k=2.0,b=0.0': (2.1621, 0.0169281),
         'k=0.5,b=1.0': (0.320155, 0.00994791), 'k=1.0,b=1.0': (0.380523, 0.00730347), 'k=2.0,b=1.0': (0.619027, 0.00484668)}
g = lambda v, n=6: ('%.' + str(n) + 'g') % v
T = (HERE / 'NOTE.md').read_text()
changes = []
def rep(a, b):
    global T
    assert a in T, a
    T = T.replace(a, b); changes.append((a[:60], b[:60]))
# far field unchanged?
for b in ('0.0', '1.0'):
    assert abs(old['far_field'][b]['lambda_inf'] - new['far_field'][b]['lambda_inf']) < 1e-12
# hole rows
for b in ('0.0', '1.0'):
    for k in ('0.5', '1.0', '2.0'):
        key = 'k=%s,b=%s' % (k, b)
        Ho, Hn = old['hole'][key], new['hole'][key]
        li = new['far_field'][b]['lambda_inf']
        row_old = '| `%s` | `%s` | %s | %s | %s | %s | %s |' % (k, b, g(Ho['hole']), g(Ho['hole_j'][0]), g(Ho['hole_j'][1]), g(Ho['hole_j'][2]), g(Ho['hole'] / li, 4))
        row_new = '| `%s` | `%s` | %s | %s | %s | %s | %s |' % (k, b, g(Hn['hole']), g(Hn['hole_j'][0]), g(Hn['hole_j'][1]), g(Hn['hole_j'][2]), g(Hn['hole'] / li, 4))
        if row_old != row_new: rep(row_old, row_new)
        a1, a2 = alpha[key]
        for L in ('6', '12', '24'):
            ro, rn = old['remote_total'][key]['L=' + L], new['remote_total'][key]['L=' + L]
            io = ro['integral_X_Lambda_periodized_direct'] if L == '6' else ro['integral_X_Lambda_approx']
            inn = rn['integral_X_Lambda_periodized_direct'] if L == '6' else rn['integral_X_Lambda_approx']
            mk = lambda integ: '| `%s` | `%s` | `%s`%s | %s | %s | %s | %s | %s | %s |' % (k, b, L, ' (direct)' if L == '6' else '', g(integ), g(float(k) * integ), g(a1), g(a1 + float(k) * integ), g(a2), g(a2 / (a1 + float(k) * integ), 3))
            if mk(io) != mk(inn): rep(mk(io), mk(inn))
# L6 deviation range
def devs(res):
    out = []
    for key in res['remote_total']:
        rt = res['remote_total'][key]['L=6']
        out.append(abs(rt['integral_X_Lambda_periodized_direct'] - rt['integral_X_Lambda_approx']) / rt['integral_X_Lambda_periodized_direct'])
    return min(out), max(out)
do, dn = devs(old), devs(new)
rep('they differ by between `%.1e` and `%.1e` relative' % do, 'they differ by between `%.1e` and `%.1e` relative' % dn)
# L12 direct
gco, gcn = old['grid_convergence']['k=1.0,b=0.0'], new['grid_convergence']['k=1.0,b=0.0']
rep('For `L = 12` the direct integral at `k = 1, b = 0` is `%s`\nagainst `%s` from the far field plus hole, so the additive form is accurate to `2e-5` there' % (g(gco['torus_L=12_direct_h=0.2']), g(gco['torus_L=12_approx'])),
    'For `L = 12` the direct integral at `k = 1, b = 0` is `%s`\nagainst `%s` from the far field plus hole, so the additive form is accurate to `%.0e` there' % (g(gcn['torus_L=12_direct_h=0.2']), g(gcn['torus_L=12_approx']), abs(gcn['torus_L=12_direct_h=0.2'] - gcn['torus_L=12_approx']) / gcn['torus_L=12_direct_h=0.2']))
rep('agrees with far field plus hole to `%.1e` relative.' % (abs(gco['torus_L=12_direct_h=0.2'] - gco['torus_L=12_approx']) / gco['torus_L=12_direct_h=0.2']),
    'agrees with far field plus hole to `%.1e` relative.' % (abs(gcn['torus_L=12_direct_h=0.2'] - gcn['torus_L=12_approx']) / gcn['torus_L=12_direct_h=0.2']))
# grid convergence numbers
for key in ('hole_h=0.3,Rc=6', 'hole_h=0.15,Rc=6', 'hole_h=0.1,Rc=6', 'hole_h=0.3,Rc=6,high_order'):
    if g(gco[key], 7) != g(gcn[key], 7): rep('`%s`' % g(gco[key], 7), '`%s`' % g(gcn[key], 7))
# near-zero mentions
def nz(res, r, ang):
    for row in res['near_zero_profile']['k=1.0,b=0.0']:
        if row['r'] == r and row['angle_deg'] == ang: return row['lambda']
for (r, ang, label) in ((0.05, 60, 'Lambda(0.05, 60 deg)'), (0.4, 90, 'Lambda(0.4, 90 deg)')):
    a, b = '%.3g' % nz(old, r, ang), '%.3g' % nz(new, r, ang)
    if a != b: rep('`%s = %s`' % (label, a), '`%s = %s`' % (label, b))
rep('{{exact_count}}', str(new['exact_regression_points']))
# quadrature dev and hole order dev
if '%.1e' % old['quadrature']['max_rel_dev_vs_high_order_at_samples'] != '%.1e' % new['quadrature']['max_rel_dev_vs_high_order_at_samples']:
    rep('`%.1e` at points where `Lambda < 10^(-16) Lambda_inf`' % old['quadrature']['max_rel_dev_vs_high_order_at_samples'], '`%.1e` at points where `Lambda < 10^(-16) Lambda_inf`' % new['quadrature']['max_rel_dev_vs_high_order_at_samples'])
(HERE / 'NOTE.md').write_text(T)
print('patched NOTE.md; changes:'); [print(' ', c) for c in changes]
assert '{{' not in T
