import json, math, sys, pathlib
HERE = pathlib.Path('/home/user/wt_tail/frontiers/c6_microscopic_tail_numerics_20260930')
res = json.load(open(HERE / 'RESULTS.json'))
g = lambda v, n=5: ('%.' + str(n) + 'g') % v
T = (HERE / 'NOTE_template.md').read_text()
cusp_rows = []; devmax = 0.0
for k in ('0.5', '1.0', '2.0'):
    c = res['per_k'][k]['cusp']; d = res['per_k'][k]['cusp_gl48_rel_dev']
    devmax = max(devmax, max(d.values()))
    cusp_rows.append('| `%s` | %s | %s | %s | %s | `%.0e` |' % (k, g(c['J_cusp'], 6), g(c['T1'], 6), g(c['T2'], 6), g(c['Tb'], 6), max(d.values())))
T = T.replace('{{cusp_rows}}', '\n'.join(cusp_rows)).replace('{{cusp_dev_max}}', '%.0e' % devmax)
rows = []
for b in ('0.0', '1.0'):
    for k in ('0.5', '1.0', '2.0'):
        t = res['table']['k=%s,b=%s' % (k, b)]
        rows.append('| `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |' % (k, b, g(t['C_star']), g(t['G_max_coeff']), g(t['G2_coeff']), g(t['B2_coeff']), g(t['singleton_coeff']), g(t['C2_over_C0']), g(t['tv_coefficient_kappa_c'], 3), g(t['B_sign'], 4)))
T = T.replace('{{table_rows}}', '\n'.join(rows))
nm_rows = []; moves = []
for k in ('0.5', '1.0', '2.0'):
    n = res['near_mass_identity']['k=%s' % k]; lv = n['K0_integral_without_prefactor_by_level']
    l32, l64, l96 = lv['32x32x64x32'], lv['64x64x64x32'], lv['96x96x64x32']
    moves.append(abs(l96 - l64) / l96)
    nm_rows.append('| `%s` | %s / %s / %s | %s | %s | %s (%s) | %s / %s |' % (k, g(l32, 6), g(l64, 6), g(l96, 6), g(n['alpha1_plus_2alpha2_b0'], 5), g(n['ref_gh60_b0'], 5), g(n['ref_mc_b0'], 5), g(n['ref_mc_b0_se_alpha1'], 2), '%+.2e' % n['rel_dev_vs_gh60'], '%+.2e' % n['rel_dev_vs_mc']))
T = T.replace('{{nm_rows}}', '\n'.join(nm_rows)).replace('{{nm_move}}', 'at most `%.0e`' % max(moves))
T = T.replace('{{minutes}}', sys.argv[1] if len(sys.argv) > 1 else '5')
# reading-sentence numbers at k = 1, b = 0
t = res['table']['k=1.0,b=0.0']
T = T.replace('`C_*\n= 52.6`', '`C_* = %s`' % g(t['C_star'], 3))
assert '{{' not in T, [l for l in T.splitlines() if '{{' in l]
(HERE / 'NOTE.md').write_text(T)
print('NOTE.md written')
