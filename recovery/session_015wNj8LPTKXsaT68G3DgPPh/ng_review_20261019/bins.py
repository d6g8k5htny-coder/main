# Independent lifetime bins (t_a, t_b]: observed count / predicted count R*pred*(t_b^(2/3)-t_a^(2/3)), by (model, eps, J).
# No grid-loss correction here (see pool.py); the loss is concentrated in the smallest bins.
import json, sys, os, math
d = sys.argv[1] if len(sys.argv) > 1 else '.'
pred6 = {float(e): json.load(open(os.path.join(d, 'kappa_eps%d.json' % e)))['pred_coeff'] for e in (0, 1, 2, 3)}
predG = {1.0: 4.12499, 2.0: 2.89599, 3.0: 2.26250}
pool = {}
for f in sorted(os.listdir(d)):
    if not f.endswith('.json') or f.startswith(('kappa', 'kk_', 'gridloss')): continue
    try: r = json.load(open(os.path.join(d, f)))
    except Exception: continue
    if 'counts' not in r: continue
    m = r.get('model', '')
    model = 'gm' if m == 'cov-matched Gaussian' else ('cubic' if m.startswith('cubic') else 'ng')
    key = (model, float(r['eps']), r.get('J', 4096)); R = r.get('realisations', r.get('R'))
    ts, cs = [0.0] + list(r['ts']), [0] + list(r['counts'])
    for i in range(1, len(ts)):
        b = pool.setdefault(key, {}).setdefault((ts[i - 1], ts[i]), [0, 0.0])
        b[0] += cs[i] - cs[i - 1]; b[1] += R * (ts[i] ** (2 / 3) - ts[i - 1] ** (2 / 3))
for key in sorted(pool, key=lambda k: (['ng', 'cubic', 'gm'].index(k[0]), k[1], k[2])):
    model, eps, J = key; p = predG[eps] if model == 'gm' else pred6[eps]
    print('\n%s eps=%g J=%d pred %.4f' % (model, eps, J, p))
    for (a, b), (c, w) in sorted(pool[key].items()):
        e = p * w; print('   (%-6g, %-6g]  obs %-7d exp %-9.1f ratio %.4f +- %.4f' % (a, b, c, e, c / e, math.sqrt(c) / e))
