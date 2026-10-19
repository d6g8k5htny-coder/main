# Pool the Monte Carlo runs by (model, eps, J) and compare with the predictions.
# rate(t) = count / (R t^(2/3)); SE from Poisson counts (short bars are rare per realisation).
# Models: 'ng'    exact short superlevel-H0 bars of F=G+eps(H^2-1)          (simulate*.py)
#         'gm'    exact short bars of the Gaussian with F's covariance       (gauss_matched.py)
#         'cubic' cubic-jet pair heights at inflection points of F           (inflect.py; no grid loss, no Taylor remainder)
# Predictions: formula (6) via the literal mixture kernel (kappa.py) for 'ng' and 'cubic'; the exact matched-Gaussian
# value for 'gm'. 'loss' is the expected grid-detection loss from gridloss.py (bar models only); 'corr' = rate/(1-loss).
import json, sys, os, math
d = sys.argv[1] if len(sys.argv) > 1 else '.'
pred6 = {float(e): json.load(open(os.path.join(d, 'kappa_eps%d.json' % e)))['pred_coeff'] for e in (0, 1, 2, 3)}
predG = {1.0: 4.12499, 2.0: 2.89599, 3.0: 2.26250}   # exact spectral-moment values (gauss_matched.py)
gl = json.load(open(os.path.join(d, 'gridloss.json')))
def loss(model, eps, J, t):
    k = '%s eps=%g J=%d' % (model, eps, J)
    if model == 'cubic' or k not in gl['loss']: return 0.0
    return gl['loss'][k][gl['ts'].index(t)]
runs = {}
for f in sorted(os.listdir(d)):
    if not f.endswith('.json') or f.startswith(('kappa', 'kk_', 'gridloss')): continue
    try: r = json.load(open(os.path.join(d, f)))
    except Exception: print('skip (empty or unreadable):', f); continue
    if 'counts' not in r: continue
    m = r.get('model', '')
    model = 'gm' if m == 'cov-matched Gaussian' else ('cubic' if m.startswith('cubic') else 'ng')
    R = r.get('realisations', r.get('R')); J = r.get('J', 4096)
    key = (model, float(r['eps']), J)
    for t, c in zip(r['ts'], r['counts']):
        a = runs.setdefault(key, {}).setdefault(t, [0, 0, []])
        a[0] += c; a[1] += R; a[2].append(f)
names = {'ng': 'short bars of F=G+eps(H^2-1)', 'gm': 'short bars of the covariance-matched Gaussian', 'cubic': 'cubic-jet count at inflection points of F'}
for key in sorted(runs, key=lambda k: (['ng', 'cubic', 'gm'].index(k[0]), k[1], k[2])):
    model, eps, J = key
    pr = predG[eps] if model == 'gm' else pred6[eps]
    files = sorted({f for v in runs[key].values() for f in v[2]})
    print('\n%s, eps=%g, J=%d  [%s]' % (names[model], eps, J, ', '.join(files)))
    print('   prediction %.4f%s' % (pr, '' if model == 'gm' or eps == 0 else '   (Gaussian with the same covariance: %.4f)' % predG[eps]))
    for t in sorted(runs[key]):
        c, R, fs = runs[key][t]
        rate = c / (R * t ** (2 / 3)); se = rate / math.sqrt(c); L = loss(model, eps, J, t); cr = rate / (1 - L)
        print('   t=%-7g R=%-8d count=%-7d rate=%.4f +- %.4f (%+.2f%%)   grid loss %.2f%% -> %.4f (%+.2f%%, %+.1f sigma)' % (
            t, R, c, rate, se, 100 * (rate / pr - 1), 100 * L, cr, 100 * (cr / pr - 1), (cr - pr) / se))
