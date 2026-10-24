"""Summarise fold_<tag>.csv: decay exponents of the relative errors and agreement with
second-order predictions; also ell vs (8 sqrt2/3)|A| tau^{3/2}."""
import sys, csv, json, math
for tag in sys.argv[1:]:
    rows = list(csv.DictReader(open('fold_%s.csv' % tag)))
    meta = json.load(open('fold_%s.json' % tag))
    A = abs(float(meta['A']))
    print('==', tag, 't*=', meta['t_star'][:22], 'x0=', meta['x0'][:20], 'A=', meta['A'][:12],
          'kappa_r=', meta['kappa_r'], 'kappa_ell=', meta['kappa_ell'])
    print('%-10s %-12s %-12s %-12s %-12s %-12s %-14s' % ('tau', 'r', 'r2/8tau-1', '(2nd-order)', 'ell/pred-1', '(2nd-order)', 'ell/[(8rt2/3)|A|tau^1.5]-1'))
    pts = []
    for r in rows:
        tau = float(r['tau'])
        if abs(math.log10(tau) - round(math.log10(tau))) < 1e-9 or tau == max(float(x['tau']) for x in rows):
            e3 = float(r['ell_over_A']) / ((8 * math.sqrt(2) / 3) * tau ** 1.5) - 1
            print('%-10.3g %-12.6g %-12.4g %-12.4g %-12.4g %-12.4g %-14.4g' % (tau, float(r['r']), float(r['relerr_r2']), float(r['pred2_relerr_r2']),
                  float(r['relerr_ell']), float(r['pred2_relerr_ell']), e3))
        pts.append((tau, float(r['relerr_r2']), float(r['relerr_ell'])))
    small = [p for p in pts if p[0] <= 1e-4]
    def slope(i):
        xs = [math.log(p[0]) for p in small]; ys = [math.log(abs(p[i])) for p in small]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    print('log-log slope (tau<=1e-4):  r2 relerr ~ tau^%.6f   ell relerr ~ tau^%.6f' % (slope(1), slope(2)))
    worst = max(abs(p[1] / (p[0] * float(meta['kappa_r'])) - 1) for p in small)
    worst2 = max(abs(p[2] / (p[0] * float(meta['kappa_ell'])) - 1) for p in small)
    print('max |relerr/(kappa*tau) - 1| for tau<=1e-4:  r2: %.2e   ell: %.2e' % (worst, worst2))
    print('complex side (|tau|, Re z - x0, Im z):', meta['complex_check'])
