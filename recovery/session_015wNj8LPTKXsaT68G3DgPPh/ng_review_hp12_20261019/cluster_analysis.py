# (30) in d = 1: rho(t) = E[N_t 1{N_t >= 2}] / E[N_t] must tend to 0 as t -> 0.
# Baseline: for a Poisson count with mean mu, E[N 1{N>=2}]/E N = 1 - exp(-mu) ~ mu, so rho/mu ~ 1 means no clustering
# beyond independent placement; a ratio staying of order one (not growing like t^(-2/3)) is what (30) predicts.
import json, sys, os, math
d = sys.argv[1] if len(sys.argv) > 1 else '.'
for f in sorted(os.listdir(d)):
    if not (f.startswith('clus_eps') and f.endswith('.json')): continue
    try: r = json.load(open(os.path.join(d, f)))
    except Exception: print('skip', f); continue
    R = r['realisations']
    print('\n%s: eps=%g R=%d J=%d seed=%d' % (f, r['eps'], R, r['J'], r['seed']))
    for t, c, m, mw in zip(r['ts'], r['counts'], r['realisations_with_N_ge_2'], r['sum_N_on_N_ge_2']):
        mu = c / R; rho = mw / c if c else float('nan')
        se = math.sqrt(mw) / c if c and mw else float('nan')
        print('   t=%-7g E N=%.5f  P(N>=2)=%.6f  rho=E[N 1{N>=2}]/E N=%.5f +- %.5f  rho/(1-exp(-mu))=%.2f' % (
            t, mu, m / R, rho, se, rho / (1 - math.exp(-mu)) if mu > 0 else float('nan')))
