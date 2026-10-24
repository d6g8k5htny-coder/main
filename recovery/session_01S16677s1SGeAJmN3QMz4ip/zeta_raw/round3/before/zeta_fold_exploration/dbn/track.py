"""Track the real zeros of H_t on [1, ZMAX] as t decreases from 0; detect pair collisions.

For each adjacent pair (z_k, z_{k+1}) we follow the critical point c_k(t) between them
(Newton on H'=0, predictor dc/dt = H'''/H'') and the 'discriminant'
    D_k(t) = sigma_k * H_t(c_k(t)),    sigma_k = sign H_0(c_k(0)).
D_k > 0  <=>  the pair is still real; D_k changes sign exactly at a (generic) double zero.
On the first sign change in the window we bisect (Illinois) in t to high precision,
then re-scan just below t* and restart, giving the sequence of collision events.

Usage: python3 dbn/track.py [ZMAX=270] [TMIN=-12] [DT=0.01] [TAG=z<ZMAX>]
Event log: results/dbn/track_log_<TAG>.txt; stdout was saved as results/dbn/track_<TAG>.out.

Runs kept in results/dbn/: `track.py 270` (track_z270.out) and `track.py 103 -40 0.02 low10`
(track_low10.out).

Completeness: the tracker stops when its own critical-point ordering check fails (in the
ZMAX=270 run at t ~ -1.355). Event lists are complete only down to that t. That stop is a
limitation of this code, not a mathematical event. For ZMAX=103 the bookkeeping becomes
step-size dependent near t ~ -6.38 (track_low10_dt_sensitivity.out), so that window's
event list is complete only down to about -6.38.
"""
import sys, json, mpmath as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "dbn"

from hflow import HFlow
from zeros import scan, refine_root

ZMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 270
TMIN = mp.mpf(sys.argv[2]) if len(sys.argv) > 2 else mp.mpf(-12)
DT = mp.mpf(sys.argv[3]) if len(sys.argv) > 3 else mp.mpf('0.01')
TAG = sys.argv[4] if len(sys.argv) > 4 else 'z%d' % ZMAX
hf = HFlow(dps=80)
rank0 = {}

def newton_crit(c, t, it=30):
    for _ in range(it):
        d = hf.derivs(c, t, 3)
        step = d[1] / d[2]
        c -= step
        if abs(step) < mp.mpf(10) ** -60:
            break
    return c

def D(k_c_sigma, t):
    c, s = k_c_sigma
    c = newton_crit(c, t)
    return s * hf.derivs(c, t, 0)[0], c

zs0, _ = scan(hf, 0, 1, ZMAX, mp.mpf('0.05'))
labels = list(range(1, len(zs0) + 1))      # zero labels = rank at t=0
zs = list(zs0)
t = mp.mpf(0)
events = []
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / ('track_log_%s.txt' % TAG), 'w')

def setup(zs, t):
    pairs = []
    for i in range(len(zs) - 1):
        a, b = zs[i], zs[i + 1]
        # critical point between consecutive zeros: bracket H' sign change
        f = lambda z: hf.derivs(z, t, 1)[1]
        c = refine_root(f, a, b, f(a), f(b))
        s = 1 if hf.derivs(c, t, 0)[0] > 0 else -1
        pairs.append([c, s])
    return pairs

pairs = setup(zs, t)
while t > TMIN and len(events) < 12:
    tn = t - DT
    newpairs, Ds = [], []
    for (c, s) in pairs:
        d = hf.derivs(c, t, 3)
        cpred = c - DT * d[3] / d[2]
        val, cn = D((cpred, s), tn)
        newpairs.append([cn, s]); Ds.append(val)
    # sanity: critical points must stay ordered
    cs = [p[0] for p in newpairs]
    if any(cs[i] >= cs[i + 1] for i in range(len(cs) - 1)):
        print('ordering violated at t=', tn); break
    neg = [i for i, v in enumerate(Ds) if v <= 0]
    if not neg:
        pairs, t = newpairs, tn
        continue
    # first event in (tn, t]: bisect each candidate, keep the largest crossing time
    best = None
    for i in neg:
        c_hi, s = pairs[i]
        lo, hi = tn, t
        f_hi = D((c_hi, s), hi)[0]; f_lo, c_lo = D((newpairs[i][0], s), lo)
        cache = {'c': c_hi}
        def g(tt):
            v, cc = D((cache['c'], s), tt); cache['c'] = cc; return v
        tstar = refine_root(g, lo, hi, f_lo, f_hi, tol=mp.mpf(10) ** -45)
        cstar = newton_crit(cache['c'], tstar)
        if best is None or tstar > best[0]:
            best = (tstar, i, cstar)
    tstar, i, cstar = best
    dd = hf.derivs(cstar, tstar, 3)
    ev = dict(t_star=mp.nstr(tstar, 40), x0=mp.nstr(cstar, 40), pair=(labels[i], labels[i + 1]),
              gamma_pair_t0=(mp.nstr(zs0[labels[i] - 1] / 2, 12), mp.nstr(zs0[labels[i + 1] - 1] / 2, 12)),
              H=mp.nstr(dd[0], 5), Hp=mp.nstr(dd[1], 5), A=mp.nstr(dd[2] / 2, 25), B=mp.nstr(dd[3] / 6, 25))
    events.append(ev)
    print('EVENT', json.dumps(ev)); sys.stdout.flush()
    log.write(json.dumps(ev) + '\n'); log.flush()
    # restart just below t*: rescan zeros, drop the annihilated pair
    t = tstar - mp.mpf('1e-6')
    znew, _ = scan(hf, t, 1, ZMAX, mp.mpf('0.02'))
    # relabel by matching continuity (remove labels i, i+1)
    labels = labels[:i] + labels[i + 2:]
    if len(znew) != len(labels):
        print('WARNING: zero count mismatch after event', len(znew), len(labels), 'at t', mp.nstr(t, 10))
        # fall back: keep labels aligned by position of nearest old zero
        labels = list(range(1, len(znew) + 1)); labels = [-l for l in labels]
    zs = znew
    pairs = setup(zs, t)
print('done at t', mp.nstr(t, 10), 'events', len(events))
