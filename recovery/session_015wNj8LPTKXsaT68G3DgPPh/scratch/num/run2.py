import math, random, json, time
from fractions import Fraction as Fr
from coeff import *
# typed-only classifier vs direct
rng = random.Random(5); bad = tot = 0; hist = {0:0,1:0,2:0}
while tot < 20000:
    k = rng.choice([0.5, 1.0, 2.0]); a, b, c = (rng.uniform(-4, 4) for _ in range(3))
    B = b - a*a/(12*k); s = -abs(B)/2 - rng.uniform(0, 6)
    n1, n2 = n_of(s, a, b, c, k), n_direct(s, a, b, c, k); tot += 1; hist[n1] += 1
    if n1 != n2: bad += 1
print('typed classifier vs direct: mismatches', bad, 'of', tot, 'histogram', hist)
# closed forms: [LM] cubic (a,b,c)=(0,-2k,0): I2 = -6k^2 B^3 = 48 k^5, I1 = 0 ; B=0 family (0,0,c): I1 = 72 k^3 D^2 = 18 k^3 c^2, I2 = 0
for k in (0.5, 1.0, 2.0):
    print('LM cubic k=%g' % k, I_j(0.0, -2*k, 0.0, k), 'expected', (0.0, 48*k**5))
    for c in (0.5, 1.0, 3.0):
        print('  B=0 c=%g' % c, I_j(0.0, 0.0, c, k), 'expected', (18*k**3*c*c, 0.0))
# exact-rational check of the antiderivative and the two closed forms
kq = Fr(1); Bq = Fr(-2); s1, s2 = Bq, Bq/2
F = lambda s: 12*kq*kq*s**3 - 9*kq*kq*Bq*Bq*s
print('exact [LM] I2 =', F(s2)-F(s1), '== 48 ?', F(s2)-F(s1) == 48)
# high precision runs at L=24
t0=time.time()
out = {}
for k in (0.5, 1.0, 2.0):
    S = setup(24.0, k, 0.0)
    g40 = alphas_gh(S, k, 40); g60 = alphas_gh(S, k, 60)
    m = alphas_mc(S, k, 1500000, seed=11)
    f = S['pb0']/S['z0']
    out[k] = dict(gh40=g40, gh60=g60, mc=m, prefactor_b0=f, I1_mc=m[0]/f, I2_mc=m[1]/f, muA=S['muA'], sA=S['sA'], z0=S['z0'], covJ=S['covJ'], muJ=S['muJ'])
    print(f'k={k}: GH40 {g40} GH60 {g60} MC {m[:2]} ± {m[2:]}  ratio2(MC)={m[1]/(m[0]+m[1]):.5f}  t={time.time()-t0:.0f}s', flush=True)
json.dump(out, open('run2.json','w'), indent=1, default=str)
