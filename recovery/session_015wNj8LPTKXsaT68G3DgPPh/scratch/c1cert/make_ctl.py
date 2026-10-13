"""Build ctl.json (the NOTE's controls section) from controls_final.txt, controls.REFERENCE and RESULTS.json."""
import json, re, sys, pathlib
SC = pathlib.Path(__file__).parent
PK = pathlib.Path('/home/user/wt_cusp/frontiers/cusp_coefficient_certified_20261001')
sys.path.insert(0, str(PK))
sys.dont_write_bytecode = True
import controls
REF = controls.REFERENCE
V = json.loads((PK / 'RESULTS.json').read_text())['values']
txt = (SC / 'controls_final.txt').read_text().splitlines()
c2 = re.search(r'control (\S+) .* difference (\S+)', txt[1]).groups()
c3 = re.search(r'control (\S+) .* difference (\S+)', txt[3]).groups()
l4 = [t for t in txt if t.startswith('4. Lemma S')][0]
s24 = re.search(r'= (\S+) \(at g = \(([^)]*)\)\); certified entry bound E = (\S+)', l4).groups()


def mid(p):
    return 0.5 * (float(p[0]) + float(p[1]))


for c, key in ((c2, 'd=2'), (c3, 'd=3')):
    lo, hi = map(float, V[key]['c1'])
    assert lo <= float(c[0]) <= hi, (key, c, lo, hi)

rows = [
    ('[CU] §8: `c1`, `d = 2`', 'Math-#207 (826b8be) PROOF.md section 8: c1, d = 2', V['d=2']['c1'], '-0.26939883'),
    ('[CU] §8: `c1`, `d = 3`', 'Math-#207 (826b8be) PROOF.md section 8: c1, d = 3', V['d=3']['c1'], '-0.21184835'),
    ('[CU] §8: `c`, `d = 2`', 'Math-#207: c, d = 2', V['d=2']['c_(d,ref)'], '0.0734069193'),
    ('[CU] §8: `c`, `d = 3`', 'Math-#207: c, d = 3', V['d=3']['c_(d,ref)'], '0.0417759318'),
    ('[CU] §8: `c1/c`, `d = 2`', 'Math-#207: c1/c, d = 2', V['d=2']['c1 / c_(d,ref)'], '-3.669938'),
    ('[CU] §8: `c1/c`, `d = 3`', 'Math-#207: c1/c, d = 3', V['d=3']['c1 / c_(d,ref)'], '-5.071062'),
    ('[CU] §8: `I^cand`, `d = 2`', 'Math-#207: I_cand, d = 2', V['d=2']['I_cand = (3^(1/4)/2) c1'], '-0.1772744'),
    ('[CU] §8: `I^cand - c1`, `d = 2`', 'Math-#207: I_cand - c1, d = 2', V['d=2']['I_cand - c1'], '0.0921244'),
    ('[CU] §8: `I^cand`, `d = 3`', 'Math-#207: I_cand, d = 3', V['d=3']['I_cand = (3^(1/4)/2) c1'], '-0.1394041'),
    ('[CU] §8: `I^cand - c1`, `d = 3`', 'Math-#207: I_cand - c1, d = 3', V['d=3']['I_cand - c1'], '0.0724443'),
    ('Math-#216 table: `c1`, `d = 1` (Math-#214 `C_1`)', 'Math-#216 table: c1, d = 1 (= Math-#214 C_1)', V['d=1']['c1'],
     '-0.227606'),
]
lines = []
for label, key, cert, q in rows:
    assert float(q) == REF[key], (key, q, REF[key])
    m = mid(cert)
    half = 0.5 * 10 ** -len(q.split('.')[1])
    assert abs(float(q) - m) < half, (key, q, m)
    lines.append('| %s | `%s` | `%.10f` | `%.1e` | `%.0e` |' % (label, q, m, float(q) - m, half))
mcs = []
for label, key in (('shipped fixed-seed', 'Math-#207: shipped fixed-seed Monte Carlo c1, d = 3'),
                   ('development (`2e8` samples)', 'Math-#207: development Monte Carlo c1, d = 3 (2e8 samples)'),
                   ('referee (`5e8` samples)', 'Math-#207: referee Monte Carlo c1, d = 3 (5e8 samples)')):
    mc, se = REF[key]
    mcs.append('%s: `%.6f +- %.6f` (`%+.2f` standard errors)' % (label, mc, se, (mc - mid(V['d=3']['c1'])) / se))
table = f"""These recompute the same quantities in libm floating point by routes that share only the Gauss–Legendre nodes
(`gauss.py`) with the certificate. They guard against a wrong reduction, not against rounding.

1. **`d = 2` without the Mellin identity.** Given `(a = -A, gamma, b)`, `Y` is Gaussian with standard deviation `a/sqrt 6`
   and mean `(ab - gamma^2)/4`, so `E|Y|^(7/4)` is a Kummer function. Averaging over `b` leaves a 2D integral over
   `(a, gamma)`, done by a product Gauss rule. Result: `{c2[0]}`, `{c2[1]}` from the certified midpoint.
2. **`d = 3` by the same route.** The eigenvalues `a1 > a2` of `-A`, the Vandermonde factor and `gamma ~ N(0, 2 I)` give a
   4D integral. Result: `{c3[0]}`, `{c3[1]}` from the certified midpoint.

   Both controls lie inside the certified intervals.
3. **The sources' quoted values.** Each is the correct rounding of the certified midpoint: the difference is below half a
   unit in its last digit.

| quoted value | quoted | certified midpoint | difference | half-unit |
|---|---|---|---|---|
""" + '\n'.join(lines) + """

[CU]'s three `d = 3` Monte Carlo runs (§8, remark 3), against the certified value:
""" + '\n'.join('- ' + m + (';' if i < len(mcs) - 1 else '.') for i, m in enumerate(mcs)) + f"""

4. **Lemma S, actual deviations.** A direct floating lattice sum over `|n_i| <= 2` computes the deviations
   `D^g K_24(0) - D^g phi(0)` of all jet covariance entries (`d = 3`, standard frame, variance-one normalization). The
   largest is `{s24[0]}`, at `g = ({s24[1]})`, the `d_u^8` entry `Cov(f4, f4)`. The certified entry bound is
   `E = {s24[2]}`, which exceeds it by a factor of about `{('%.0e' % (float(s24[2]) / float(s24[0]))).replace('e+0', 'e')}`."""
(SC / 'ctl.json').write_text(json.dumps({'table': table}, indent=1))
print(table)
