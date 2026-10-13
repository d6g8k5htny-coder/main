"""Mutation experiments on a scratch copy (never on the worktree)."""
import importlib, sys, pathlib, types
from fractions import Fraction as Q
SRC = pathlib.Path('population_certificate.py').read_text()

def load(name, text):
    mod = types.ModuleType(name); mod.__file__ = str(pathlib.Path('population_certificate.py').resolve())
    exec(compile(text, name, 'exec'), mod.__dict__); return mod

def overlap(mod, j=1):
    d = mod.direct_mass(j); cdf = mod.cdf_mass(Q(1, 1 << (3*j))); n = 4**j
    lo = max(d['normalized_interval'][0], n*cdf[0]); hi = min(d['normalized_interval'][1], n*cdf[1])
    return d['status'], (lo <= hi), lo, hi, d['normalized_interval'], tuple(n*x for x in cdf)

base = load('base', SRC)
print('BASE  negative_controls:', base.negative_controls()['status'], ' row1 overlap:', overlap(base)[:2])

# M1: missing factor 2 in the control measure, applied identically to BOTH routes (1/15 -> 1/30, n/15 -> n/30)
m1 = SRC.replace("Q(1, 15)", "Q(1, 30)").replace("Q(n, 15)", "Q(n, 30)")
assert m1 != SRC
M1 = load('m1', m1)
nc = M1.negative_controls()
print('M1 (shared factor-2 loss) negative_controls:', nc['status'], {k: v for k, v in nc['gates'].items() if 'mass' in k or 'sign' in k or 'cond' in k})
st, ov, lo, hi, dn, cn = overlap(M1)
print('M1 row1 direct status', st, 'routes overlap', ov, 'intersection ~', float(lo), float(hi))
b, err = M1.coefficient_interval(), M1.error_interval(1)
print('M1 row1 coefficient classify:', M1.classify_coefficient((lo, hi), b, err), ' row-8-like classify of same value with E(8):', M1.classify_coefficient((lo, hi), b, M1.error_interval(8)))

# M2: wrong split root (bracket a point that is not the crossing) shared by both routes
m2 = SRC.replace("""    require(36 * lo ** 3 <= tau * (3 + lo ** 2) ** 2
            and 36 * hi ** 3 >= tau * (3 + hi ** 2) ** 2,
            'Split root is not enclosed')
    return lo, hi""", """    lo, hi = lo * Q(11, 10), hi * Q(11, 10)
    return lo, hi""")
assert m2 != SRC
M2 = load('m2', m2)
st, ov, lo, hi, dn, cn = overlap(M2)
print('M2 (wrong root, shared) row1 direct status', st, 'routes overlap', ov, 'direct', float(dn[0]), float(dn[1]), 'cdf', float(cn[0]), float(cn[1]))
print('   true n*m(1/8) ~ 0.24989 ; M2 agreed value is an over-estimate:', float(lo) > 0.2499)

# M3: wrong closed-form constant (36/7 -> 36/5) in the CDF route only
m3 = SRC.replace("Q(36, 7)", "Q(36, 5)")
assert m3 != SRC
M3 = load('m3', m3)
st, ov, lo, hi, dn, cn = overlap(M3)
print('M3 (CDF antiderivative constant wrong) row1 direct status', st, 'routes overlap', ov)

# M4: wrong tail exponent in the direct route only (t^-11/4 -> t^-9/4)
m4 = SRC.replace("quarter[0] ** 11, quarter[1] ** 11", "quarter[0] ** 9, quarter[1] ** 9")
assert m4 != SRC
M4 = load('m4', m4)
st, ov, lo, hi, dn, cn = overlap(M4)
print('M4 (direct tail exponent wrong) row1 direct status', st, 'routes overlap', ov)

# M5: saturation literal is not computed: change the literal 1/5 to 1/10 in cdf_mass only -> whole_rectangle_mass gate flips but nothing computed
m5 = SRC.replace("""    if tau >= Q(9, 4):
        return Q(1, 5), Q(1, 5)""", """    if tau >= Q(9, 4):
        return Q(1, 10), Q(1, 10)""")
M5 = load('m5', m5)
print('M5 (saturation literal changed): whole_rectangle_mass gate', M5.negative_controls()['gates']['whole_rectangle_mass'], '- shows the gate reads a literal, not a computed mass')
