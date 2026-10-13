import pathlib, types
from fractions import Fraction as Q
SRC = pathlib.Path('population_certificate.py').read_text()
def load(name, text):
    mod = types.ModuleType(name); mod.__file__ = str(pathlib.Path('population_certificate.py').resolve())
    exec(compile(text, name, 'exec'), mod.__dict__); return mod
def overlap(mod, j=1):
    d = mod.direct_mass(j); cdf = mod.cdf_mass(Q(1, 1 << (3*j))); n = 4**j
    if d['normalized_interval'] is None:
        return d['status'], {k: v for k, v in d['gates'].items() if not v}
    lo = max(d['normalized_interval'][0], n*cdf[0]); hi = min(d['normalized_interval'][1], n*cdf[1])
    return d['status'], (lo <= hi), float(lo), float(hi), tuple(float(x) for x in d['normalized_interval']), tuple(float(n*x) for x in cdf)
# M2: shared wrong split root: shift the certified bracket by +2^-20 (keeps width, positivity, all direct gates)
m2 = SRC.replace("""            'Split root is not enclosed')
    return lo, hi""", """            'Split root is not enclosed')
    return lo + Q(1, 1 << 20), hi + Q(1, 1 << 20)""")
assert m2 != SRC
M2 = load('m2', m2)
print('M2 (shared root shifted by 2^-20) row1:', overlap(M2))
print('   true n*m(1/8) = 0.2280146264648009764 (40-digit); both mutated routes agree on a value above it and the overlap gate does not fire')
# Row-1 vacuity: inner envelope vs admissible range using the module's own predicates
base = load('base', SRC)
b, e = base.coefficient_interval(), base.error_interval(1)
env = base.coefficient_envelopes(b, e)
print('row1 inner envelope', float(env['inner'][0]), float(env['inner'][1]), ' admissible n*m range [0, 4/5]')
for obs in ((Q(0), Q(0)), (Q(4,5), Q(4,5)), (Q(0), Q(4,5)), (Q(1,10), Q(1,10))):
    print('   classify_coefficient(row1, observed=%s) ->' % (tuple(str(x) for x in obs),), base.classify_coefficient(obs, b, e))
b, e = base.coefficient_interval(), base.error_interval(2)
print('row2 inner envelope', float(base.coefficient_envelopes(b, e)['inner'][0]), float(base.coefficient_envelopes(b, e)['inner'][1]), ' classify(0.114)->', base.classify_coefficient((Q(114,1000),Q(114,1000)), b, e))
