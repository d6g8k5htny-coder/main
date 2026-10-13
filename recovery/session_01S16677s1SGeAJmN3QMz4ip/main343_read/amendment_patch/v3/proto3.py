"""Prototype: tail-interval oracle under reduced core, at several TAIL_PANELS, pristine vs M06_CEILING_ONLY."""
import importlib.util, sys, time
from fractions import Fraction as Q
from unittest import mock
copy_pkg, mutant_file = sys.argv[1], sys.argv[2]
sys.path.insert(0, copy_pkg)
sys.path.insert(0, copy_pkg + '/../periodic_h0')
import population_certificate as pc
spec = importlib.util.spec_from_file_location('mutant_pc', mutant_file)
mut = importlib.util.module_from_spec(spec); spec.loader.exec_module(mut)

def row(mod, core, tail):
    with mock.patch.object(mod, 'CORE_PANELS', core), mock.patch.object(mod, 'TAIL_PANELS', tail):
        t = time.monotonic(); r = mod.direct_mass(1); dt = time.monotonic() - t
    return r, dt

res = {}
for name, mod in (('pristine', pc), ('mutant', mut)):
    for tail in (512, 1024, 2048):
        r, dt = row(mod, 64, tail)
        res[name, tail] = r
        ti = r['tail_interval']
        print('%-8s core=64 tail=%4d status=%s evals=%d t=%.2fs tail_lo=%.12e tail_hi=%.12e width=%.3e disc_tail=%.3e gates_false=%s' % (
            name, tail, r['status'], r['evaluations'], dt, float(ti[0]), float(ti[1]), float(ti[1]-ti[0]),
            float(r['discretization_components']['tail']), [k for k, v in r['gates'].items() if not v]))
print()
for name in ('pristine', 'mutant'):
    c = res[name, 512]['tail_interval']
    for tail in (1024, 2048):
        f = res[name, tail]['tail_interval']
        print('%-8s coarse_upper(512) - finer_lower(%d) = %s  (%.6e)   coarse_lower<=finer_lower: %s   finer_lo<=finer_hi: %s' % (
            name, tail, 'POS' if c[1] - f[0] >= 0 else 'NEG', float(c[1] - f[0]), c[0] <= f[0], f[0] <= f[1]))
        if name == 'mutant':
            print('   exact finer_lower - mutant coarse_upper =', f[0] - c[1])
# full-core pristine run: is tail_interval independent of CORE_PANELS?
t = time.monotonic(); full = pc.direct_mass(1); dt = time.monotonic() - t
print()
print('pristine full core: status=%s evals=%d t=%.2fs' % (full['status'], full['evaluations'], dt))
print('tail_interval(full core, 512) == tail_interval(core 64, 512):', full['tail_interval'] == res['pristine', 512]['tail_interval'])
print('tail discretization equal:', full['discretization_components']['tail'] == res['pristine', 512]['discretization_components']['tail'])
print('pristine full tail_hi - mutant(core64) 1024 finer lower =', float(full['tail_interval'][1] - res['mutant', 1024]['tail_interval'][0]))
# C270's exact rational
c270 = Q(110770127964024035006453969387323164636630381124117933425519677, 1482521263040042663009052462615821307352825354839967634589869622165504)
print('C270 exact margin reproduced (finer pristine 1024 lower - mutant coarse upper):', res['pristine', 1024]['tail_interval'][0] - res['mutant', 512]['tail_interval'][1] == c270)
