import sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf, global_h0_selector as g
def run(p, dom, **lim):
    b = sf.Budget(time.monotonic(), **lim); b.begin_fixture('e')
    r = g.isolate_roots(p, dom, budget=b)
    return r['status'], r.get('failure'), [(x['isolator'][0], x['isolator'][1], x['exact_value'], x['multiplicity']) for x in r['roots']], r['work']['max_branch_splits']
D = (Q(-1, 64), Q(1, 64))
# two roots closer than ROOT_WIDTH -> must not certify
e = Q(1, 2**170)
print('close pair', run(g._mul((-e, Q(1)), (e, Q(1))), D)[:2])
# exact root at 0 plus a root at 2^-170: isolator would share endpoint 0
print('shared endpoint', run(g._mul((Q(0), Q(1)), (-e, Q(1))), D)[:2])
# root exactly at domain endpoint and a nearby root
print('endpoint+near', run(g._mul((Q(1,64), Q(1)), (Q(1,64)-e, Q(1))), D)[:2])
# root at 2^-170 alone (not exact), isolator must contain it strictly
st, f, roots, depth = run((-e, Q(1)), D)
print('single tiny root', st, roots, 'contains', roots[0][0] < e < roots[0][1], 'depth', depth)
# triple root and a tie: x^3 - u x with u a non-square
u = Q(3, 2**33)
st, f, roots, depth = run((Q(0), -u, Q(0), Q(1)), (-g.A_STAR, g.A_STAR))
print('tie cubic', st, [(float(a), ex, m) for a, c, ex, m in roots])
# x^3 (u=v=0)
print('triple', run((Q(0), Q(0), Q(0), Q(1)), (-g.A_STAR, g.A_STAR))[2])
# squarefree sanity vs brute force on (x-1)^2 (x+1)^3 (x-1/3)
p = (Q(1),)
for r, m in ((Q(1), 2), (Q(-1), 3), (Q(1, 3), 1)):
    for _ in range(m): p = g._mul(p, (-r, Q(1)))
b = sf.Budget(time.monotonic()); b.begin_fixture('s')
print('squarefree', g._squarefree(p, b, 'sq'))
# Sturm count semantics: V(a)-V(b) on (a,b] when b is a root
p = (Q(-1), Q(0), Q(1))  # x^2-1
seq = g._sturm(p, b, 's')
print('V(0)-V(1) (b root) =', g._variation(seq, Q(0), b, 'a') - g._variation(seq, Q(1), b, 'b'),
      ' V(-1)-V(0) (a root) =', g._variation(seq, Q(-1), b, 'a') - g._variation(seq, Q(0), b, 'b'))
# _CALL_IDS determinism: same input twice in one process -> different phase names in refinements
b1 = sf.Budget(time.monotonic()); b1.begin_fixture('x'); g.isolate_roots((Q(-2), Q(0), Q(1)), (Q(-2), Q(2)), budget=b1)
b2 = sf.Budget(time.monotonic()); b2.begin_fixture('x'); g.isolate_roots((Q(-2), Q(0), Q(1)), (Q(-2), Q(2)), budget=b2)
print('phase names differ across identical calls:', b1.snapshot()['root_refinements'][0]['phase'], b2.snapshot()['root_refinements'][0]['phase'])
# deadline monotonicity: Budget rejects future start; checkpoint after deadline
try:
    sf.Budget(time.monotonic() + 10.0)
except ValueError as exc: print('future start rejected:', exc)
b = sf.Budget(time.monotonic(), seconds=0)
try: b.checkpoint('x')
except sf.Inconclusive as exc: print('zero-second budget:', exc)
