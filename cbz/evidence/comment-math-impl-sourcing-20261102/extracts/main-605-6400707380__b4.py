"""Exact rational controls for C320; finite checks do not prove the lemma."""
from fractions import Fraction as F
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = []
negative_controls = []


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def bad(h):
    return F(1) if h == F(1, 2) else abs(h - F(1, 2))


def good(h):
    return min(h * h, F(1, 16))


for m in range(9):
    n = 2 ** (m + 3)
    h = F(1, 2) + F(1, n)
    ratio = bad(h) / h ** m
    check(f'bad m={m} point positive and in domain', 0 < h <= 1 and 0 < bad(h) <= 1)
    check(f'bad m={m} exact sequence identity', bad(h) == F(1, n))
    check(f'bad m={m} displayed ratio bound', ratio <= F(2 ** m, n))
    check(f'bad m={m} global c=1/4 candidate fails', ratio < F(1, 4))
negative_controls.append('pointwise-positive semialgebraic whole-interval claim fails')
check('bad exceptional point is positive', bad(F(1, 2)) == 1)
for h in [F(1, 64), F(1, 16), F(1, 8), F(1, 4)]:
    check(f'bad near-zero control at {h}', bad(h) >= F(1, 4))

h1, c0, m = F(1, 4), F(1), 2
for h0 in [F(1, 4), F(1), F(2)]:
    cstar = min(c0, good(h1) / h0 ** m)
    probes = sorted({F(1, 64), F(1, 16), F(1, 8), h1, (h1+h0)/2, h0})
    check(f'extension positive constant h0={h0}', cstar > 0)
    for h in probes:
        check(f'good bound h0={h0} h={h}', good(h) >= cstar * h ** m)
        check(f'half-infimum bound h0={h0} h={h}', good(h)/2 >= cstar/2 * h ** m)

check('unshrunk c0 counterexample', good(F(1)) < c0)
negative_controls.append('reusing c0 without shrinking fails')
check('omitting h0 power counterexample', good(F(2)) < good(h1) * F(2) ** m)
negative_controls.append('omitting h0^m in extension constant fails')
check('omitting half counterexample', good(F(1))/2 < F(1, 16))
negative_controls.append('dropping half-infimum factor fails')

# A finite analogue checks inclusion direction, not continuum compactness.
points = [(F(1, 8), F(1, 8)), (F(1, 4), F(1, 16)),
          (F(1, 2), F(1, 2)), (F(3, 4), F(1, 4)), (F(1), F(1, 8))]
etas = [F(1, 16), F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(1)]
sets = [{i for i, (distance, quality) in enumerate(points) if distance >= eta} for eta in etas]
infs = [min(points[i][1] for i in subset) for subset in sets]
check('all selected finite slices nonempty', all(sets))
check('finite slices shrink with increasing eta', all(b <= a for a, b in zip(sets, sets[1:])))
check('finite infima are nondecreasing', all(a <= b for a, b in zip(infs, infs[1:])))
check('reversed infimum inequality fails somewhere', any(b > a for a, b in zip(infs, infs[1:])))
negative_controls.append('reversed nested-infimum direction fails')
check('empty slice separated from infimum case', not [p for p in points if p[0] >= F(2)])

result = {
    'scope': 'Finite exact rational controls only; analytic proof and substantive review are separate',
    'checks_passed': len(checks), 'checks': checks,
    'negative_control_classes': negative_controls,
    'arithmetic': 'fractions.Fraction; no floating point or sampled Gaussian field',
    'package_code_executed': False, 'theorem_proved_by_tests': False,
}
(ROOT / 'EXACT_CONTROL_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))
