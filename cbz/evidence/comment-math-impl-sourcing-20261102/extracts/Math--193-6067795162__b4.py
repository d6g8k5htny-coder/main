"""Explanatory exact arithmetic only; no Lean or source-theorem execution."""
from fractions import Fraction
import json
x = y = Fraction(1, 4)
value = x*x-y*y
energy = x*x+y*y
assert (x, y) != (0, 0)
assert value == 0 and energy == Fraction(1, 8) and energy < 1
assert not value > 0
assert (2*Fraction(1))**2 == 4 != 1
print(json.dumps(dict(q=[str(x), str(y)], value=str(value), energy=str(energy),
    old_set=True, patch_set=True, strict_superlevel=False,
    any_positive_epsilon_superlevel=False, scaling_unit_energy="4",
    scope="exact explanatory rational calculation; Lean NOT_RUN"), indent=2))
