from fractions import Fraction as F
import json

checks = 0
rejected = 0

def check(condition):
    global checks
    if not condition:
        raise RuntimeError('exact control failed')
    checks += 1

def reject(condition):
    global rejected
    if condition:
        raise RuntimeError('invalid inference accepted')
    rejected += 1

# Uniform bounds proving the whole lifetime band stays in the marked set.
check(F(5, 6)**6 > F(1, 8))
check(F(9, 8)**3 * F(7, 8) > 1)
check(F(6, 5)**3 * F(9, 8) < 2)
check(F(6, 5) - F(9, 8) == F(3, 40))

for n in range(1, 9):
    R = F(1, 2)**(2*n)
    ell = R**3
    check(R/4 < R*F(5, 6))
    for j in range(7):
        r = R*(F(5, 6) + F(j, 6)*(F(8, 9)-F(5, 6)))
        center = ell/r**3
        lo = max(F(1), center-r**3)
        hi = min(F(2), center+r**3)
        q = max(F(0), hi-lo)
        check(q <= 2*r**3)
        check(r**2*q <= 2*r**5)
        check((r**2*q)/r**2 == q)
        for shift in [F(-1, 8), F(0), F(1, 8)]:
            lifetime = ell + shift*ell**2
            k = lifetime/r**3
            check(1 <= k <= 2)
            check(abs(k-center) <= r**3)
            check(abs(lifetime-ell) <= r**6)
    # Independent r- and k-coordinate integration over a rational cube window.
    a, b = F(9, 8), F(6, 5)
    integral_r = 1/(R/b) - 1/(R/a)
    integral_k = F(1, 3)/R * (3*b-3*a)
    check(integral_r == integral_k == F(3, 40)/R)
    reject(integral_r == (3*b-3*a)/R)  # omitted 1/3 Jacobian
    reject(F(3, 40)/R <= 2*R**2)      # false fiberwise r^3 suppression

for delta in [F(1), F(2), F(3), F(4), F(5)]:
    density_power = (delta-1)/3
    cumulative_power = (delta+2)/3
    check(density_power+1 == cumulative_power)
    check(F(1, 3)/cumulative_power == 1/(delta+2))
    reject(density_power == delta/3)  # omitted density Jacobian cost

print(json.dumps({'exact_checks': checks, 'invalid_cases_rejected': rejected,
                  'field_theorem_verified': False}, sort_keys=True))
