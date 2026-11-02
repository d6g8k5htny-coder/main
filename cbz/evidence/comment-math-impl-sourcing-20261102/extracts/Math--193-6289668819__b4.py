"""Paper-only exact rational controls; no field, source-program or coefficient run."""
from fractions import Fraction as F
import json

out = []

def record(name, condition, detail):
    if not condition:
        raise AssertionError(name)
    out.append({"name": name, "pass": True, "detail": detail})

def integral_power(j, a):
    return F(0) if j % 2 else 2*a**(j+1)/F(j+1)

for r in (F(1, 2), F(1, 17), F(3, 5)):
    a = r/2
    kernel_mass = 6/r**3*(a*a*integral_power(0, a)-integral_power(2, a))
    record("positive_kernel_mass", kernel_mass == 1, str(r))
    for degree in range(6):
        endgap = (-a)**degree-a**degree
        derivative_sum = F(0) if degree == 0 else degree*((-a)**(degree-1)+a**(degree-1))
        direct = 12/r**3*(endgap+r*derivative_sum/2)
        via_kernel = F(0)
        if degree >= 3:
            coeff = degree*(degree-1)*(degree-2)
            via_kernel = 6*coeff/r**3*(a*a*integral_power(degree-3, a)-integral_power(degree-1, a))
        expected = F(6) if degree == 3 else 3*r*r if degree == 5 else F(0)
        record("trapezoid_cubic_observation", direct == via_kernel == expected,
               {"r": str(r), "degree": degree, "value": str(direct)})

block_cases = 0
for a in (F(1, 3), F(1), F(5)):
    for v in (F(1, 7), F(2)):
        for q1, q2 in ((F(0), F(0)), (F(2), F(0)), (F(1, 3), F(-4, 5))):
            floor = min(a, v)/(2*(1+q1*q1+q2*q2))
            for x1 in (F(-2), F(0), F(3, 7)):
                for x2 in (F(-1), F(1, 5)):
                    for y in (F(-1), F(0), F(2)):
                        variance = a*((x1-q1*y)**2+(x2-q2*y)**2)+v*y*y
                        assert variance >= floor*(x1*x1+x2*x2+y*y)
                        block_cases += 1
record("odd_block_floor", True, {"deterministic_cases": block_cases})
record("naive_diagonal_floor_adverse", F(1) < F(5),
       "q=(2,0), a=V=1, x=q,y=1 has variance1 but input norm-squared5")

cap_cases = 0
for m in (F(1), F(3), F(100)):
    for eta in (m, m/3, m/100):
        for zeta in (m, m/2, m/100):
            h = min(F(1, 4), eta/(48*m), zeta*eta/(52*m*m))
            s = 8*m*h*h/eta
            r = h*h*zeta/(16*m)
            p = 6*m*h/eta
            gap = m*r**3/6
            assert s <= h and r <= h/2 and p <= F(1, 8)
            assert m*(h+s) <= eta/2
            assert 2*m*h+24*m*m*h/eta <= zeta/2
            assert zeta*h**3/16 > gap
            assert eta*s*s/16 > gap
            # Literal cubic ridge: phi'=zeta/2*(t^2-r^2/4).
            phi = lambda t: zeta*t**3/6-zeta*r*r*t/8
            birth, death = phi(-r/2), phi(r/2)
            assert birth-death == zeta*r**3/12
            assert phi(-h) < death < birth < phi(h)
            cap_cases += 1
record("local_cap_constants_and_literal_cubic", True, {"deterministic_cases": cap_cases})

ratio = F(130, 129)**4/2
record("full_spectrum_norm_geometric_tail", ratio < F(3, 4), str(ratio))
record("positive_moment_witness_constants",
       F(1, 1056)**2 == F(1, 1115136)
       and F(72, 4096*4*66*5) == F(3, 225280),
       "elementary lower witnesses only; no actual moment evaluation")

atoms = {F(1): 1}
weak = lambda t: sum(n for x, n in atoms.items() if 0 < x <= t)
strict = lambda t: sum(n for x, n in atoms.items() if 0 < x < t)
halfopen = sum(n for x, n in atoms.items() if F(1) <= x < F(2))
record("endpoint_atom_adverse_and_strict_transfer",
       weak(F(2))-weak(F(1)) == 0 and halfopen == 1
       and strict(F(2))-strict(F(1)) == halfopen,
       "two weak endpoint differences alone do not produce [a,b)")

print(json.dumps({"status": "PASS_PAPER_CONTROLS_ONLY", "records": out,
                  "scientific_acceptance": False, "field_calls": 0,
                  "coefficient_calls": 0, "project_program_calls": 0},
                 sort_keys=True, indent=2))

