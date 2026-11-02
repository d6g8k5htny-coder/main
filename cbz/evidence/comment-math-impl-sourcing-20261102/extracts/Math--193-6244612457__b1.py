#!/usr/bin/env python3
"""Paper algebra/dimensional checks ONLY. No project imports, fields, samples,
moments, coefficients, confidence evaluation or actual scientific experiment."""
from fractions import Fraction as F
import json

checks = []
def require(name, predicate, detail):
    if not predicate:
        raise AssertionError(name)
    checks.append({"name": name, "pass": True, "detail": detail})

# P1 antiderivative evaluated at cube endpoints; no radical approximation.
w_cube = F(3, 2) * (2**2 - 1**2)
require("integrated_bin_shape", w_cube == F(9, 2),
        "For [1^3,2^3), integral ell^(-1/3) is 9/2.")
require("endpoint_density_is_not_bin_mass", w_cube != 8-1,
        "Density at lower endpoint times width gives7, not9/2.")
require("physical_exposure_once", F(3, 7*24**2) * (7*24**2) == 3,
        "Pure count/exposure identity;7 and3 are algebra placeholders, not data.")
# Affine exponent pairs represent coefficient of d and constant.
field_exp = (2-1-1, -2+F(4,3))
space_exp = (-4+2+1, 4-4)
require("coefficient_height_units", field_exp == (0,F(-2,3)),
        "det transverse squared + third residual^(2/3) - Hessian column - gradient.")
require("coefficient_spatial_units", space_exp == (-1,0),
        "Physical coefficient units X^(-d) H^(-2/3).")
require("bin_mass_units", field_exp[1]+F(2,3) == 0,
        "Multiplying C by integrated lifetime shape cancels height units.")

# Exact two-tube witness and its contrary one-tube case.
q = F(21,19)
require("eta_one_twentieth_two_tube", q**6 < 2,
        f"21^6={21**6}<2*19^6={2*19**6}; hence q^2<2^(1/3).")
q_old = F(11,9)
require("eta_one_tenth_one_tube", q_old**3 < 2,
        "q<2^(1/3) separates the exact constant ratio from one leading tube.")
require("eta_one_tenth_not_two_tubes", q_old**6 > 2,
        "The weaker witness does not justify two perturbed ratio tubes.")

# Pure finite diagram endpoint arithmetic for the P8 guard; no field/control model.
delta, a, b = F(1,10), F(1), F(2)
flife = [F(3,4),F(17,20),F(1),F(6,5),F(2),F(11,5)]
plife = [F(19,20),F(21,20),F(1),F(1),F(9,5),F(2)]
require("literal_coordinate_matching", all(abs(x-y)/2 <= delta for x,y in zip(flife,plife)),
        "Assign birth/death as +life/2,-life/2, so every coordinate cost<=delta.")
def count(xs, lo, hi):
    return sum(lo <= x < hi for x in xs) if lo < hi else 0
require("half_open_matching_sandwich",
        count(flife,a+2*delta,b-2*delta) <= count(plife,a,b)
        <= count(flife,a-2*delta,b+2*delta),
        "Exact [a,b) inclusion/injectivity check, including shifted boundaries.")
require("last_bin_upper_endpoint_excluded", count([a,b],a,b)==1,
        "The final right endpoint is excluded exactly like every other bin.")
edge_life=2*delta
require("equality_guard_counterexample",
        count([edge_life],edge_life,F(1)) == 1 and count([],F(0),F(1)+2*delta)==0,
        "A length2delta bar can match diagonal atdelta; a>2delta is necessary.")

# P11/P13: pointwise identities, with the universal proof in the paper.
Q=F(5)
for x in [F(0),F(1),Q,F(8),2*Q,F(17)]:
    plus=max(F(0),x-Q)
    require("unclipping_identity_"+str(x), min(x,Q)+plus==x,
            "Exact min(x,Q)+(x-Q)_+=x.")
    require("tail_square_bound_"+str(x), plus<=x*x/(4*Q),
            "For x>=Q this is the nonnegative square(x-2Q)^2/(4Q).")
require("tail_square_polynomial_identity",
        # Coefficients of x^2, x, constant in x^2-4Q(x-Q).
        (F(1),-4*Q,4*Q*Q)==(F(1),-4*Q,(2*Q)**2),
        "Universal polynomial witness, not a numerical estimate of a count tail.")
for m in [2,10,100]:
    beta=F(1,m); rare_count=m*m
    require("bad_event_bias_countercheck_"+str(m), beta*rare_count==m and beta*rare_count>beta,
            "Small bad probability alone permits large exceptional expected count.")

# Same bounded-variable MGF optimizer already proved by the owned confidence source.
n, r, cap = F(7),F(3),F(5)
tilt=4*r/(cap*cap)
require("hoeffding_quadratic_optimizer",
        -tilt*n*r+n*tilt*tilt*cap*cap/8 == -2*n*r*r/(cap*cap),
        "Exact quadratic minimization; these symbols carry no observed sample.")
require("statistical_normalization",
        (r/(24**2))/(cap/(24**2)) == r/cap,
        "Normalize radius and cap together, not counts alone.")
# Why T+D allocation is insufficient for whole-interval support.
T,D=F(1,100),F(1,50); eta=T+D
require("whole_interval_factor_two_countercheck",
        T+D<=eta and T+2*D>eta,
        "A center at T+D and interval radiusD can extend toT+2D.")

print(json.dumps({"scope":"PAPER_ALGEBRA_ONLY_NO_SCIENTIFIC_EXECUTION",
                  "checks":checks,"count":len(checks),"all_pass":True},
                 indent=2))

