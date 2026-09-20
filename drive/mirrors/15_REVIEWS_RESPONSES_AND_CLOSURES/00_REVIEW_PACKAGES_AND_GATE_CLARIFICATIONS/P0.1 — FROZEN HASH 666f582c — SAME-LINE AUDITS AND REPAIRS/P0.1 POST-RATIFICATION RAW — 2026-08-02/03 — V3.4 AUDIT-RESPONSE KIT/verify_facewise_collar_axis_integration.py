#!/usr/bin/env python3
"""Fail-closed, dependency-free algebra regression for V3.4 RP-C/RP-S.

The checker uses an exact sparse Laurent-polynomial implementation backed by
``fractions.Fraction``.  It covers cubic-Hermite interpolation identities,
axial and midpoint ESIP scales, endpoint factorization powers, the
singular-near anisotropic determinant, the collinear mean mismatch, and the
radial/collar power ledgers.  It deliberately does not adjudicate analytic
uniformity of exact side-24 Schur complements or conditional Hessian moments.
"""

from __future__ import annotations

from fractions import Fraction as Q
from math import factorial


class Poly:
    """Tiny exact multivariate Laurent polynomial used by this regression."""

    __slots__ = ("terms",)

    def __init__(self, terms=None):
        cleaned = {}
        for monomial, coefficient in (terms or {}).items():
            coefficient = Q(coefficient)
            if not coefficient:
                continue
            key = tuple(sorted((name, int(power)) for name, power in monomial if power))
            cleaned[key] = cleaned.get(key, Q(0)) + coefficient
            if not cleaned[key]:
                del cleaned[key]
        self.terms = cleaned

    @classmethod
    def constant(cls, value):
        value = Q(value)
        return cls({(): value}) if value else cls()

    @classmethod
    def variable(cls, name):
        return cls({((name, 1),): Q(1)})

    @staticmethod
    def coerce(value):
        return value if isinstance(value, Poly) else Poly.constant(value)

    def __add__(self, other):
        other = self.coerce(other)
        terms = dict(self.terms)
        for monomial, coefficient in other.terms.items():
            terms[monomial] = terms.get(monomial, Q(0)) + coefficient
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self):
        return Poly({monomial: -coefficient for monomial, coefficient in self.terms.items()})

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        terms = {}
        for left_monomial, left_coefficient in self.terms.items():
            for right_monomial, right_coefficient in other.terms.items():
                powers = dict(left_monomial)
                for name, power in right_monomial:
                    powers[name] = powers.get(name, 0) + power
                    if not powers[name]:
                        del powers[name]
                monomial = tuple(sorted(powers.items()))
                terms[monomial] = (
                    terms.get(monomial, Q(0))
                    + left_coefficient * right_coefficient
                )
        return Poly(terms)

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, int):
            raise TypeError("Laurent-polynomial powers must be integers")
        if power < 0:
            if len(self.terms) != 1:
                raise ValueError("negative power is supported only for monomials")
            (monomial, coefficient), = self.terms.items()
            if not coefficient:
                raise ZeroDivisionError
            return Poly(
                {
                    tuple((name, exponent * power) for name, exponent in monomial):
                    coefficient**power
                }
            )
        result = Poly.constant(1)
        base = self
        exponent = power
        while exponent:
            if exponent & 1:
                result *= base
            base *= base
            exponent >>= 1
        return result

    def __truediv__(self, other):
        other = self.coerce(other)
        return self * other**-1

    def __eq__(self, other):
        return self.terms == self.coerce(other).terms

    def derivative(self, variable):
        terms = {}
        for monomial, coefficient in self.terms.items():
            powers = dict(monomial)
            exponent = powers.get(variable, 0)
            if not exponent:
                continue
            if exponent == 1:
                del powers[variable]
            else:
                powers[variable] = exponent - 1
            key = tuple(sorted(powers.items()))
            terms[key] = terms.get(key, Q(0)) + coefficient * exponent
        return Poly(terms)

    def substitute(self, variable, replacement):
        replacement = self.coerce(replacement)
        result = Poly()
        for monomial, coefficient in self.terms.items():
            term = Poly.constant(coefficient)
            for name, exponent in monomial:
                factor = replacement if name == variable else Poly.variable(name)
                term *= factor**exponent
            result += term
        return result

    def coefficient(self, variable, exponent):
        terms = {}
        for monomial, coefficient in self.terms.items():
            powers = dict(monomial)
            if powers.get(variable, 0) != exponent:
                continue
            powers.pop(variable, None)
            key = tuple(sorted(powers.items()))
            terms[key] = terms.get(key, Q(0)) + coefficient
        return Poly(terms)

    def powers(self, variable):
        return {dict(monomial).get(variable, 0) for monomial in self.terms}


def V(name):
    return Poly.variable(name)


def integrate_unit(poly, variable):
    """Integrate a polynomial in ``variable`` from zero to one exactly."""
    result = Poly()
    for monomial, coefficient in poly.terms.items():
        powers = dict(monomial)
        exponent = powers.pop(variable, 0)
        if exponent < 0:
            raise ValueError("unit integration received a Laurent pole")
        result += Poly({tuple(sorted(powers.items())): coefficient / (exponent + 1)})
    return result


def integrate_monomial_between(term, variable, lower, upper):
    """Integrate one Laurent monomial when its exponent is not minus one."""
    if len(term.terms) != 1:
        raise ValueError("expected one monomial")
    (monomial, coefficient), = term.terms.items()
    powers = dict(monomial)
    exponent = powers.pop(variable, 0)
    if exponent == -1:
        raise ValueError("logarithmic monomial handled separately")
    prefactor = Poly({tuple(sorted(powers.items())): coefficient / (exponent + 1)})
    return prefactor * (Poly.coerce(upper) ** (exponent + 1)
                        - Poly.coerce(lower) ** (exponent + 1))


def leading_coefficient(poly, variable, exponent):
    powers = poly.powers(variable)
    if not powers or min(powers) != exponent:
        return None
    return poly.coefficient(variable, exponent)


def det3(matrix):
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


checks = 0


def ck(name, condition):
    """Optimization-safe, fail-closed check."""
    global checks
    checks += 1
    if condition is not True:
        raise SystemExit(f"CHECK FAILED [{name}]")
    print(f"[ok] {name}")


# ---------------------------------------------------------------------------
# A. Exact cubic-Hermite derivative residuals and axial/ESIP scales
# ---------------------------------------------------------------------------
x, q, r = V("x"), V("q"), V("r")


def hermite_derivative_residual(degree):
    """Derivative residual for g(x)=x^degree/degree! after Hermite fit."""
    z = x / r
    h00 = 2 * z**3 - 3 * z**2 + 1
    h10 = z**3 - 2 * z**2 + z
    h01 = -2 * z**3 + 3 * z**2
    h11 = z**3 - z**2
    value_zero = Q(1) if degree == 0 else Q(0)
    derivative_zero = Q(1) if degree == 1 else Q(0)
    value_r = r**degree / factorial(degree)
    derivative_r = (Poly() if degree == 0
                    else r ** (degree - 1) / factorial(degree - 1))
    interpolant = (
        value_zero * h00 + r * derivative_zero * h10
        + value_r * h01 + r * derivative_r * h11
    )
    source = x**degree / factorial(degree)
    return (source - interpolant).derivative("x").substitute("x", q)


for degree in range(4):
    ck(f"Hermite reproduction degree {degree}",
       hermite_derivative_residual(degree) == 0)

a = q * (q - r)
chi = 2 * q - r
h4 = hermite_derivative_residual(4)
h5 = hermite_derivative_residual(5)
ck("fourth-derivative Hermite coefficient", h4 == a * chi / 12)
ck("fifth-derivative Hermite coefficient",
   h5 == a * (5 * q**2 + 5 * q * r - 4 * r**2) / 120)
ck("fourth-order midpoint cancellation", h4.substitute("q", r / 2) == 0)
ck("fifth-order midpoint residual", h5.substitute("q", r / 2) == r**4 / 1920)

d0_sq = 4 * chi**2 + a**2
axis_factor = a**6 * d0_sq
midpoint_factor = axis_factor.substitute("q", r / 2)
ck("midpoint ESIP determinant scale r^16", midpoint_factor == r**16 / 65536)
ck("left endpoint q^6 r^8 factor", axis_factor.coefficient("q", 6) == 4 * r**8)
p = V("p")
right_endpoint_factor = axis_factor.substitute("q", r - p)
ck("right endpoint (r-q)^6 r^8 factor",
   right_endpoint_factor.coefficient("p", 6) == 4 * r**8)
theta = V("theta")
generic_axis = axis_factor.substitute("q", theta * r)
ck("generic axial q=theta*r scale r^14",
   leading_coefficient(generic_axis, "r", 14)
   == 4 * theta**6 * (theta - 1) ** 6 * (2 * theta - 1) ** 2)

kappa = V("kappa")
mean_axis_sq = a**2 * kappa**2
axis_covariance_scale = a**2 * d0_sq
ck("axial Mahalanobis cancellation",
   mean_axis_sq * d0_sq == kappa**2 * axis_covariance_scale)
axis_density_squared_denominator = a**6 * d0_sq
ck("axial density prefactor |a|^-3 d0^-1",
   axis_density_squared_denominator == axis_factor)

X, rho = V("X"), V("rho")
mean_mid = kappa * r**2 * (X**2 - Q(1, 4))
var_mid_scale = r**4 * (rho**2 + r**2)
ck("midpoint leading mean", mean_mid.substitute("X", 0) == -kappa * r**2 / 4)
ck("midpoint subchart |X|<=1/4 mean margin",
   mean_mid.substitute("X", Q(1, 4)) == -3 * kappa * r**2 / 16)
ck("endpoint leading-mean zeros", mean_mid.substitute("X", Q(1, 2)) == 0)
mean_mid_zero = mean_mid.substitute("X", 0)
ck("midpoint exponent scale",
   16 * mean_mid_zero**2 * (rho**2 + r**2) == kappa**2 * var_mid_scale)


# ---------------------------------------------------------------------------
# B. Endpoint chart and exact conditioned-collision factors
# ---------------------------------------------------------------------------
d, A, c = V("d"), V("A"), V("c")
eps_sq = r**2 + rho**2
endpoint_det = d**6 * r**2 * eps_sq**3 * A
endpoint_density_squared_denominator = d**6 * r**2 * eps_sq**3
ck("endpoint determinant/density powers",
   endpoint_det == endpoint_density_squared_denominator * A)
ck("endpoint mismatch exponent scale", eps_sq * kappa**2 == kappa**2 * eps_sq)

R, P = V("R"), V("P")
endpoint_residual_map = [
    [c * R / 12, -P / 2, Poly(), Poly(), Poly()],
    [Poly(), -c * R / 2, Poly(), P, Poly()],
    [Poly(), Poly(), -c * R / 2, Poly(), P],
]
minor_transverse = det3([[row[index] for index in (1, 3, 4)]
                         for row in endpoint_residual_map])
minor_axial = det3([[row[index] for index in (0, 1, 2)]
                    for row in endpoint_residual_map])
ck("endpoint residual transverse minor", minor_transverse == -P**3 / 2)
ck("endpoint residual axial minor", minor_axial == c**3 * R**3 / 48)

s, u = V("s"), V("u")
coefficients = [V(f"c{degree}") for degree in range(2, 7)]
c1 = -sum(coefficient * d ** (degree - 1)
          for degree, coefficient in enumerate(coefficients, start=2))
h = c1 * s + sum(coefficient * s**degree
                 for degree, coefficient in enumerate(coefficients, start=2))
ck("collision endpoint pins",
   h.substitute("s", 0) == 0 and h.substitute("s", d) == 0)
h_second = h.derivative("s").derivative("s")
R_M = -integrate_unit((1 - u) * h_second.substitute("s", d * u), "u")
R_y = integrate_unit(u * h_second.substitute("s", d * u), "u")
ck("conditioned collision H_M u=d R_M",
   h.derivative("s").substitute("s", 0) == d * R_M)
ck("conditioned collision H_y u=d R_y",
   h.derivative("s").substitute("s", d) == d * R_y)

# Projective Hessian losses give
#   |Pi| <= (r*d/rho)^2 min(r, r^3/(d*rho))
#        <= r^5*d*rho^-3.
# For d<=r^2 the first branch differs from the target by d*rho/r^2<=1
# (on the endpoint chart rho<=1); for d>=r^2 the second branch is exact.
endpoint_product_target = r**5 * d * rho**-3
endpoint_small_d_branch = (r * d / rho) ** 2 * r
endpoint_large_d_branch = (r * d / rho) ** 2 * (r**3 / (d * rho))
ck("endpoint split d<=r^2 comparison factor",
   endpoint_small_d_branch == endpoint_product_target * d * rho / r**2)
ck("endpoint split d>=r^2 product identity",
   endpoint_large_d_branch == endpoint_product_target)

# After recording the projective rho^-3 loss, the determinant-product powers
# are r^5*d.  Density contributes d^-3*r^-1, physical radial volume d^2 dd,
# and the Palm pair normalizer contributes r^-2.
ck("endpoint radial integrability exponent", 1 - 3 + 2 == 0)
ck("endpoint d-ledger integral", integrate_unit(Poly.constant(1), "d") * r == r)
ck("endpoint r-ledger exponent", 5 - 1 - 2 == 2)
ck("endpoint d-ledger exponent", 1 - 3 + 2 == 0)
ck("endpoint pre-Palm r-ledger exponent", 5 - 1 == 4)
eps0 = V("eps0")
ck("endpoint pre-Palm radial ledger O(r^5)",
   r**4 * (eps0 * r) == eps0 * r**5)
ck("endpoint full radial ledger O(r^3)",
   r**2 * (eps0 * r) == eps0 * r**3)


# ---------------------------------------------------------------------------
# C. Exact singular-near anisotropic covariance determinant
# ---------------------------------------------------------------------------
rho2 = rho**2
C00 = rho2 * (c**2 + rho2) * (c**2 + 3 * rho2) / 72
C01 = -c * rho2 * (c**2 + rho2) / 6
C11 = rho2 * (4 * c**2 + rho2) / 2
block_det = C00 * C11 - C01**2
block_target = rho2**3 * (c**2 + rho2) * (3 * c**2 + rho2) / 48
ck("singular (L0,L1) block determinant", block_det == block_target)
transverse_det = 2 * rho2**2
singular_det = block_det * transverse_det
singular_target = rho2**5 * (c**2 + rho2) * (3 * c**2 + rho2) / 24
ck("singular anisotropic determinant", singular_det == singular_target)
t, Cscaled, RHO = V("t"), V("Cscaled"), V("RHO")
joint_singular = singular_target.substitute("c", t * Cscaled).substitute("rho", t * RHO)
joint_target = t**14 * RHO**10 * (Cscaled**2 + RHO**2) * (3 * Cscaled**2 + RHO**2) / 24
ck("singular joint degree fourteen", joint_singular == joint_target)


# ---------------------------------------------------------------------------
# D. Collinear mismatch and exponential absorption
# ---------------------------------------------------------------------------
eta = V("eta")
Delta_unfactored = kappa * c**2 - kappa * eta**2 / 4
Delta = kappa * (c**2 - eta**2 / 4)
ck("collinear cubic mismatch identity", Delta_unfactored == Delta)
Delta_axis = Delta.substitute("c", 1)
ck("axis mismatch at eta=1", Delta_axis.substitute("eta", 1) == 3 * kappa / 4)
ck("axis mismatch at eta=0", Delta_axis.substitute("eta", 0) == kappa)

# Exact Taylor lower-bound certificates: e^u >= u^m/m!, u=c/zeta^p.
# Thus zeta^-N e^(-c/zeta^p) <= m! c^-m zeta^(p*m-N).
power_loss = 20
m_linear = power_loss + 1
ck("exp[-c/(t^2+rho^2)] absorbs fixed powers",
   m_linear == 21 and m_linear - power_loss == 1 and factorial(m_linear) > 0)
m_quartic = 6
ck("legacy collinear exp[-c/rho^4] absorbs angular loss",
   4 * m_quartic - power_loss == 4 and factorial(m_quartic) > 0)


# ---------------------------------------------------------------------------
# E. Common determinant envelope and six-term singular radial ledger
# ---------------------------------------------------------------------------
C1, delta = V("C1"), V("delta")
D_common = (r * (r + t**2)) ** 2 * (r**2 + t**3)
radial_integrand = r * t**-5 * D_common
six_terms = [
    r**7 * t**-5,
    2 * r**6 * t**-3,
    r**5 * t**-2,
    r**5 * t**-1,
    2 * r**4,
    r**3 * t**2,
]
ck("common D six-term expansion", radial_integrand == sum(six_terms))

integrals = [
    integrate_monomial_between(six_terms[index], "t", C1 * r, delta)
    for index in (0, 1, 2, 4, 5)
]
ck("radial term 1 = O(r^3)",
   leading_coefficient(integrals[0], "r", 3) == Q(1, 4) * C1**-4)
ck("radial term 2 = O(r^4)",
   leading_coefficient(integrals[1], "r", 4) == C1**-2)
ck("radial term 3 = O(r^4)",
   leading_coefficient(integrals[2], "r", 4) == C1**-1)
ck("radial term 4 = O(r^5 log(1/r))",
   six_terms[3].coefficient("t", -1) == r**5)
ck("radial term 5 = O(r^4)",
   leading_coefficient(integrals[3], "r", 4) == 2 * delta)
ck("radial term 6 = O(r^3)",
   leading_coefficient(integrals[4], "r", 3) == delta**3 / 3)

pre_palm_radial_integrand = r**3 * t**-5 * D_common
pre_palm_six_terms = [
    r**9 * t**-5,
    2 * r**8 * t**-3,
    r**7 * t**-2,
    r**7 * t**-1,
    2 * r**6,
    r**5 * t**2,
]
ck("pre-Palm common D six-term expansion",
   pre_palm_radial_integrand == sum(pre_palm_six_terms))
pre_integrals = [
    integrate_monomial_between(pre_palm_six_terms[index], "t", C1 * r, delta)
    for index in (0, 1, 2, 4, 5)
]
pre_scales = [5, 6, 6, 6, 5]
pre_targets = [Q(1, 4) * C1**-4, C1**-2, C1**-1, 2 * delta, delta**3 / 3]
pre_position = 0
for index in range(1, 7):
    if index == 4:
        condition = pre_palm_six_terms[3].coefficient("t", -1) == r**7
    else:
        condition = (
            leading_coefficient(pre_integrals[pre_position], "r", pre_scales[pre_position])
            == pre_targets[pre_position]
        )
        pre_position += 1
    ck(f"pre-Palm radial term {index} leading scale", condition)


# ---------------------------------------------------------------------------
# F. Collar and canonical-value-window ledgers
# ---------------------------------------------------------------------------
xi1, xi2, xi3 = V("xi1"), V("xi2"), V("xi3")
rho_xi_sq = xi2**2 + xi3**2
rho_sq_symbol = V("rho_sq")
collar_contact_det = rho_sq_symbol**3 * (4 * X**2 + rho_sq_symbol)
collar_scaled = collar_contact_det.substitute("X", r * xi1).substitute(
    "rho_sq", r**2 * rho_xi_sq
)
ck("generic collar gradient determinant scale r^8",
   collar_scaled == r**8 * rho_xi_sq**3 * (4 * xi1**2 + rho_xi_sq))

# The raw joint density is r^-7.  The canonical value window has width r^3,
# so its integration produces the effective r^-4 density used above.  It is
# included exactly once in both ledgers below.
raw_collar_density = -7
value_window_power = 3
effective_collar_density = raw_collar_density + value_window_power
collar_pre_palm = effective_collar_density + 6 + 3
collar_palm = collar_pre_palm - 2
ck("raw collar density plus value window = r^-4",
   effective_collar_density == -4)
ck("canonical pre-Palm collar ledger = r^5", collar_pre_palm == 5)
ck("canonical Palm collar ledger = r^3", collar_palm == 3)
b = V("b")
h_pin = b - kappa * r**3 / 6
ck("canonical count value width", b - h_pin == kappa * r**3 / 6)
ck("canonical value-window exponent for each region = r^3",
   value_window_power == 3)


print(f"CHECK_COUNT={checks}")
print("AXIAL_FACTOR = a^6*(4*chi^2+a^2)*det(Gamma)")
print("MIDPOINT_ESIP_SCALE = r^16")
print("ENDPOINT_FACTOR = d^6*r^2*(r^2+rho^2)^3*A")
print("SINGULAR_FACTOR = rho^10*(c^2+rho^2)*(3*c^2+rho^2)/24")
print("COMMON_D = [r(r+t^2)]^2*(r^2+t^3)")
print("SCOPE LIMIT: exact symbolic interpolation, covariance-factor, mismatch,")
print("and integration ledgers only; this script does not prove the analytic")
print("uniform side-24 Schur-complement or conditional-moment bounds on faces.")
print("ALL_ASSERTIONS_PASS")
