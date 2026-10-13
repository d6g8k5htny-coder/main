"""Finite rational identities for the punctured endpoint-disk candidate.

These helpers do not check Gaussian conditioning, analytic remainders, infinite
Fourier sums, Kac--Rice hypotheses, or the continuum theorem. Standard library.
"""
from fractions import Fraction
from typing import Sequence

F = Fraction
Rational = int | Fraction


def rational(value: Rational) -> Fraction:
    """Reject floating-point inputs so the finite controls remain exact."""
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("expected int or Fraction")
    return F(value)


def frame_rows(p: Rational, q: Rational, alpha: Rational, beta: Rational) -> list[list[Fraction]]:
    """The 2x4 leading row matrix, columns (f_xxxx,f_xxz,f_zz,f_xzz)."""
    p,q,alpha,beta = map(rational,(p,q,alpha,beta))
    if abs(p)>F(1,4) or abs(q)>F(1,4) or alpha*alpha+beta*beta!=1:
        raise ValueError("outside the rational compactified chart")
    a=(p-1)*(2*p-1)/12
    b=(p-1)/2
    c=p-F(1,2)
    return [[a*alpha, c*beta, F(0), q*beta/2],
            [F(0), b*alpha, beta, F(0)]]


def endpoint_frame(r: Rational, data: Sequence[Rational]) -> tuple[Fraction,...]:
    """Exact invertible six-pin transform; data=(g0,gp0,h0,gr,gpr,hr)."""
    r=rational(r)
    if r<=0 or len(data)!=6:
        raise ValueError("positive r and six observations required")
    g0,gp0,h0,gr,gpr,hr=map(rational,data)
    return (g0,gp0,h0,(gpr-gp0)/r,
            12*(gr-g0-r*(gpr+gp0)/2)/r**3,(hr-h0)/r)


def gradient_numerators(r: Rational, k: Rational, p: Rational, q: Rational,
                        jets: Sequence[Rational]) -> tuple[Fraction,Fraction]:
    """Exact (f_x/r^2,f_z/r) of the specified pinned quartic fixture only."""
    r,k,p,q=map(rational,(r,k,p,q))
    if r<=0 or len(jets)!=5:
        raise ValueError("positive r and five fixture jets required")
    a4,t,s,c,d=map(rational,jets)
    return (6*k*p*(p-1)+r*a4*p*(p-1)*(2*p-1)/12
            +t*q*(p-F(1,2))+c*q*q/2,
            r*t*p*(p-1)/2+q*s+r*p*q*c+r*q*q*d/2)


def relative_remainder(r: Rational, p: Rational) -> Fraction:
    """Derivative of x^2(x-r)^2(x+2r) at x=rp (not a remainder estimator)."""
    r,p=map(rational,(r,p))
    return r**4*p*(5*p**3-9*p+4)


def valid_point(r: Rational,p: Rational,q: Rational) -> bool:
    """Geometric chart membership only; q=0 is allowed off the pin."""
    r,p,q=map(rational,(r,p,q))
    return 0<r<=1 and q!=0 and 0<p*p+q*q<=F(1,16)


def scale_ledger() -> dict[str,int]:
    """Bookkeeping of the stated estimates, not evidence they hold."""
    normalizer_power = -2
    height_window_power = 0
    outer=normalizer_power-3+6+height_window_power
    axial=normalizer_power-3+3-2-6+height_window_power
    return {'outer_intensity':outer,'axial_prefactor':axial,
            'axial_after_sixth_term':axial+12,
            'fixed_scaled_area_count':outer+2,
            'nested_scaled_area_count':outer+4,
            'height_window_factors':int(height_window_power!=0),
            'endpoint_normalizers':int(normalizer_power==-2)}
