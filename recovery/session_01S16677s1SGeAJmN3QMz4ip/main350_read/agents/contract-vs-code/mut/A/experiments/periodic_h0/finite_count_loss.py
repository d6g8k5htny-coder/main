"""Finite-polynomial H0 count budgets; probabilistic premises remain explicit.

The proof in FINITE_COUNT_LOSS.md supplies the uniform cap. These functions
evaluate its exact consequences, not a proof of an input source's random law.
"""
from fractions import Fraction as Q
import finite_certificate as fc


def bar_cap(cutoff):
    """Conservative total positive finite H0 bar cap for planar square cutoff K."""
    fc.require(type(cutoff) is int and cutoff>=1,'Positive integer cutoff required')
    return 16*cutoff*cutoff


def _rational(value):
    fc.require(type(value) is Q,'Exact Fraction input required')
    return value


def expectation_interval(cutoff, failure, contracted_mean_lower, expanded_mean_upper):
    """Conditional bounds from genuine same-ensemble expectation bounds.

    Inputs must bound the expectations of the observables in the proof, not
    sample means. The caller supplies those premises; this does not certify them.
    """
    cap=bar_cap(cutoff)
    p=_rational(failure);lo=_rational(contracted_mean_lower);hi=_rational(expanded_mean_upper)
    fc.require(0<=p<=1,'Failure probability bound must be in [0,1]')
    fc.require(0<=lo<=hi<=cap,'Ordered expectation bounds within the count cap required')
    loss=cap*p
    return max(Q(0),lo-loss),min(Q(cap),hi+loss)


def unresolved_grid(cutoff):
    """Keep an unresolved realization using [0,cap]; never discard the draw."""
    return 0,bar_cap(cutoff)


def grid_observables(cutoff, a, b, epsilon, rho, lifetimes):
    """Bounded bin observables for a separately certified finite-polynomial grid.

    The caller must certify d_B(D(P),Q)<=epsilon for this grid. rho is the
    same-cutoff coupling error on the good event. Lifetimes are positive exact
    finite grid lengths. On the good event L<=N_F([a,b))<=U; on every word
    0<=L<=U<=cap. A failed/missing grid certificate uses unresolved_grid instead.
    """
    cap=bar_cap(cutoff)
    a,b,epsilon,rho=map(_rational,(a,b,epsilon,rho))
    fc.require(0<a<b,'Positive half-open lifetime bin required')
    fc.require(epsilon>=0 and rho>=0,'Nonnegative error budgets required')
    fc.require(type(lifetimes) is list,'Explicit finite lifetime list required')
    fc.require(all(type(x) is Q and x>0 for x in lifetimes),'Positive rational lifetimes required')
    radius=2*(epsilon+rho)
    low=sum(a+radius<=x<b-radius for x in lifetimes) if a+radius<b-radius else 0
    # At equality a bar can match to the diagonal; the clean upper test is strict.
    high=min(cap,sum(a-radius<=x<b+radius for x in lifetimes)) if a>radius else cap
    fc.require(low<=high,'Count observables contradict the claimed polynomial/grid certificate')
    return low,high
