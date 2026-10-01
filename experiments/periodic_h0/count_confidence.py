"""Exact arithmetic for conditional, fixed-sample mean confidence bounds.

COUNT_CONFIDENCE.md proves the statistical statement. These functions do not
authenticate an IID source, a grid certificate, or a prospective sampling plan.
"""
from fractions import Fraction as Q
import finite_certificate as fc
import finite_count_loss as cl
from gaussian_tail import exp_neg_bounds


def _positive_integer(value):
    fc.require(type(value) is int and value>0,'Positive integer required')
    return value


def _radius(value):
    fc.require(type(value) is Q and value>0,'Positive exact rational radius required')
    return value


def risk_upper(cutoff, sample_count, bins, radius):
    """Upper bound min(1,2*m*exp(-2*n*(r/C)^2)), C=16*K^2.

    n, bins and radius must be chosen before seeing the sample. Dependence
    between bins is allowed; the n latent word blocks must be IID. Large
    exponents are conservatively capped for the inherited interval evaluator.
    """
    cap=cl.bar_cap(cutoff)
    n=_positive_integer(sample_count);m=_positive_integer(bins);r=_radius(radius)
    exponent=2*n*(r/cap)**2
    return min(Q(1),2*m*exp_neg_bounds(min(exponent,Q(4096)))[1])


def mean_interval(cutoff, failure, rows, radius, sample_count):
    """Conditional target-mean interval from one predeclared bin's rows.

    Each pair must actually enclose the corresponding fixed latent polynomial
    observables as in the proof. Only the numeric domain is checked here.
    None denotes an unresolved grid and contributes [0,C] to the original n.
    The returned interval has the risk bound from risk_upper only under its
    statistical premises. Neither an arithmetic row nor this output is data
    authentication or a test of the IID assumption.
    """
    cap=cl.bar_cap(cutoff);n=_positive_integer(sample_count);r=_radius(radius)
    fc.require(type(failure) is Q and 0<=failure<=1,'Exact probability bound in [0,1] required')
    fc.require(type(rows) is list and len(rows)==n,'Every planned draw must have a row')
    low=high=0
    for row in rows:
        if row is None:
            lo,hi=0,cap
        else:
            fc.require(type(row) in (list,tuple) and len(row)==2,'A lower/upper count pair is required')
            lo,hi=row
            fc.require(type(lo) is int and type(hi) is int and 0<=lo<=hi<=cap,
                       'Ordered integer count bounds within the cap required')
        low+=lo;high+=hi
    mu_lo=max(Q(0),Q(low,n)-r)
    mu_hi=min(Q(cap),Q(high,n)+r)
    return cl.expectation_interval(cutoff,failure,mu_lo,mu_hi)
