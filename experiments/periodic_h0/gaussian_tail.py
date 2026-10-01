"""Exact arithmetic for a specified ideal Gaussian Fourier tail, not RNG certification."""
from fractions import Fraction as Q
from functools import lru_cache
import finite_certificate as fc

EXP_BITS = 256


def _outward(lo, hi):
    scale = 1 << EXP_BITS
    a, b = lo*scale, hi*scale
    return Q(a.numerator//a.denominator, scale), Q(-((-b.numerator)//b.denominator), scale)


@lru_cache(maxsize=2048, typed=True)
def exp_neg_bounds(x):
    """Enclose exp(-x); alternating Taylor then outward rounded squaring."""
    fc.require(type(x) is Q and 0 <= x <= 4096,
               'Rational exponential argument in [0,4096] required')
    if x == 0:
        return Q(1), Q(1)
    y, shifts = x, 0
    while y > Q(1,8):
        y /= 2
        shifts += 1
    term, total = Q(1), Q(1)
    for j in range(1,50):
        term *= -y/j
        total += term
    # P49 is lower; P48 = P49 - term49 is upper. Terms decrease at y<=1/8.
    lo, hi = _outward(total, total-term)
    for _ in range(shifts):
        lo, hi = _outward(lo*lo, hi*hi)
    return lo, hi


def theta_tail_upper(c_lower, cutoff):
    """Two-sided sum over |j|>cutoff; c_lower is positive and no larger than c."""
    fc.require(type(c_lower) is Q and c_lower > 0, 'Positive rational decay required')
    fc.require(type(cutoff) is int and cutoff >= 0, 'Nonnegative integer cutoff required')
    first = exp_neg_bounds(c_lower*(cutoff+1)**2)[1]
    ratio = exp_neg_bounds(c_lower*(2*cutoff+3))[1]
    fc.require(ratio < 1, 'Decay too small for this precision')
    return 2*first/(1-ratio)


def _theta_partial(c_lo, c_hi, cutoff):
    low = high = Q(1)
    for j in range(1,cutoff+1):
        low += 2*exp_neg_bounds(c_hi*j*j)[0]
        high += 2*exp_neg_bounds(c_lo*j*j)[1]
    return low, high


def tail_bound(cutoff, threshold):
    """Side24 planar ideal-field shell event; fixed sum through M=64."""
    fc.require(type(cutoff) is int and 1 <= cutoff <= 64, 'Integer cutoff in [1,64] required')
    fc.require(type(threshold) is Q and 1 <= threshold <= 16,
               'Rational Rayleigh threshold in [1,16] required')
    pi_lo, pi_hi = fc.pi_bounds()
    beta_lo, beta_hi = (pi_lo/24)**2, (pi_hi/24)**2
    alpha_lo, alpha_hi = 2*beta_lo, 2*beta_hi
    a_lo, a_hi = _theta_partial(alpha_lo,alpha_hi,64)
    a_tail = theta_tail_upper(alpha_lo,64)
    b_lo, b_hi = _theta_partial(beta_lo,beta_hi,64)
    b_tail = theta_tail_upper(beta_lo,64)
    # Half-plane shell sum equals exp(-beta*m²)(S_beta,m+S_beta,m-1).
    b_partial = [Q(1)]
    for j in range(1,65):
        b_partial.append(b_partial[-1]+2*exp_neg_bounds(beta_lo*j*j)[1])
    finite = sum(((threshold+m-cutoff-1)*exp_neg_bounds(beta_lo*m*m)[1]
                  *(b_partial[m]+b_partial[m-1]) for m in range(cutoff+1,65)),Q(0))
    q = exp_neg_bounds(beta_lo*131)[1]
    remainder = 2*(b_hi+b_tail)*exp_neg_bounds(beta_lo*65**2)[1]*(
        (threshold+64-cutoff)/(1-q)+q/(1-q)**2)
    factor = fc.sqrt_bounds(Q(2))[1]/a_lo
    r = exp_neg_bounds(threshold)[1]
    failure = 4*exp_neg_bounds(threshold**2/2)[1]*(
        (cutoff+1)/(1-r)+r/(1-r)**2)
    tau = _outward(Q(0),factor*(finite+remainder))[1]
    eta = min(Q(1),_outward(Q(0),failure)[1])
    loss = _outward(Q(0),a_tail/a_lo)[1]
    return {'side':24, 'dimension':2, 'cutoff':cutoff, 'threshold':str(threshold),
            'summation_cutoff':64, 'tail_supremum_upper':str(tau),
            'tail_supremum_decimal_up':fc.decimal_up(tau,24),
            'failure_probability_upper':str(eta),
            'failure_probability_decimal_up':fc.decimal_up(eta,24),
            'normalization_loss_upper':str(loss),
            'normalization_loss_decimal_up':fc.decimal_up(loss,76),
            'finite_shell_sum_upper':str(finite),
            'infinite_remainder_upper':str(remainder),
            'theta_alpha_64_interval':[str(a_lo),str(a_hi)],
            'theta_alpha_infinite_interval':[str(a_lo),str(a_hi+a_tail)],
            'theta_beta_infinite_interval':[str(b_lo),str(b_hi+b_tail)],
            'theta_alpha_tail_upper':str(a_tail),
            'theta_beta_tail_upper':str(b_tail)}


def total_error(finite_error, tail, coupling_error, normalization_loss, polynomial_norm):
    """Conditional deterministic sum. None means a missing premise, never zero."""
    for value in (finite_error,tail,normalization_loss,polynomial_norm):
        fc.require(type(value) is Q and value >= 0, 'Nonnegative rational budget required')
    if coupling_error is None:
        return None
    fc.require(type(coupling_error) is Q and coupling_error >= 0,
               'Nonnegative rational coupling error required')
    return finite_error+tail+coupling_error+normalization_loss*polynomial_norm
