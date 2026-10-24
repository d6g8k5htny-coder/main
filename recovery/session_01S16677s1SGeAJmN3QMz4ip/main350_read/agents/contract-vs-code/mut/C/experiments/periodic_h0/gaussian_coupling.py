"""A prospective finite-bit Gaussian coupling under an explicit IID input model."""
from fractions import Fraction as Q
from functools import lru_cache
import finite_certificate as fc
import gaussian_tail as gt

WORD_BITS=128
QUANTILE_BITS=64
WEIGHT_BITS=128
CLIP=Q(8)


def _round(q,bits):
    s=1<<bits
    v=q*s+Q(1,2)
    return Q(v.numerator//v.denominator,s)


@lru_cache(maxsize=1)
def _constants():
    pl,ph=fc.pi_bounds()
    sl=fc.sqrt_bounds(2*pl)[0];sh=fc.sqrt_bounds(2*ph)[1]
    el,eh=gt.exp_neg_bounds(CLIP**2/2)
    return sl,sh,el/sh,eh/sl


def precision_contract():
    """Uniform width bound for every CDF call, including directed term rounding."""
    ulp=Q(1,1<<gt.EXP_BITS)
    difference=2*ulp;total_difference=difference
    _,hi=gt._outward(CLIP,CLIP);total_hi=hi
    for n in range(1,192):
        ratio=CLIP**2/(2*n+1)
        difference=ratio*difference+2*ulp
        total_difference+=difference
        _,hi=gt._outward(Q(0),hi*ratio)
        total_hi+=hi
    ratio=CLIP**2/385;remainder=hi*ratio/(1-ratio)
    fc.require(total_hi+remainder<1<<50,'Positive series majorant exceeded')
    # For x²/2<=32, at most8 reductions; outward exp gap <2^-246.
    de=Q(1,1<<246)
    sl,sh,density_lower,_=_constants();dl,dh=1/sh,1/sl
    integral_gap=de*(1<<50)+total_difference+remainder
    width=(dh-dl)*(CLIP+integral_gap)+dl*integral_gap
    radius=width/density_lower
    fc.require(width<Q(1,1<<123) and 2*radius<Q(1,1<<65),
               'CDF precision cannot guarantee quantile termination')
    return {'cdf_width_upper':str(width),'ambiguous_radius_upper':str(radius),
            'series_rounding_width_upper':str(total_difference),
            'series_remainder_upper':str(remainder),'exponential_width_upper':str(de),
            'maximum_ordinary_halvings':69}


@lru_cache(maxsize=4096,typed=True)
def cdf_bounds(x):
    """Enclose Phi(x) on [-8,8] using a positive integral series."""
    fc.require(type(x) is Q and -CLIP<=x<=CLIP,'Rational CDF argument in [-8,8] required')
    if x<0:
        a,b=cdf_bounds(-x)
        return 1-b,1-a
    if x==0:return Q(1,2),Q(1,2)
    # int_0^x exp(-t²/2)dt = exp(-x²/2) sum_{n>=0} x^(2n+1)/(2n+1)!!.
    lo,hi=gt._outward(x,x);total_lo,total_hi=lo,hi
    for n in range(1,192):
        ratio=x*x/(2*n+1)
        lo,hi=gt._outward(lo*ratio,hi*ratio)
        total_lo+=lo;total_hi+=hi
    ratio=x*x/385
    remainder=hi*ratio/(1-ratio)
    el,eh=gt.exp_neg_bounds(x*x/2);sl,sh,_,_=_constants()
    return max(Q(0),Q(1,2)+el*total_lo/sh),min(Q(1),Q(1,2)+eh*(total_hi+remainder)/sl)


@lru_cache(maxsize=4096,typed=True)
def quantile(word):
    """Dyadic64 approximation to clip(Phi^-1((word+1/2)/2^128),[-8,8])."""
    fc.require(type(word) is int and 0<=word<1<<WORD_BITS,'Unsigned128-bit integer word required')
    u=Q(2*word+1,1<<(WORD_BITS+1))
    edge=cdf_bounds(CLIP)
    if u>=edge[1]:return CLIP
    if u<=1-edge[1]:return -CLIP
    lo,hi=-CLIP,CLIP
    density_lower=_constants()[2]
    for _ in range(80):
        if hi-lo<=Q(1,1<<(QUANTILE_BITS+1)):
            return _round((lo+hi)/2,QUANTILE_BITS)
        mid=(lo+hi)/2;a,b=cdf_bounds(mid)
        if u<a:hi=mid
        elif u>b:lo=mid
        else:
            radius=max(u-a,b-u)/density_lower
            lo=max(lo,mid-radius);hi=min(hi,mid+radius)
    raise ValueError('CDF precision did not resolve the quantile; no uncertified fallback')


def weights(cutoff):
    """Dyadic128 real cosine/sine weights and exact aggregate rounding bound."""
    fc.require(type(cutoff) is int and 1<=cutoff<=64,'Integer cutoff in [1,64] required')
    pl,ph=fc.pi_bounds();bl,bh=(pl/24)**2,(ph/24)**2
    al,ah=gt._theta_partial(2*bl,2*bh,64)
    rtlo,rthi=fc.sqrt_bounds(Q(2))
    low,high=1/ah,1/al;dc=_round((low+high)/2,WEIGHT_BITS)
    dc_error=max(abs(dc-low),abs(dc-high))
    aggregate_error=dc_error;weight_sum=high;rounded_sum=abs(dc)
    modes=[]
    for x in range(cutoff+1):
        for y in range(-cutoff,cutoff+1):
            if x==0 and y<=0:continue
            a=gt.exp_neg_bounds(bh*(x*x+y*y))[0]
            b=gt.exp_neg_bounds(bl*(x*x+y*y))[1]
            low,high=rtlo*a/ah,rthi*b/al
            w=_round((low+high)/2,WEIGHT_BITS)
            error=max(abs(w-low),abs(w-high))
            modes.append((x,y,w,error))
            aggregate_error+=2*error;weight_sum+=2*high;rounded_sum+=2*abs(w)
    return {'dc':dc,'dc_error':dc_error,'modes':modes,
            'aggregate_error':aggregate_error,'weight_sum_upper':weight_sum,
            'rounded_weight_sum':rounded_sum}


def budget(cutoff):
    """Uniform prospective error on max|Z_i|<=8; no historical-sample inference."""
    w=weights(cutoff);_,_,density_lower,density_upper=_constants()
    dz=Q(1,1<<(WORD_BITS+1))/density_lower+Q(1,1<<QUANTILE_BITS)
    n=(2*cutoff+1)**2
    rho=dz*w['weight_sum_upper']+CLIP*w['aggregate_error']
    failure=min(Q(1),2*n*density_upper/CLIP)
    norm=CLIP*w['rounded_weight_sum']
    return {'cutoff':cutoff,'real_gaussians':n,'clip':str(CLIP),
            'word_bits':WORD_BITS,'quantile_bits':QUANTILE_BITS,'weight_bits':WEIGHT_BITS,
            'gaussian_coordinate_error_upper':str(dz),
            'weight_sum_upper':str(w['weight_sum_upper']),
            'aggregate_weight_error_upper':str(w['aggregate_error']),
            'coefficient_error_upper':str(rho),
            'coefficient_error_decimal_up':fc.decimal_up(rho,30),
            'clipping_failure_upper':str(failure),
            'clipping_failure_decimal_up':fc.decimal_up(failure,24),
            'polynomial_norm_upper':str(norm),
            'polynomial_norm_decimal_up':fc.decimal_up(norm,24)}


def polynomial(cutoff,words):
    """Construct real-basis dyadic coefficients from explicit words; no RNG involved."""
    w=weights(cutoff)
    fc.require(type(words) is list and len(words)==(2*cutoff+1)**2,'Complete ordered word list required')
    z=[quantile(word) for word in words]
    modes=[(x,y,a*z[2*i+1],a*z[2*i+2]) for i,(x,y,a,_) in enumerate(w['modes'])]
    return {'dc':w['dc']*z[0],'modes':modes}
