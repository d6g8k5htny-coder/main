"""Exact scalar continuation of the frozen fixed-axis H3 analytic proof.

All-small-r conclusion depends on the explicitly cited analytic sector,
regression, transport and determinant arguments, not on a sampled r grid.
This file certifies their numerical sufficient conditions by a separate
rational construction; it does not replay the original repository wrapper.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from h3_scalar_arithmetic import I, sqrt, sf, cdf, pdf, outward_decimal


def need(condition,message):
    if not condition: raise ValueError(message)


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()


def hermite(n):
    p=[1]; q=[0,1]
    if n==0:return p
    for k in range(1,n):
        a=[0]+q
        for j,v in enumerate(p): a[j]-=k*v
        p,q=q,a
    return q


def moments():
    # e^12 > 100000 by its positive Taylor partial sum; hence e^-288<10^-120.
    term=F(1); partial=term
    for k in range(1,40): term*=F(12,k); partial+=term
    need(partial>100000,'elementary exponential comparison')
    eps=F(1,10**100)
    errors={}
    for n in (0,2,4,6,8):
        coeff=hermite(n); c=sum(abs(a) for a in coeff)
        h24=sum(a*24**k for k,a in enumerate(coeff))
        # |H_n(24j)| <= c |24j|^n, j>=2.
        # Successive envelope ratio <= (3/2)^n exp(-1440)<(3/2)^n 10^-600.
        ratio=F(3,2)**n/F(10**600)
        tail=2*c*48**n*F(1,10**480)/(1-ratio)
        errors[n]=2*abs(h24)*F(1,10**120)+tail
    z=I(1,1+errors[0])
    m={}
    for n,central in ((2,1),(4,3),(6,15),(8,105)):
        exact_range=I(central-errors[n],central+errors[n])/z
        # Arithmetic rounds at 2^-256, so use direct rational comparison for
        # the much smaller analytic 1e-100 kernel perturbation first.
        need((F(central)-errors[n])/(1+errors[0]) >= F(central)-eps,'normalized lower tail')
        need(F(central)+errors[n] <= F(central)+eps,'normalized upper tail')
        m[n]=I(F(central)-eps,F(central)+eps)
    return m,{'exp12_partial_gt_100000':True,'kernel_normalizer':z.encode(),
              'normalized_moment_error':str(eps),'raw_derivative_errors':{str(n):str(v) for n,v in errors.items()}}


def gram_and_energy(m,R):
    shift=F(3,80)
    # Shifted two nontrivial limiting covariance blocks and two singleton blocks.
    p0=I(1)-shift; p1=m[4]-shift-(m[2]**2)/p0
    p2=m[2]-shift; p3=m[6]/144-shift-(m[4]/12)**2/p2
    pivots=[p0,p1,p2,p3,m[2]-shift,m[2]**2-shift]
    need(all(x.lo>0 for x in pivots),'strict limiting Gram eigenfloor')
    transport2=m[2]/4+m[4]/4+m[2]**2/4+m[6]/16+m[4]*m[2]/16+m[8]/4096
    need(transport2.hi<F(31,20)**2,'transport budget')
    need(F(193,1000)**2<shift,'singular-value floor')
    q0=F(36,25)*m[4]/(m[4]-m[2]**2)+4*m[2]/(m[2]*m[6]-m[4]**2)
    need(q0.hi<F(17,6),'limiting pin-energy budget')
    gap=1-F(31,20)*R/F(193,1000)
    need(gap>0,'uniform covariance gap')
    Q=F(17,6)/(gap*gap)
    return Q,{'shifted_gram_pivots':[x.encode() for x in pivots],
              'transport_squared':transport2.encode(),'limiting_pin_energy':q0.encode(),
              'uniform_energy_upper':str(Q),'radius_max':str(R)}


def prove(epsilon=F(103,500),gap_half=F(9,100),radius=F(1,20),target=F(1747,1000),
          axial_tails=F(2),difference_tails=F(2),variance_loss=F(1,4),beta_scale=F(1,4)):
    for x in (epsilon,gap_half,radius,target,axial_tails,difference_tails,variance_loss,beta_scale): need(type(x) is F,'exact Fraction parameters required')
    need(0<radius<=F(1,20),'fixed-axis band radius must be in (0,1/20]')
    need(0<epsilon<1 and 0<gap_half<1 and target>0,'invalid optimization parameters')
    need(axial_tails>=2 and difference_tails>=2,'both event unions retain two tails')
    need(variance_loss>=F(1,4) and beta_scale>=F(1,4),'variance majorants cannot be understated')
    m,tail= moments(); Q,energy=gram_and_energy(m,radius)
    axial_argument=epsilon/(radius*sqrt(m[8])/12)-sqrt(I(Q))
    need(axial_argument.lo>0,'axial threshold exceeds conditional mean')
    pA=1-axial_tails*sf(axial_argument)
    sigma2=m[4]-m[2]**2
    variance=sigma2*I(1-variance_loss*m[2].hi*radius**2,1)
    mu=I(m[2].lo*(F(6,5)-radius**3/12)-gap_half,m[2].hi*F(6,5)-gap_half)
    need(variance.lo>0 and mu.lo>0,'positive midpoint parameters')
    sd=sqrt(variance); t=mu/sd; phi=pdf(t); Phi=cdf(t)
    M1=sd*phi+mu*Phi
    M2=(variance+mu**2)*Phi+mu*sd*phi
    delta_mean=m[2]*radius**3/6
    delta_sd=radius*sqrt(sigma2*m[2])
    delta_argument=(2*gap_half-delta_mean)/delta_sd
    need(delta_argument.lo>0,'two-sided difference threshold exceeds its mean')
    pD=1-difference_tails*sf(delta_argument)
    need(pA.lo>0 and pD.lo>0,'positive event probability floors')
    a=1-epsilon; beta=beta_scale*m[4]*m[2]
    bracket=a*a*M2-a*radius*beta*M1
    need(bracket.lo>0,'positive budget BEFORE lower probability multiplication')
    lower=pA*pD*bracket
    need(lower.lo>=target,'requested target not certified by this sufficient estimate')
    values={'moments':{str(k):v.encode() for k,v in m.items()},'tail':tail,'energy':energy,
            'axial_argument':axial_argument.encode(),'p_axial':pA.encode(),
            'midpoint_variance':variance.encode(),'midpoint_mean':mu.encode(),
            'positive_first_moment':M1.encode(),'positive_second_moment':M2.encode(),
            'difference_argument':delta_argument.encode(),'p_difference':pD.encode(),
            'beta':beta.encode(),'positive_bracket':bracket.encode(),
            'derived_lower_coefficient':lower.encode(),'derived_outward_decimal':outward_decimal(lower)}
    return {'schema':'h3-scalar-candidate-v1','status':'AUTHOR_SIDE_PROVED_CANDIDATE',
            'parameters':{'epsilon':str(epsilon),'gap_half':str(gap_half),'radius_max':str(radius),'axial_tails':str(axial_tails),'difference_tails':str(difference_tails),
                          'variance_loss':str(variance_loss),'beta_scale':str(beta_scale)},
            'conclusion':{'coefficient':str(target),'statement':'Z(r) >= coefficient * r^2 for every 0<r<=radius_max',
                          'fixed_geometry':'2D normalized SIDE24; x axis; b=6/5; gap=r^3/6; canonical six-pin Gaussian law',
                          'endpoint_at_rmax':str(target*radius**2)},
            'values':values,'scientific_status_changed':False,'organizational_independence_credit':0,
            'review':'OPEN; source-exposed alternate-method author-side calculation',
            'limits':['Not an all-angle result; no full RN cover, selected-event transfer, q0 or prize closure.',
                      'The previous sharper fixed-radius RN floor is retained, not replaced.',
                      'Original full repository wrapper not replayed; analytic proof dependencies are explicit.',
                      'Exact scalar certificate plus the supplied analytic derivation, not failed sampling, warrants the candidate.']}
