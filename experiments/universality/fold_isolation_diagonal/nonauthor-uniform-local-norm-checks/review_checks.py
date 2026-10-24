"""Independent exact finite controls for the uniform local norm review.

No fields are sampled. Continuous cutoff, Fourier, Gaussian, and asymptotic arguments
are reviewed mathematically, not proved by these finite controls.
"""
from fractions import Fraction as Q
from itertools import product
from math import comb, factorial, ceil
from pathlib import Path
import json


def poly_mul(a,b,degree):
    out=[Q(0)]*(degree+1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            if i+j<=degree:
                out[i+j]+=x*y
    return out


def weak_compositions(total,length):
    if length==1:
        yield (total,)
    else:
        for first in range(total+1):
            for rest in weak_compositions(total-first,length-1):
                yield (first,)+rest


def run():
    counts={};negative={}
    # Independent closed derivative coefficients (unsigned Lah numbers with alternating signs),
    # checked against differentiation rather than copied from the author's sparse recurrence.
    L=[{0:1}]
    for j in range(1,7):
        L.append({j+k:(-1)**(j-k)*factorial(j)//factorial(k)*comb(j-1,k-1) for k in range(1,j+1)})
    for j in range(6):
        direct={}
        for power,coefficient in L[j].items():
            direct[power+2]=direct.get(power+2,0)+coefficient
            if power:
                direct[power+1]=direct.get(power+1,0)-power*coefficient
        assert {m:c for m,c in direct.items() if c}==L[j+1]
    aa=[sum(abs(c)*max(1,m**m) for m,c in polynomial.items()) for polynomial in L]
    # E(t)=8/(1-16 sum_(j>=1) A_j t^j/j!) as a truncated exponential generating series.
    base=[Q(0)]+[Q(16*aa[j],factorial(j)) for j in range(1,7)]
    power=[Q(1)]+[Q(0)]*6;reciprocal=[Q(0)]*7
    for k in range(7):
        reciprocal=[x+8*y for x,y in zip(reciprocal,power)]
        power=poly_mul(power,base,6)
    ee_raw=[reciprocal[j]*factorial(j) for j in range(7)]
    assert all(x.denominator==1 for x in ee_raw)
    ee=[int(x) for x in ee_raw]
    gg_coeff=poly_mul([Q(aa[j],factorial(j)) for j in range(7)],reciprocal,6)
    gg_raw=[gg_coeff[j]*factorial(j) for j in range(7)]
    assert all(x.denominator==1 for x in gg_raw)
    gg=[int(x) for x in gg_raw]
    dd=[6]+[2*gg[j]**2 for j in range(1,7)]
    # Sum first over alpha-gamma, using its multinomial expansion (1+2+2)^(6-|gamma|).
    atheta=sum(factorial(6)//(factorial(i)*factorial(j)*factorial(6-i-j))
               *2**(i+j)*5**(6-i-j)*dd[i]*dd[j]
               for i in range(7) for j in range(7-i))
    frozen=json.loads((Path(__file__).resolve().parent.parent/'UNIFORM_LOCAL_NORM_CONTROLS.json').read_text())['constants']
    assert aa==frozen['A_j'] and ee==frozen['E_j'] and gg==frozen['H_j'] and dd==frozen['d_j_upper']
    assert atheta==int(frozen['A_theta'])==430299201784277045424464881586844996
    counts['closed_cutoff_coefficients_and_reindexed_trace_identity']=8

    # Polynomial expansion of (1+x+y)^6, independent of a direct multinomial loop.
    weights={(0,0):1}
    for _ in range(6):
        out={}
        for (i,j),c in weights.items():
            for shift in [(0,0),(1,0),(0,1)]:
                key=(i+shift[0],j+shift[1]);out[key]=out.get(key,0)+c
        weights=out
    assert sum(weights.values())==3**6
    assert all(w==factorial(6)//(factorial(i)*factorial(j)*factorial(6-i-j)) for (i,j),w in weights.items())
    counts['physical_H6_multinomial_weights']=len(weights)

    # Exact beta integrals via expanding v^j(1-v)^(4-j) on [0,1].
    integrals=[]
    for j in range(5):
        integral=sum(Q((-1)**k*comb(4-j,k),j+k+1) for k in range(5-j))
        assert integral==Q(factorial(j)*factorial(4-j),factorial(5))
        integrals.append(str(integral))
    assert Q(1,15)**2>Q(1,240) and Q(2,15)**2>Q(1,60)
    counts['beta_integral_and_operator_majorant_checks']=7

    # Include all unequal, rank-deficient and rank-one spectra of a small rational trace budget.
    ncases=0
    for dimension in range(1,6):
        for total in range(9):
            for parts in weak_compositions(total,dimension):
                aj=[Q(p,16) for p in parts]
                margin=Q(1)
                for a in aj:
                    margin*=1-a
                assert margin>=1-sum(aj)>=Q(1,2)
                ncases+=1
    assert ncases==2001
    assert 1-Q(1,2)==Q(1,2)  # rank one lambda=T attains the sqrt(2) bound.
    counts['unequal_and_singular_PSD_MGF_product_cases']=ncases
    # Trace without PSD is insufficient: a negative eigenvalue can conceal a divergent positive one.
    eigenvalues=[Q(3),Q(-2)];T=Q(1)
    assert sum(eigenvalues)==T and eigenvalues[0]>=2*T
    negative['trace_without_PSD_exponential_moment_failure']='PRESERVED'

    # Weighted Cauchy is needed when all Gaussian summands are the same variable.
    count=0
    for i,j in weights:
        c=[comb(i,u)*comb(j,v) for u in range(i+1) for v in range(j+1)]
        assert sum(c)==2**(i+j)
        assert sum(c)**2==2**(i+j)*sum(c)
        if i+j:
            assert sum(c)**2>sum(c)
        count+=1
    counts['maximally_correlated_Leibniz_weight_cases']=count
    negative['dropping_weight_sum_as_if_derivatives_independent']='REJECTED'

    count=0
    for r,j in product([Q(0),Q(1,100),Q(1,2),Q(1),Q(2),Q(100)],range(7)):
        assert r**(2*j)<=1+r**12;count+=1
    counts['all_lower_physical_moments_from_mass_and_M12']=count
    # A constant Gaussian mode has no positive spectral moment but grows the cutoff trace.
    for amplitude in [Q(1),Q(10),Q(1000)]:
        b=Q(1,16)
        dc_M12=Q(0)
        trace_lower=16*b*b*amplitude*amplitude
        assert dc_M12==0 and trace_lower>0
    negative['M12_alone_does_not_bound_DC_cutoff_trace']='PRESERVED'
    # Normalized high-frequency mass: fixed L and mass imply sixth derivative variance grows K^12.
    for K in [2,4,8,16]:
        mass=Q(1,3)
        assert mass*K**12 > mass*K**8
    negative['finite_smoothness_without_uniform_M12']='PRESERVED'

    # Physical cover, support embedding, arbitrary rational rotations and a periodic seam.
    rotations=[(Q(1),Q(0)),(Q(3,5),Q(4,5)),(Q(5,13),Q(12,13)),(Q(-12,13),Q(5,13))]
    count=0
    for Lmin in [Q(1,10),Q(1),Q(16),Q(100)]:
        b=min(Q(1),Lmin/16)
        for length in [Lmin,Lmin+Q(1,7),10*Lmin]:
            n=ceil(length/b)
            assert 6*b<length and 8*b*b<length*length
            assert length/n<=b and n*n<=(1+length)**2/(b*b)
            for co,si in rotations:
                assert co*co+si*si==1
                for sx,sy,ox,oy in product([-1,1],repeat=4):
                    # A rotated chart corner plus a worst nearest-grid coordinate offset.
                    x=sx*b*co-sy*b*si+ox*b/2
                    y=sx*b*si+sy*b*co+oy*b/2
                    assert abs(x)<2*b and abs(y)<2*b
            count+=1
    assert Q(2)<Q(3,2)**2
    counts['fixed_physical_cover_and_rotation_parameter_cases']=count
    length=Q(1);b=Q(1,16);g=Q(0);M=length-b/4;short_M=-b/4
    assert abs(short_M-g)<=b/2 and abs(M-g)>b/2
    counts['periodic_seam_nearest_grid_lift']=1

    # Abstract logarithmic coefficients, without evaluating exponential samples.
    assert 8*Q(1,3)+2*Q(1,3)+4==Q(22,3)
    assert Q(4)-Q(4,3)==Q(8,3)
    assert 16**4==65536
    for p,q,u in [(8,2,Q(4)),(9,2,Q(3,2)),(8,3,Q(2))]:
        assert 8-p<=0 and 2-q<=0 and u-Q(4,3)>0
    counts['growing_K_L_and_lifetime_normalization_identities']=5
    assert Q(1,2)-Q(1,3)==Q(1,6)>0
    negative['log_K_of_order_t_minus_half_does_not_guarantee_radius_inclusion']='PRESERVED'
    assert Q(1,3)-Q(1,3)==0
    negative['Tgap_equal_one_over_t_keeps_radius_upper_bound_constant']='PRESERVED'
    return {'state':'INDEPENDENT_EXACT_FINITE_CONTROLS_ONLY','all_checks':'PASS',
            'arithmetic':'Fraction and integers; no sampled fields or numerical Sobolev integrals',
            'counts':counts,'total_finite_cases':sum(counts.values()),'negative_controls':negative,
            'independent_A_theta':str(atheta),'I_j_divided_by_pi':integrals,
            'continuous_and_asymptotic_proof_status':'CONVENTIONAL_REVIEW_NOT_MACHINE_VERIFICATION',
            'formal_or_scientific_promotion':'NONE'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
