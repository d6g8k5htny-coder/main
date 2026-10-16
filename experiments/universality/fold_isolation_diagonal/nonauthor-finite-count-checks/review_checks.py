"""Independent finite algebra and counting controls; no sampled fields or analytic proof automation."""
from fractions import Fraction as Q
from itertools import product
import json


def merge(items):
    result={}
    for key,value in items:
        result[key]=result.get(key,Q(0))+value
    return {key:value for key,value in result.items() if value}


def multiply(a,b):
    return merge(((i+k,j+l),x*y) for (i,j),x in a.items() for (k,l),y in b.items())


def gradient_from_laurent(modes,K,axis):
    # Differentiate z^m w^n, divide out i*omega, clear the Laurent monomial.
    return merge(((m+K,n+K),Q((m,n)[axis])*value) for (m,n),value in modes.items())


def on_line(poly,K):
    # z=zeta*w with zeta^K=1: keep the zeta power as a formal quotient coordinate.
    return merge(((i%K,i+j),v) for (i,j),v in poly.items())


def on_hyperbola(poly,K):
    # z=rho/w with rho^K=-1: exact quotient in rho and Laurent w.
    return merge(((i%K,j-i),v*(-1)**(i//K)) for (i,j),v in poly.items())


def run():
    counts={}
    ncases=0
    for K,c in product(range(1,13),[Q(-3,7),Q(1),Q(5,2)]):
        modes={(K,0):Q(1,2),(-K,0):Q(1,2),(0,K):Q(1,2),(0,-K):Q(1,2),
               (K,K):c/2,(-K,-K):c/2}
        assert all(modes[(-m,-n)]==v for (m,n),v in modes.items())
        P={key:2*v/K for key,v in gradient_from_laurent(modes,K,0).items()}
        R={key:2*v/K for key,v in gradient_from_laurent(modes,K,1).items()}
        difference=merge(list(P.items())+[(key,-v) for key,v in R.items()])
        factored=multiply({(K,0):Q(1),(0,K):Q(-1)},{(K,K):Q(1),(0,0):Q(1)})
        assert difference==factored
        assert max(i+j for i,j in P)==max(i+j for i,j in R)==4*K
        assert on_line(P,K)=={(0,3*K):Q(1),(0,K):Q(-1),(0,4*K):c,(0,0):-c}
        assert on_hyperbola(P,K)=={(0,-K):Q(1),(0,K):Q(-1)}
        for poly in (P,R):
            homogenized={(i,j,4*K-i-j):v for (i,j),v in poly.items()}
            assert min(t for i,j,t in homogenized)==0
            assert {key:v for key,v in homogenized.items() if key[2]==0}=={(2*K,2*K,0):c}
        # A rational real Gaussian-coordinate witness for a valid positive full-square law.
        N=(2*K+1)**2; H=N
        q=Q(1,2*H*H);qzero=1-(N-1)*q
        assert q>0 and qzero>0 and qzero+(N-1)*q==1
        assert Q(H,1)/H==1 and (c*H)/H==c
        ncases+=1
    counts['lifted_factor_restrictions_full_degree_and_real_parameter_witness']=ncases

    for K in range(1,13):
        modes={(K,0):Q(1,2),(-K,0):Q(1,2),(0,K):Q(1,2),(0,-K):Q(1,2)}
        for axis in (0,1):
            poly=gradient_from_laurent(modes,K,axis)
            assert max(i+j for i,j in poly)==3*K
            assert min(4*K-i-j for i,j in poly)==K
    counts['failed_lower_degree_witness_common_homogenizing_power']=24

    # Endpoint identities count endpoint pairs once; every ordered pair configuration is distinct.
    total=0
    for maxima,saddles in product(range(1,7),range(1,7)):
        crit=maxima+saddles
        candidates=[(('M',i),('S',j)) for i in range(maxima) for j in range(saddles)]
        ordered=[(p,q) for p in candidates for q in candidates if p!=q]
        assert len(candidates)==maxima*saddles<=crit**2
        assert len(ordered)==len(candidates)*(len(candidates)-1)<=crit**4
        shared_M=sum(p[0]==q[0] for p,q in ordered)
        shared_S=sum(p[1]==q[1] for p,q in ordered)
        assert shared_M==maxima*saddles*(saddles-1)
        assert shared_S==saddles*maxima*(maxima-1)
        presentations=[(p,sign) for p in candidates for sign in (-1,1)]
        assert len({p for p,sign in presentations})==len(candidates)
        assert all(p[0]!=q[1] and p[1]!=q[0] for p,q in ordered)
        total+=1
    counts['ordered_factorial_count_and_shared_role_incidence_cases']=total

    # New cutoff permits coincidence and arbitrarily smaller actual center separation.
    L=Q(1);R0=L/16;eta0=R0/4;rho=Q(3,4);eta=eta0
    r1=eta/2;r2=eta/3
    for s in [Q(0),eta/10000,eta]:
        for u1,u2 in product([-1,1],repeat=2):
            M1=-r1*u1/2;M2=s-r2*u2/2
            S1=r1*u1/2;S2=s+r2*u2/2
            assert r1<=rho*eta and r2<=rho*eta
            assert abs(M2-M1)<=(1+rho)*eta
            assert max(abs(M1),abs(M2),abs(S1),abs(S2))<L/2
    counts['fresh_outer_cutoff_geometry_including_zero_center_separation']=12
    vectors=[(Q(1),Q(0)),(Q(3,5),Q(4,5)),(Q(12,13),Q(5,13)),(Q(-5,13),Q(12,13))]
    for s in [Q(0),eta/10000,eta]:
        for u1,u2 in product(vectors,repeat=2):
            M1=tuple(-r1*v/2 for v in u1)
            M2=(s-r2*u2[0]/2,-r2*u2[1]/2)
            assert sum((M1[j]-M2[j])**2 for j in range(2))<=((1+rho)*eta)**2
    counts['rational_arbitrary_orientation_cutoff_geometry']=48
    assert r1>rho*(eta/10000)
    shared_M=Q(0);S1=r1;S2=-r2
    assert S1!=S2 and abs(S1-shared_M)<=rho*eta and abs(S2-shared_M)<=rho*eta
    assert abs((shared_M+S1)/2-(shared_M+S2)/2)<=eta
    counts['shared_maximum_three_distinct_critical_points_geometry']=1

    # Periodic seam: choose common short lifts rather than subtracting canonical [0,L) coordinates.
    z1=L-eta/4; z2_mod=eta/4; z2_lift=L+eta/4
    assert abs(z2_lift-z1)==eta/2
    assert abs(z2_mod-z1)>eta
    counts['torus_seam_short_lift_control']=1

    # Scalar contradiction constants for the new bounding-cutoff hypothesis.
    ncases=0
    for a,km,b3,b4,rho,eta0 in product([Q(1,3),Q(2)],[Q(1,7),Q(3)],
                                      [Q(1,2),Q(4)],[Q(1,5),Q(2)],
                                      [Q(1,4),Q(3,4)],[Q(1,100),Q(1,1000)]):
        R0=Q(1,16);tau=6*km
        kap=min(R0,a/(32*b3),tau/(32*(b4+b3*b3/a)))
        D0=b4/2+3*b3*b3/a
        cs=min(a*a/(rho*rho*b3*b3*eta0),6*km/(rho*D0),kap/(1+rho))
        assert cs>0
        assert rho*rho*b3*b3*cs*eta0<=a*a
        assert rho*D0*cs<=6*km
        assert kap/cs>=1+rho
        # Choose a coefficient norm and cutoff strictly inside the contradiction premise.
        W=Q(2);cutoff=min(eta0,cs/(2*W*W))
        assert W*W<cs/cutoff
        B,C=b3*W,b4*W
        R=min(R0,a/(32*B),tau/(32*(C+B*B/a)))
        assert rho*cutoff*B<=a
        assert rho*cutoff*(C/2+3*B*B/a)<6*km
        assert R>(1+rho)*cutoff
        ncases+=1
    counts['fresh_cutoff_constant_and_radius_contradiction_cases']=ncases

    for cs in [Q(1,101),Q(3,7),Q(19),Q(1000,3)]:
        c0=cs/64
        assert cs/(16*c0)==4
        assert cs/(16*c0)-Q(4,3)==Q(8,3)
        assert 4*c0/cs==Q(1,16)
    counts['literal_logarithmic_and_cumulative_normalization_exponents']=4
    return {'state':'INDEPENDENT_EXACT_FINITE_CONTROLS_ONLY','all_checks':'PASS',
            'arithmetic':'Python Fraction; no field samples or approximate roots',
            'counts':counts,'total_cases':sum(counts.values()),
            'all_K_coprimeness_projective_closedness_and_analytic_lemma':'REVIEWED_CONVENTIONALLY_NOT_MECHANICALLY_PROVED',
            'gaussian_conditioning_or_Rice_integrals':'NOT_USED','scientific_acceptance':'NONE'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
