"""Independent finite algebra attacks; these are not an analytic proof or field experiment."""
from fractions import Fraction as Q
from itertools import product
import json


def d_at(coeffs, x, order=1):
    ans = Q(0)
    for j, a in enumerate(coeffs):
        if j >= order:
            factor = 1
            for k in range(order):
                factor *= j-k
            ans += a * factor * x**(j-order)
    return ans


def value(coeffs, x):
    return d_at(coeffs, x, 0)


def kernel_third(coeffs, r):
    # Integrate the monomials exactly against 6s(r-s)/r^3.
    ans = Q(0)
    for j, a in enumerate(coeffs):
        if j >= 3:
            factor = j*(j-1)*(j-2)*a
            ans += 6*factor/r**3 * (r*r**(j-1)/Q(j-1)-r**j/Q(j))
    return ans


def det(matrix):
    a = [list(row) for row in matrix]
    ans = Q(1)
    for i in range(len(a)):
        pivot = next((j for j in range(i,len(a)) if a[j][i]),None)
        if pivot is None:
            return Q(0)
        if pivot != i:
            a[i],a[pivot]=a[pivot],a[i]
            ans=-ans
        v=a[i][i]
        ans*=v
        for j in range(i+1,len(a)):
            c=a[j][i]/v
            for k in range(i+1,len(a)):
                a[j][k]-=c*a[i][k]
    return ans


def run():
    counts = {}
    # Independent jet cancellation: start with the unreduced differentiated f''.
    count=0
    for A,p,xxx,xxy,xyy,yyy in product([Q(-1),Q(-7,3)], [Q(-1,8),Q(0),Q(1,11)],
                                       [Q(2,3),Q(-3,5)], [Q(5,7),Q(-2)],
                                       [Q(-4,9),Q(3)], [Q(7,11),Q(-5)]):
        xy=-A*p
        hpp=-(xxy+2*xyy*p+yyy*p*p)/A
        direct=xxx+2*xxy*p+xyy*p*p+xy*hpp
        assert direct==xxx+3*xxy*p+3*xyy*p*p+yyy*p**3
        count+=1
    counts['mixed_chain_rule_jet_cancellations']=count

    # Check integration by parts before imposing endpoint criticality, then impose it.
    count=0
    for degree,r in product(range(3,10),[Q(1,7),Q(2,3),Q(3,2)]):
        coeffs=[Q((-1)**j*(j+1),j+2) for j in range(degree+1)]
        lhs=kernel_third(coeffs,r)
        rhs=6/r**3*(r*(d_at(coeffs,0)+d_at(coeffs,r))-2*(value(coeffs,r)-value(coeffs,0)))
        assert lhs==rhs
        coeffs[1]=Q(0)
        coeffs[2]=-sum(Q(j)*coeffs[j]*r**(j-2) for j in range(3,len(coeffs)))/2
        assert d_at(coeffs,0)==0 and d_at(coeffs,r)==0
        k=(value(coeffs,0)-value(coeffs,r))/r**3
        assert kernel_third(coeffs,r)==12*k
        count+=1
    counts['general_kernel_identity_and_endpoint_mark']=count

    # Closed-square endpoint roots: the proposed radius itself, for both cubic signs.
    R=Q(1,32); mu=R/2; a=Q(1)
    boundary=[]
    for tau in [Q(1),Q(-1)]:
        F=[Q(0),Q(0),-mu/2,tau/6]
        second=2*mu/tau
        assert abs(second)==R
        assert d_at(F,0)==0 and d_at(F,second)==0
        assert d_at(F,0,2)==-mu<0 and d_at(F,second,2)==mu>0
        assert min(Q(1),a/32,Q(1)/(32*(Q(0)+Q(1)/a)))==R
        assert -a*d_at(F,second,2)<0
        boundary.append(str(second))
    counts['closed_x_face_saddles_both_signs']=2

    # Negative hard pivot: Schur sign, including nonzero off-diagonal Hessians.
    count=0
    for A,b,s in product([Q(-1),Q(-2,3),Q(-11)], [Q(0),Q(5),Q(-7,13)], [Q(-1,9),Q(2,7)]):
        xx=s+b*b/A
        assert xx-b*b/A==s
        assert xx*A-b*b==A*s
        if s<0:
            assert xx<0 and xx*A-b*b>0
        else:
            assert xx*A-b*b<0
        count+=1
    counts['hard_pivot_schur_classifications']=count

    # Full boundary height barrier fails even at the actual stated chart radius.
    escape=[Q(0),Q(0),-Q(1,1000)/2,Q(1,6)]
    assert value(escape,R)>value(escape,0)
    # At a stationary mixed-cubic ridge the reduced cubic is tau+3b^2*x/a.
    aa,tau,b=Q(3),Q(2),Q(7)
    reduced=[Q(0),Q(0),-Q(1,10)/2,tau/6,b*b/(8*aa)]
    xloss=-aa*tau/(3*b*b)
    assert d_at(reduced,xloss,3)==0
    assert d_at(reduced,xloss-Q(1,100),3)<0
    assert d_at(reduced,xloss+Q(1,100),3)>0
    counts['retained_false_barrier_and_D4zero_obstruction']=2

    # Exact anisotropic spectral row/operator bound at arbitrary rational frames.
    vectors=[(Q(3,5),Q(4,5)),(Q(12,13),Q(5,13)),(Q(-5,13),Q(12,13)),(Q(1),Q(0))]
    count=0
    for K,j in product([2,3],[3,4]):
        modes=[(n,m) for n in range(-K,K+1) for m in range(-K,K+1)]
        raw={(n,m):Q(1+n*n+2*m*m) for n,m in modes}
        total=sum(raw.values());weights={n:w/total for n,w in raw.items()}
        beta2=sum(weights[n]*(n[0]*n[0]+n[1]*n[1])**j for n in modes)
        for offset in range(4):
            slots=[vectors[(offset+k)%4] for k in range(j)]
            row2=Q(0)
            for n in modes:
                factor=Q(1)
                for v in slots:
                    assert v[0]*v[0]+v[1]*v[1]==1
                    projection=n[0]*v[0]+n[1]*v[1]
                    assert projection**2 <= n[0]*n[0]+n[1]*n[1]
                    factor*=projection**2
                row2+=weights[n]*factor
            assert row2<=beta2
            count+=1
    counts['finite_spectral_multilinear_rows']=count

    # Radius lower bound is a scalar algebraic inequality for every W>=1.
    count=0
    for R0,a,tau,b3,b4,W in product([Q(1,16),Q(2)], [Q(1,9),Q(3)],
                                    [Q(1,11),Q(4)], [Q(1,3),Q(7)],
                                    [Q(0),Q(5)], [Q(1),Q(3,2),Q(9)]):
        B,C=b3*W,b4*W
        R=min(R0,a/(32*B),tau/(32*(C+B*B/a)))
        kappa=min(R0,a/(32*b3),tau/(32*(b4+b3*b3/a)))
        assert R>=kappa/(W*W)
        count+=1
    counts['inverse_W_squared_radius_inequalities']=count

    # Zero-target regression projectors can be singular and need no covariance floor.
    projectors=[([[Q(0)]],0),([[Q(1)]],1),
                ([[Q(1,2),Q(1,2)],[Q(1,2),Q(1,2)]],1),
                ([[Q(int(i==j))-Q(1,3) for j in range(3)] for i in range(3)],2)]
    for P,rank in projectors:
        n=len(P)
        for i,j in product(range(n),repeat=2):
            assert P[i][j]==P[j][i]
            assert sum(P[i][k]*P[k][j] for k in range(n))==P[i][j]
        # det(I-P/2)^(-1/2)=2^(rank/2), the Gaussian MGF at t=1/4.
        assert det([[Q(int(i==j))-P[i][j]/2 for j in range(n)] for i in range(n)])==Q(1,2**rank)
    counts['singular_centered_regression_projector_MGF_checks']=len(projectors)
    return {'state':'INDEPENDENT_EXACT_FINITE_CHECKS_ONLY','all_checks':'PASS',
            'counts':counts,'total_cases':sum(counts.values()),
            'boundary_second_roots':boundary,'actual_radius_full_height_escape':str(value(escape,Q(1,32))),
            'mixed_cubic_sign_loss':str(xloss),
            'sampling':'NONE','analytic_proof_mechanically_verified':False,
            'formal_or_scientific_acceptance':'NONE'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
