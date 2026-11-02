#!/usr/bin/env python3
"""Exact rational finite-law stability support, not a production certificate.
No project imports, network, numerical libraries or filesystem access.
"""
from fractions import Fraction as Q
from math import factorial
import hashlib
import json
import sys

D = 10**50
COUNTS = {}

def ck(ok, name):
    if not ok:
        raise ValueError(name)
    COUNTS[name] = COUNTS.get(name, 0) + 1

def ceilq(q):
    return -((-q.numerator)//q.denominator)

class I:
    """Interval [a/D,b/D]; only integer outward rounding is used."""
    def __init__(self, a, b):
        if a > b: raise ValueError('reversed_interval')
        self.a, self.b = a, b
    @classmethod
    def q(cls, q):
        q = Q(q)*D
        return cls(q.numerator//q.denominator, ceilq(q))
    def __add__(self, v):
        if not isinstance(v,I): v=I.q(v)
        return I(self.a+v.a,self.b+v.b)
    __radd__ = __add__
    def __neg__(self): return I(-self.b,-self.a)
    def __sub__(self,v): return self+-as_i(v)
    def __rsub__(self,v): return as_i(v)+-self
    def __mul__(self,v):
        if not isinstance(v,I): v=I.q(v)
        vals=[self.a*v.a,self.a*v.b,self.b*v.a,self.b*v.b]
        return I(min(vals)//D,-((-max(vals))//D))
    __rmul__ = __mul__
    def scale(self,q):
        q=Q(q)
        vals=[q*self.a,q*self.b]
        lo,hi=min(vals),max(vals)
        return I(lo.numerator//lo.denominator,ceilq(hi))
    def __pow__(self,n):
        if type(n) is not int or n<0: raise ValueError('invalid_power')
        out=I.q(1)
        for _ in range(n): out=out*self
        return out
    def contains(self,q): return Q(self.a,D)<=q<=Q(self.b,D)

def as_i(v): return v if isinstance(v,I) else I.q(v)
def ca(z,w): return (z[0]+w[0],z[1]+w[1])
def cm(z,w): return (z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0])
def cs(z,q): return (z[0].scale(q),z[1].scale(q))

def atan_series(x,n):
    return sum(((-1)**j*x**(2*j+1)/Q(2*j+1) for j in range(n)),Q(0))

def pi_interval():
    # Alternating partial sums: even truncation is lower, odd is upper.
    lo=16*atan_series(Q(1,5),40)-4*atan_series(Q(1,239),41)
    hi=16*atan_series(Q(1,5),41)-4*atan_series(Q(1,239),40)
    ck(Q(314159,100000)<lo<hi<Q(22,7),'pi_bracket')
    t=Q(1,5)
    t=2*t/(1-t*t);t=2*t/(1-t*t)
    ck(t==Q(120,119) and (t-Q(1,239))/(1+t/Q(239))==1,'machin_tangent')
    ck(4*Q(5,26)-Q(1,239)>0 and 4*Q(1,5)<1,'machin_branch')
    return I(I.q(lo).a,I.q(hi).b)

def trig_point(q,sine):
    """Alternating Taylor bounds, 0<=q<=1: 20/21 terms."""
    ck(0<=q<=1,'trig_domain')
    def partial(n):
        return sum(((-1)**j*q**(2*j+int(sine))/factorial(2*j+int(sine))
                    for j in range(n)),Q(0))
    return partial(20),partial(21)

def trig(t):
    # cos is even, sin odd; monotonic on [0,1].
    if t.a==0 and t.b==0: return I.q(1),I.q(0)
    if t.b<=0:
        c,s=trig(-t)
        return c,-s
    if t.a<0: raise ValueError('straddling_trig_interval')
    a,b=Q(t.a,D),Q(t.b,D)
    sa,_=trig_point(a,True);_,sb=trig_point(b,True)
    cb,_=trig_point(b,False);_,ca_=trig_point(a,False)
    return I(I.q(cb).a,I.q(ca_).b),I(I.q(sa).a,I.q(sb).b)

PINS=[[(0,0,0,Q(1),0)],[(0,1,0,Q(1),0)],[(0,0,1,Q(1),0)],
      [(1,1,0,Q(1),-1),(0,1,0,Q(-1),-1)],
      [(1,0,1,Q(1),-1),(0,0,1,Q(-1),-1)],
      [(1,0,0,Q(1),-3),(0,0,0,Q(-1),-3),
       (1,1,0,Q(-1,2),-2),(0,1,0,Q(-1,2),-2)]]
SOFT=[[(0,0,2,Q(1),0)],[(1,0,2,Q(1),0)],
      [(0,2,0,Q(1),-1)],[(1,2,0,Q(1),-1)],
      [(0,1,1,Q(1),-1)],[(1,1,1,Q(1),-1)]]
FRAME=PINS+SOFT

def features(x,y,phase,r=Q(1,20)):
    result=[]
    for rows in FRAME:
        out=(I.q(0),I.q(0))
        for station,a,b,w,k in rows:
            q=x**a*y**b
            derivative=((q,I.q(0)),(I.q(0),q),(-q,I.q(0)),(I.q(0),-q))[(a+b)%4]
            value=cm(derivative,phase) if station else derivative
            out=ca(out,cs(value,w*r**k))
        result.append(out)
    return result

def ldl(A):
    n=len(A);L=[[Q(int(i==j)) for j in range(n)] for i in range(n)];ds=[]
    for i in range(n):
        v=A[i][i]-sum((L[i][k]**2*ds[k] for k in range(i)),Q(0))
        if v<=0: raise ValueError('not_positive_definite')
        ds.append(v)
        for j in range(i+1,n):
            L[j][i]=(A[j][i]-sum((L[j][k]*L[i][k]*ds[k] for k in range(i)),Q(0)))/v
    for i in range(n):
        for j in range(n):
            ck(sum((L[i][k]*ds[k]*L[j][k] for k in range(n)),Q(0))==A[i][j],
               'exact_ldl_reconstruction')
    return ds

def qcomplex_add(z,w): return z[0]+w[0],z[1]+w[1]
def qcomplex_mul(z,w): return z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0]
def qcomplex_scale(z,t): return z[0]*t,z[1]*t

def frame_exact(x,y,c,s,r):
    """Literal-table evaluation over arbitrary rational phase components."""
    phase=(c,s);out=[]
    for rows in FRAME:
        z=(Q(0),Q(0))
        for station,a,b,w,k in rows:
            q=x**a*y**b
            der=((q,Q(0)),(Q(0),q),(-q,Q(0)),(Q(0),-q))[(a+b)%4]
            v=qcomplex_mul(der,phase) if station else der
            z=qcomplex_add(z,qcomplex_scale(v,w*r**k))
        out.append(z)
    return out

def frame_reference(x,y,c,s,r):
    """Independently expanded real/imaginary formulas in the stated order."""
    return [(1,0),(0,x),(0,y),(-x*s/r,x*(c-1)/r),
            (-y*s/r,y*(c-1)/r),
            ((c-1)/r**3+x*s/(2*r*r),s/r**3-x*(c+1)/(2*r*r)),
            (-y*y,0),(-y*y*c,-y*y*s),(-x*x/r,0),
            (-x*x*c/r,-x*x*s/r),(-x*y/r,0),(-x*y*c/r,-x*y*s/r)]

def expect_reject(fn,name):
    try: fn()
    except ValueError as exc:
        ck(str(exc)==name,'named_rejection_'+name)
    else: raise ValueError('negative_control_accepted')

def self_tests():
    values=[Q(a,b) for a in range(-3,4) for b in (1,3,7)]
    for x in values:
        for y in values:
            ix,iy=I.q(x),I.q(y)
            ck((ix+iy).contains(x+y),'interval_add_point')
            ck((ix*iy).contains(x*y),'interval_mul_point')
            ck(ix.scale(y).contains(x*y),'interval_scale_point')
    for a,b,c,d in ((-7,2,-4,5),(-3,-1,2,8),(0,4,0,9),(-1,1,-1,1)):
        x,y=I(a,b),I(c,d)
        z=x*y
        for u in (a,b):
            for v in (c,d):
                ck(z.contains(Q(u*v,D*D)),'interval_endpoint_product')
    z=I(1,1)*I(-1,-1)
    ck((z.a,z.b)==(-1,0),'negative_subgrid_rounding')
    c,s=trig(I.q(0));ck((c.a,c.b,s.a,s.b)==(D,D,0,0),'zero_phase')
    for q in (Q(1,10),Q(1,5),Q(1,2),Q(1)):
        c,s=trig(I.q(q))
        for sine,iv in ((False,c),(True,s)):
            lo=sum(((-1)**j*q**(2*j+int(sine))/factorial(2*j+int(sine))
                    for j in range(30)),Q(0))
            hi=lo+q**(60+int(sine))/factorial(60+int(sine))
            ck(iv.contains(lo) and iv.contains(hi),'trig_nested_bounds')
        cn,sn=trig(I.q(-q))
        ck((cn.a,cn.b)==(c.a,c.b) and (sn.a,sn.b)==(-s.b,-s.a),'trig_parity')
    for x in (Q(-2,3),Q(0),Q(7,5)):
        for y in (Q(-3,4),Q(0),Q(5,7)):
            for r in (Q(1,20),Q(2,7)):
                for c,s in ((Q(-1,3),Q(2,5)),(Q(1),Q(0))):
                    a,b=frame_exact(x,y,c,s,r),frame_reference(x,y,c,s,r)
                    ck(len(a)==len(b)==12,'frame_dimension')
                    for v,w in zip(a,b):ck(v==w,'literal_frame_correspondence')
    ck(ldl([[Q(2),Q(1)],[Q(1),Q(2)]])==[Q(2),Q(3,2)],'known_ldl')
    expect_reject(lambda:ldl([[Q(1),Q(1)],[Q(1),Q(1)]]),'not_positive_definite')
    expect_reject(lambda:ldl([[Q(1),Q(2)],[Q(2),Q(1)]]),'not_positive_definite')

def gram_enclosure(h,collapse=False):
    """Unweighted sum of real feature outer products, all 35 frequencies."""
    n=12;G=[[I.q(0) for _ in range(n)] for _ in range(n)]
    phases={j:((I.q(1),I.q(0)) if collapse or j==0 else trig(h.scale(Q(j,20))))
            for j in range(-3,4)}
    terms=0
    for j in range(-3,4):
        for k in range(-2,3):
            z=features(h.scale(j),h.scale(k),phases[j]);terms+=1
            for a in range(n):
                for b in range(a,n):
                    G[a][b]=G[a][b]+z[a][0]*z[b][0]+z[a][1]*z[b][1]
                    G[b][a]=G[a][b]
    ck(terms==35,'selected_frequency_count')
    H=[[Q((v.a+v.b)//2,D) for v in row] for row in G]
    err=max(sum((max(abs(H[a][b]-Q(G[a][b].a,D)),abs(H[a][b]-Q(G[a][b].b,D)))
                 for b in range(n)),Q(0)) for a in range(n))
    return H,err

def witness(H,err,t=Q(1,10**14),g=Q(1,10**20)):
    ck(err>=0,'nonnegative_error')
    piv=ldl([[H[i][j]-(t if i==j else 0) for j in range(12)] for i in range(12)])
    ck(t-err > (3*233**2)*g,'joint_floor_margin')
    return piv

def stability_constants(h):
    # All selected exponent arguments are <=13*h^2/2<1; e<3, so w>1/3.
    ck(Q(13,2)*(Q(h.b,D))**2<1,'selected_weight')
    ck(116*Q(h.b,D)<31,'retained_frequency_radius')
    # Z_J<=233 because every Gaussian weight is <=1.
    ck(2*116+1==233,'normalizer_cardinality')
    g=Q(1,10**20);L=Q(2*10**9)
    x=Q(31);r=Q(1,20)
    feature_ceiling=1+2*x*x+2*(1+2/r**2)*x**4+2*x**4+x**6/144
    ck(feature_ceiling<L,'finite_trace_ceiling')
    beta=L/g
    # Inputs below are the previously separately reviewed exact-frame tail result.
    tau=Q(7,10**192);a=Q(1,10**200)
    ck(a<Q(1,2),'alpha_lower')
    ck(Q(6,5)**2+Q(1,6)**2<4,'conditioning_vector_norm')
    mean=4*(1+beta)*tau/g
    covariance=a*L+(1+beta*beta)*tau
    ck(mean<Q(6,10**142),'conditional_mean_ceiling')
    ck(covariance<Q(3,10**133),'conditional_covariance_ceiling')
    ck(covariance/g<Q(3,10**113),'relative_covariance_ceiling')
    ck(mean*mean/g<Q(4,10**263),'normalized_mean_square_ceiling')
    return {'g':str(g),'s':str(g),'beta_upper':str(beta),
            'finite_joint_trace_upper':str(L),'feature_trace_ceiling':str(feature_ceiling),
            'imported_tau':str(tau),'imported_mass_defect':str(a),
            'mean_bound':str(mean),'covariance_bound':str(covariance),
            'relative_covariance_bound':str(covariance/g),
            'normalized_mean_square_bound':str(mean*mean/g)}

def main():
    self_tests()
    h=pi_interval().scale(Q(1,12))
    H,err=gram_enclosure(h)
    ck(err<Q(1,10**40),'gram_error_ceiling')
    piv=witness(H,err)
    constants=stability_constants(h)
    # Positive checks plus nontrivial bad witnesses. These are not theorem counts.
    expect_reject(lambda:witness(H,err,t=Q(1,10**10)),'not_positive_definite')
    expect_reject(lambda:witness(H,err,g=Q(1,10**18)),'joint_floor_margin')
    expect_reject(lambda:witness(H,Q(1,10**13)),'joint_floor_margin')
    collapsed,ce=gram_enclosure(h,collapse=True)
    expect_reject(lambda:witness(collapsed,ce),'not_positive_definite')
    matrix=[[str(v) for v in row] for row in H]
    matrix_bytes=json.dumps(matrix,separators=(',',':')).encode()
    out={'status':'AUTHOR_SUPPORT_REQUIRES_NONAUTHOR_REVIEW','python':sys.version.split()[0],
         'source_commit':'3279db8bd8cf93905376eb2ff782e8afc3fdb544',
         'program_blob':'34a1a3cd17f7a78975e3ed824520ef8b84ff8742',
         'field_definition_blob':'a160edb4cfe4f32eaec4fd44772aaeb44ef8c893',
         'r':'1/20','square_cutoff':116,'selected_x':[-3,3],'selected_y':[-2,2],
         'grid_denominator':str(D),'midpoint_matrix_sha256':hashlib.sha256(matrix_bytes).hexdigest(),
         'gram_error':str(err),'ldl_threshold':'1/100000000000000',
         'positive_ldl_pivots':len(piv),'ldl_pivots':[str(v) for v in piv],
         'constants':constants,'checks':sum(COUNTS.values()),'counts':COUNTS,
         'limits':['literal unrounded frame only','fixed r only','no expectation floor',
                   'no original certifier or numerical library run','tail inputs separately imported',
                   'not scientific acceptance or repository adoption']}
    print(json.dumps(out,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
