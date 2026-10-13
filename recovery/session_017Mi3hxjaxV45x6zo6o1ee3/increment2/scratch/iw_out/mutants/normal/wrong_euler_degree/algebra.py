"""Finite exact identities for PROOF.md; not a continuum or Gaussian checker."""
from fractions import Fraction as F


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(int,F)):
        raise TypeError('int or Fraction required')
    return F(x)


def monomials(degree):
    return [(i,n-i) for n in range(degree+1) for i in range(n+1)]


def evaluate(poly,x,z):
    return sum((c*x**i*z**j for (i,j),c in poly.items()),F(0))


def derivative(poly,axis):
    out={}
    for ij,c in poly.items():
        d=list(ij)
        if d[axis]:
            c*=d[axis];d[axis]-=1
            out[tuple(d)]=c
    return out


def hermite_data(r,raw):
    r=rational(r)
    if r<=0 or len(raw)!=6:raise ValueError('positive separation and six data required')
    gm,gpm,gp,gpp,hm,hp=map(rational,raw)
    a=r/2;s0=(gm+gp)/2;d0=(gp-gm)/r;s1=(gpm+gpp)/2;d1=(gpp-gpm)/r
    return (s0-a*a*d1/2,(3*d0-s1)/2,d1,3*(s1-d0)/(a*a),(hm+hp)/2,(hp-hm)/r)


def observation_matrix(eps,u,v,degree=5):
    eps,u,v=map(rational,(eps,u,v))
    if not 0<=eps<=F(1,4):raise ValueError('epsilon outside compact chart')
    rows=[[] for _ in range(9)]
    for ij in monomials(degree):
        p={ij:F(1)};px=derivative(p,0);pz=derivative(p,1)
        if eps==0:
            ps=[p,px,derivative(px,0),derivative(derivative(px,0),0),pz,derivative(pz,0)]
            first=[evaluate(q,0,0) for q in ps]
        else:
            raw=(evaluate(p,-eps/2,0),evaluate(px,-eps/2,0),evaluate(p,eps/2,0),
                 evaluate(px,eps/2,0),evaluate(pz,-eps/2,0),evaluate(pz,eps/2,0))
            first=hermite_data(eps,raw)
        vals=[*first,evaluate(p,u,v),evaluate(px,u,v),evaluate(pz,u,v)]
        for row,x in zip(rows,vals):row.append(x)
    return rows


def reduced_value_gradient(s,v,fx,fz,height):
    s,v,fx,fz,height=map(rational,(s,v,fx,fz,height))
    if s<=0:raise ValueError('positive physical shell scale required')
    return fx/s**2,fz/s,(height-s*v*fz/2)/s**3


def transverse_matrix(eps,u,v):
    eps,u,v=map(rational,(eps,u,v));de=u*u-eps*eps/4
    return [[u*v,v*v/2,F(0),F(0)],
            [F(0),F(0),F(0),v],
            [v*de/4,F(0),-v**3/12,F(0)]]


def physical_matrix(s,v):
    s,v=map(rational,(s,v))
    return [[s*s,F(0),F(0)],[F(0),s,F(0)],[F(0),s*s*v/2,s**3]]


def euler_remainder(poly,x,z):
    x,z=map(rational,(x,z))
    degree_coefficient=2
    return sum(((degree_coefficient-i-j)*c*x**i*z**j for (i,j),c in poly.items() if i+j),F(0))


def height_in_window(r,k,height):
    r,k,height=map(rational,(r,k,height))
    if r<=0 or k<=0:raise ValueError('positive original r and k required')
    return abs(height)<=k*r**3/2


def shell_weight(r,s):
    r,s=map(rational,(r,s))
    if not 0<r<=s/4:raise ValueError('outside shell chart')
    return (r/s)**2+s*s


def ledger():
    normalizer=-2
    endpoint_product=2
    actual_observations=9
    raw_density=-15
    raw_moment=-60
    absorption=300//4
    return {'actual_observations':actual_observations,'height_length_r_power':3,
            'endpoint_product_r_power':endpoint_product,'normalizer_r_power':normalizer,
            'raw_density_s_power':raw_density,'raw_moment_s_power':raw_moment,
            'near_absorption_s_power':absorption,'near_final_s_power':raw_density+raw_moment+absorption,
            'far_density_s_power':-6,'far_product_s_power':2,'far_total_v_power':-6-6-48,
            'shell_height_r_power':3+normalizer+endpoint_product,'shell_gain':'(r/s)^2+s^2'}
