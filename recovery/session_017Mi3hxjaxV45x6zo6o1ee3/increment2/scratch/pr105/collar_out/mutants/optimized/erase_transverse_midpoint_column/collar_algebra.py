"""Exact finite polynomial controls for COLLAR_PROOF.md, not a continuum checker.

Only the Python standard library is used. No archived research program,
network request, Gaussian numerical integrator, or automatic status mutation.
"""
from fractions import Fraction as F


def multiply(p,q):
    out=[F(0)]*(len(p)+len(q)-1)
    for i,x in enumerate(p):
        for j,y in enumerate(q): out[i+j]+=x*y
    return out


def derivative(p):
    return [i*p[i] for i in range(1,len(p))] or [F(0)]


def evaluate(p,t):
    out=F(0)
    for x in reversed(p): out=out*t+x
    return out


def cardinals(nodes,i):
    if len(nodes)!=3 or len(set(nodes))!=3 or not 0<=i<3:
        raise ValueError('three distinct nodes and valid index required')
    ti=nodes[i]; ell=[F(1)]
    for j,tj in enumerate(nodes):
        if j!=i: ell=multiply(ell,[-tj/(ti-tj),F(1)/(ti-tj)])
    sq=multiply(ell,ell); slope=evaluate(derivative(ell),ti)
    h=multiply([1+2*slope*ti,-2*slope],sq)
    d=multiply([-ti,F(1)],sq)
    return h,d,sq


def jet_matrix(points, degree=5):
    mons=[(i,j) for n in range(degree+1) for i in range(n+1) for j in [n-i]]
    out=[]
    for x,y in points:
        out.append([x**i*y**j for i,j in mons])
        out.append([i*x**(i-1)*y**j if i else F(0) for i,j in mons])
        out.append([j*x**i*y**(j-1) if j else F(0) for i,j in mons])
    return out


def rank(matrix):
    if not matrix: return 0
    rows=[list(map(F,row)) for row in matrix]
    if any(len(row)!=len(rows[0]) for row in rows): raise ValueError('ragged matrix')
    r=0
    for c in range(len(rows[0])):
        p=next((j for j in range(r,len(rows)) if rows[j][c]),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]
        pivot=rows[r][c]; rows[r]=[x/pivot for x in rows[r]]
        for j in range(r+1,len(rows)):
            t=rows[j][c]
            if t: rows[j]=[x-t*y for x,y in zip(rows[j],rows[r])]
        r+=1
        if r==len(rows): break
    return r


def in_collar(u,v,eta,outer):
    if eta<=0 or outer<=0: raise ValueError('positive radii required')
    return (u*u+v*v<=outer*outer and
            (u+F(1,2))**2+v*v>=eta*eta and
            (u-F(1,2))**2+v*v>=eta*eta)


def transverse_block(u,v):
    return [[u*v,F(0),F(0)],[F(0),F(0),v]]


def floor_orders():
    return {'jet_degree':5,'std_floor':5,'variance_floor':10,'remainder':6}


def near_axis(v,r):
    if r<=0: raise ValueError('positive r required')
    return abs(v)**3<=r


def powers():
    normalizer=-2
    raw_density=-10
    raw_moment=-60
    transverse_density=-3
    triple_determinant=6
    n=111
    return {'near_raw':normalizer+raw_density+raw_moment,
            'near_absorption':F(2*n,3),
            'near_final':normalizer+raw_density+raw_moment+F(2*n,3),
            'transverse_r':normalizer+transverse_density+triple_determinant,
            'transverse_v':-4-24-6,'transverse_absorption':2*17}


def observations():
    return {'auxiliary_gram':9,'actual_conditioning':8,'height_window_factor':0}
