"""H0-only interpolation comparison; numerical diagnostics are not certificates."""
import math
from fractions import Fraction
import numpy as np
import gudhi
import experiment as ex


def triangulated_h0(values,diagonal='plus'):
    """Periodic triangulated PL upper-star H0 via its graph (no higher-H claim)."""
    a=ex._grid(values)
    if min(a.shape)<3:raise ValueError('Triangulation requires at least three vertices per periodic side')
    if diagonal not in ('plus','minus','minimum'):raise ValueError('Unknown diagonal')
    indices=np.arange(a.size).reshape(a.shape);v=-a.ravel();tree=gudhi.SimplexTree()
    tree.insert_batch(indices.reshape(1,-1),v)
    def insert(u,w):
        u=u.ravel();w=w.ravel()
        tree.insert_batch(np.stack((u,w)),np.maximum(v[u],v[w]))
    right=np.roll(indices,-1,axis=0);up=np.roll(indices,-1,axis=1)
    opposite=np.roll(right,-1,axis=1)
    insert(indices,right);insert(indices,up)
    if diagonal=='plus':insert(indices,opposite)
    elif diagonal=='minus':insert(right,up)
    else:
        corners=np.stack((a,np.roll(a,-1,axis=0),np.roll(a,-1,axis=1),np.roll(np.roll(a,-1,axis=0),-1,axis=1)))
        k=np.argmin(corners,axis=0);plus=(k==0)|(k==3)
        insert(np.where(plus,indices,right),np.where(plus,opposite,up))
    tree.compute_persistence(homology_coeff_field=2,min_persistence=-1,persistence_dim_max=False)
    bars=tree.persistence_intervals_in_dimension(0);finite=bars[np.isfinite(bars[:,1])]
    return {'intervals':sorted([float(-u),float(-v)] for u,v in finite if v>u),
            'essential':sorted(float(-u) for u,v in bars if math.isinf(v)),
            'zero_count':int(np.sum(finite[:,0]==finite[:,1]))}


def oracle_triangulated_h0(values,diagonal):
    """Separate descending union-find, explicit six-neighbor periodic stencil."""
    a=ex._grid(values);rows,cols=a.shape
    if min(a.shape)<3 or diagonal not in ('plus','minus'):raise ValueError('Invalid triangulation')
    f=a.ravel();parent=list(range(a.size));active=[False]*a.size;birth=f.copy();intervals=[];zero=0
    direction=1 if diagonal=='plus' else -1
    offsets=[(-1,0),(1,0),(0,-1),(0,1),(1,direction),(-1,-direction)]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in sorted(range(a.size),key=lambda k:(-f[k],k)):
        active[i]=True;x,y=divmod(i,cols)
        for dx,dy in offsets:
            j=((x+dx)%rows)*cols+(y+dy)%cols
            if not active[j]:continue
            u,v=root(i),root(j)
            if u==v:continue
            if (birth[u],-u)<(birth[v],-v):u,v=v,u
            if birth[v]>f[i]:intervals.append([float(birth[v]),float(f[i])])
            else:zero+=1
            parent[v]=u
    return {'intervals':sorted(intervals),'essential':sorted(float(birth[i]) for i in range(a.size) if root(i)==i),'zero_count':zero}


def hessian_diagnostic(bank,cutoff):
    """Floating evaluation of sum 2|z_k||omega_k|² for the retained Fourier field."""
    den=ex.denominator();terms=[]
    for (x,y),(a,b) in bank.items():
        if (x,y)==(0,0) or max(abs(x),abs(y))>cutoff:continue
        weight=math.exp(-2*math.pi**2*(x*x+y*y)/24**2)/den
        terms.append(math.sqrt(2*weight)*math.hypot(a,b)*(2*math.pi/24)**2*(x*x+y*y))
    return math.fsum(terms)


def interpolation_diagnostic(hessian,grid):
    return (24/grid)**2*hessian/4


def bin_sandwich(lifetimes,a,b,epsilon):
    """Exact rational combinatorics conditional on an externally certified matching.

    Returns no finite upper bound when a<=2 epsilon: unknown true bars may
    match the diagonal. This helper does not certify epsilon or the matching.
    """
    if not all(isinstance(x,Fraction) for x in [a,b,epsilon,*lifetimes]) or not 0<a<b or epsilon<0 or any(x<=0 for x in lifetimes):
        raise ValueError('Positive rational lifetimes/bin and nonnegative rational epsilon required')
    delta=2*epsilon
    lower=sum(a+delta<=x<b-delta for x in lifetimes) if a+delta<b-delta else 0
    upper=None if a<=delta else sum(a-delta<=x<b+delta for x in lifetimes)
    return {'lower':lower,'upper':upper,'diagonal_obstruction':a<=delta}
