"""Planar periodic Gaussian-field pilot. Numerical software, not a proof certificate."""
import math
from fractions import Fraction
import numpy as np
import gudhi

SIDE=24.0
COEFFICIENT=0.07340691930603427103
DENOMINATOR_CUTOFF=64

def _grid(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or min(a.shape)<2 or not np.isfinite(a).all():
        raise ValueError('Expected finite rectangular 2D grid with each side at least two')
    return a

def oracle_h0(values,periodic=True):
    """Independent descending axis-edge union-find; ties use row-major index order."""
    a=_grid(values); rows,cols=a.shape; f=a.ravel()
    parent=list(range(f.size)); active=[False]*f.size
    births=f.copy(); intervals=[]; zero=0
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in sorted(range(f.size),key=lambda k:(-f[k],k)):
        active[i]=True;x,y=divmod(i,cols)
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            xx,yy=x+dx,y+dy
            if periodic:xx,yy=xx%rows,yy%cols
            elif not (0<=xx<rows and 0<=yy<cols):continue
            j=xx*cols+yy
            if not active[j]:continue
            u,v=root(i),root(j)
            if u==v:continue
            # Higher maximum survives; equal births break ties deterministically.
            if (births[u],-u)<(births[v],-v):u,v=v,u
            death=float(f[i]);birth=float(births[v])
            if birth>death:intervals.append([birth,death])
            else:zero+=1
            parent[v]=u
    essential=sorted(float(births[i]) for i in range(f.size) if root(i)==i)
    return {'intervals':sorted(intervals),'essential':essential,'zero_count':zero}

def gudhi_h0(values):
    a=_grid(values)
    complex_=gudhi.PeriodicCubicalComplex(vertices=-a,periodic_dimensions=[True,True])
    complex_.compute_persistence(homology_coeff_field=2,min_persistence=-1)
    bars=complex_.persistence_intervals_in_dimension(0)
    finite=bars[np.isfinite(bars[:,1])]
    positive=finite[finite[:,1]>finite[:,0]]
    return {'intervals':sorted([float(-u),float(-v)] for u,v in positive),
            'essential':sorted(float(-u) for u,v in bars if math.isinf(v)),
            'zero_count':int(np.sum(finite[:,1]==finite[:,0]))}

def mode_bank(seed,max_cutoff):
    if not isinstance(max_cutoff,int) or max_cutoff<1:raise ValueError('Positive integer cutoff required')
    rng=np.random.Generator(np.random.PCG64(seed))
    bank={(0,0):(float(rng.normal()),0.)}
    for x in range(max_cutoff+1):
        for y in range(-max_cutoff,max_cutoff+1):
            if x==0 and y<=0:continue
            bank[x,y]=tuple(float(z) for z in rng.normal(size=2))
    return bank

def denominator(side=SIDE):
    if not math.isfinite(side) or side<=0:raise ValueError('Positive finite side required')
    return math.fsum(math.exp(-2*math.pi**2*k*k/side**2)
                     for k in range(-DENOMINATOR_CUTOFF,DENOMINATOR_CUTOFF+1))**2

def field_grid(bank,n,cutoff,side=SIDE,*,decay=1.,variance_factor=1.):
    """Shared mode bank across grids/cutoffs; mutation factors are model diagnostics."""
    if not isinstance(n,int) or not isinstance(cutoff,int) or cutoff<1 or n<=2*cutoff:
        raise ValueError('Grid must resolve every retained mode: n > 2*cutoff >= 2')
    if cutoff>DENOMINATOR_CUTOFF:raise ValueError('Cutoff exceeds declared denominator')
    if decay<=0 or variance_factor<=0:raise ValueError('Positive model factors required')
    den=denominator(side);coeff=np.zeros((n,n),dtype=complex)
    coeff[0,0]=bank[0,0][0]/math.sqrt(den)
    for (x,y),(a,b) in bank.items():
        if (x,y)==(0,0) or max(abs(x),abs(y))>cutoff:continue
        weight=math.exp(-decay*2*math.pi**2*(x*x+y*y)/side**2)/den
        z=math.sqrt(variance_factor*weight/2)*complex(a,-b)
        coeff[x%n,y%n]=z;coeff[-x%n,-y%n]=z.conjugate()
    values=np.fft.ifft2(coeff,norm='forward')
    if np.max(np.abs(values.imag))>1e-12:raise ValueError('Non-real Fourier result')
    return values.real

def reference_covariance(cutoff,lag,side=SIDE):
    """Separable full-lattice cosine sum, independent of conjugate-bank assembly."""
    alpha=2*math.pi**2/side**2
    sums=[math.fsum(math.exp(-alpha*k*k)*math.cos(2*math.pi*k*z/side)
                    for k in range(-cutoff,cutoff+1)) for z in lag]
    return math.prod(sums)/denominator(side)

def spectral_diagnostics(cutoff,side=SIDE):
    alpha=2*math.pi**2/side**2; den=denominator(side)
    tails={0:0.,1:0.,2:0.}
    for x in range(-DENOMINATOR_CUTOFF,DENOMINATOR_CUTOFF+1):
        for y in range(-DENOMINATOR_CUTOFF,DENOMINATOR_CUTOFF+1):
            if max(abs(x),abs(y))<=cutoff:continue
            w=math.exp(-alpha*(x*x+y*y))/den
            frequency=(2*math.pi/side)**2*(x*x+y*y)
            for order in tails:tails[order]+=w*frequency**order
    return {'variance_retained':reference_covariance(cutoff,(0,0),side),
            'omitted_sums_to_64':{str(k):v for k,v in tails.items()},
            'scope':'Floating-point spectral variance/derivative diagnostics; modes beyond64 not enclosed; no uniform pathwise bound'}

def _edges(edges):
    e=np.asarray(edges,dtype=float)
    if e.ndim!=1 or len(e)<2 or not np.isfinite(e).all() or e[0]<=0 or not np.all(np.diff(e)>0):
        raise ValueError('Strictly increasing finite positive bin edges required')
    return e

def bin_counts(lifetimes,edges):
    e=_edges(edges);v=np.asarray(lifetimes,dtype=float)
    if v.ndim!=1 or not np.isfinite(v).all() or np.any(v<=0):
        raise ValueError('Finite strictly positive lifetimes required')
    # Unlike numpy.histogram, every bin including the final one is half open.
    return [int(np.sum((v>=a)&(v<b))) for a,b in zip(e[:-1],e[1:])]

def rational_shape(t,u):
    if not isinstance(t,Fraction) or not isinstance(u,Fraction) or not 0<t<u:
        raise ValueError('Positive ordered rational cube-root endpoints required')
    return Fraction(3,2)*(u*u-t*t)

def predicted_bin_mass(a,b,coefficient=COEFFICIENT):
    _edges([a,b])
    return 1.5*coefficient*(b**(2/3)-a**(2/3))

def summarize_counts(counts,edges,side=SIDE):
    e=_edges(edges);c=np.asarray(counts,dtype=float)
    if c.ndim!=2 or c.shape[0]<1 or c.shape[1]!=len(e)-1 or not np.isfinite(c).all() or np.any(c<0) or np.any(c!=np.floor(c)):
        raise ValueError('Nonnegative integer per-realization bin matrix required')
    if not math.isfinite(side) or side<=0:raise ValueError('Positive finite side required')
    mass=c/side**2
    means=np.mean(mass,axis=0)
    errors=np.std(mass,axis=0,ddof=1)/math.sqrt(len(c)) if len(c)>1 else [None]*c.shape[1]
    return [{'a':float(a),'b':float(b),'total':int(np.sum(c[:,j])),
             'mean_mass':float(means[j]),'se_mass':None if errors[j] is None else float(errors[j]),
             'intensity':float(means[j]/(b-a)),
             'leading_prediction':predicted_bin_mass(a,b),
             'ratio_to_leading':float(means[j]/predicted_bin_mass(a,b))}
            for j,(a,b) in enumerate(zip(e[:-1],e[1:]))]
