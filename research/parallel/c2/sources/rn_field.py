"""Exact periodized SIDE24 Gaussian field: verified fixed-r pair-sector bounds.

All certificate arithmetic uses Arb balls. No floating point sample is a bound.
The fixed forms are conditioned/whitened by interval Cholesky. Spatial boxes
use grouped one-variable Taylor series so near-pin cancellation is performed
before interval evaluation. Infinite images, spectral tails, and normalization
denominators are enclosed. See RN_FIELD_PROOF.md for the precise scope.
"""
from flint import arb, arb_mat, ctx
from math import factorial, comb
from pathlib import Path
import hashlib, json, sys, time

ctx.prec = 384
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
R = arb(1)/20
HEIGHT = arb(6)/5
VMARK = HEIGHT-R**3/12 + arb(0,(R**3/12).upper())
N = 28
JETS = [(0,0),(1,0),(0,1)]
HJETS = [(2,0),(0,2),(1,1)]
RAW = [(p,a) for p in (-R/2,R/2) for a in JETS] + [(p,a) for p in (-R/2,R/2) for a in HJETS]

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def add_radius(x, radius):
    return x + arb(0, radius.abs_upper())

def largest(xs):
    return max(x.abs_upper() for x in xs)

def frob(A):
    return sum((x.abs_upper()**2 for x in A.entries()), arb(0)).sqrt().upper()

def eye(n):
    return arb_mat([[int(i==j) for j in range(n)] for i in range(n)])

def chol(A):
    n=A.nrows()
    L=arb_mat(n,n)
    for i in range(n):
        p=A[i,i]-sum((L[i,k]*L[i,k] for k in range(i)),arb(0))
        need(p>0, 'Cholesky pivot not certainly positive')
        L[i,i]=p.sqrt()
        for j in range(i+1,n):
            L[j,i]=(A[j,i]-sum((L[j,k]*L[i,k] for k in range(i)),arb(0)))/L[i,i]
    return L

def coeff_sum(n):
    a,b=1,1
    if n==0: return a
    for k in range(1,n): a,b=b,b+k*a
    return b

def image_tail(n, radius):
    radius=arb(radius)
    need(radius>=0 and radius<47, 'image-tail domain')
    first=coeff_sum(n)*(48+radius)**n*(-(48-radius)**2/2).exp()
    ratio=((72+radius)/(48+radius))**n*(-((72-radius)**2-(48-radius)**2)/2).exp()
    need(ratio<1, 'image-tail contraction')
    return (2*first/(1-ratio)).upper()

Z0=1+2*arb(-288).exp()
ZT=image_tail(0,0)
# Symmetric tail interval is slightly wider than the positive true tail.
Z=add_radius(Z0,ZT)
need(Z>0, 'positive image normalizer')

def kernel_all(s,nmax):
    s=arb(s)
    need(s.abs_upper()<18, 'kernel domain exceeded')
    vals=[arb(0) for _ in range(nmax+1)]
    for m in (-1,0,1):
        x=s+24*m
        g=(-x*x/2).exp()
        h0,h1=arb(1),x
        vals[0]+=g
        if nmax: vals[1]-=h1*g
        for n in range(1,nmax):
            h0,h1=h1,x*h1-n*h0
            vals[n+1]+=(-1)**(n+1)*h1*g
    return [add_radius(v,image_tail(n,18))/Z for n,v in enumerate(vals)]

def absolute_moments(nmax):
    h=arb.pi()/12
    ks=[j*h for j in range(117)]
    ws=[(-k*k/2).exp() for k in ks]
    z=ws[0]+2*sum(ws[1:],arb(0))
    result=[]
    for n in range(nmax+1):
        rho=arb(-235)*h*h/2
        ratio=rho.exp()*(arb(118)/117)**n
        need(ratio<1, 'spectral-tail contraction')
        tail=2*(-(117*h)**2/2).exp()*(117*h)**n/(1-ratio)
        val=(int(n==0)*ws[0]+2*sum((w*k**n for w,k in zip(ws[1:],ks[1:])),arb(0))+tail)/z.lower()
        result.append(val.upper())
    return result

MOM=absolute_moments(N+8)

def source_contract():
    files={
      'intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py':'85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930',
      'intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_perc.py':'5bc092412943e1c2185a8cf86cee23ae9cd9a2aad9bd9ef0b02807e9ff58dfa4',
      'intake/rn_source/K3_SIDE24_LB/UPPER2D/H2_foundations/cov_exact.py':'f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783',
      'intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md':'6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa',
      'output/rn_modulus/sqrt_chi2_modulus.py':'7ed331968cdec20e19a095788191968b6a4366ac20baffb1ecc02d92d0202aca'}
    for path,expected in files.items():
        need(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,'source hash mismatch: '+path)
    return files

class Field:
    def __init__(self):
        # Cov(d^a f(p),d^b f(q))=(-1)^|b| K^(a+b)(p-q).
        cache={}
        def k(dx,n):
            key=(dx.str(110),n)
            if key not in cache: cache[key]=kernel_all(dx,n)[n]
            return cache[key]
        G=arb_mat([[(-1)**sum(b)*k(p-q,a[0]+b[0])*k(arb(0),a[1]+b[1])
                    for q,b in RAW] for p,a in RAW])
        self.G=G
        G6=arb_mat([[G[i,j] for j in range(6)] for i in range(6)])
        X=arb_mat([[G[i+6,j] for j in range(6)] for i in range(6)])
        HH=arb_mat([[G[i+6,j+6] for j in range(6)] for i in range(6)])
        C=chol(G6)
        Ci=C.inv()
        Gi=G6.inv()
        S=HH-X*Gi*X.transpose()
        self.S0=S
        L=chol(S)
        W=L.inv()
        T=-W*X*Gi
        pv=arb_mat([[HEIGHT],[0],[0],[HEIGHT-R**3/6],[0],[0]])
        self.Hmean=X*Gi*pv
        mucoeff=Gi*pv
        self.forms=arb_mat(13,12)
        for i in range(6):
            for j in range(6):
                self.forms[i,j]=Ci[i,j]
                self.forms[i+6,j]=T[i,j]
                self.forms[i+6,j+6]=W[i,j]
        for j in range(6): self.forms[12,j]=mucoeff[j,0]
        self.YY=arb_mat([[(-1)**sum(b)*k(arb(0),a[0]+b[0])*k(arb(0),a[1]+b[1]) for b in JETS] for a in JETS])
        self.yy_floor=min(self.YY[i,i].lower() for i in range(3))
        need(self.yy_floor>0, 'YY diagonal floor')
        # Exact model identities; interval residuals must enclose zero.
        need((Ci*G6*Ci.transpose()-eye(6)).contains(arb_mat(6,6)), 'pin whitening residual')
        need((W*S*W.transpose()-eye(6)).contains(arb_mat(6,6)), 'pair whitening residual')
        self.fixed=dict(pin_cholesky_pivots=[C[i,i].str(40) for i in range(6)],
                        pair_cholesky_pivots=[L[i,i].str(40) for i in range(6)],
                        maximum_form_weight=largest(self.forms.entries()).str(30),
                        yy_floor=self.yy_floor.str(40),
                        image_normalizer=Z.str(50))

    def x_stencils(self, center, halfwidth):
        center,halfwidth=arb(center),arb(halfwidth)
        need(halfwidth>=0, 'negative halfwidth')
        points=[kernel_all(center-p,N+7) for p,a in RAW]
        t=arb(0,halfwidth)
        ans={}
        for i in range(13):
            for bx in range(6):
                for ay in range(3):
                    terms=[(self.forms[i,j]*(-1)**sum(a), a[0], points[j])
                           for j,(p,a) in enumerate(RAW) if a[1]==ay and self.forms[i,j]!=0]
                    co=[sum((w*ks[ax+bx+n] for w,ax,ks in terms),arb(0))/factorial(n) for n in range(N+1)]
                    v=co[-1]
                    for c in reversed(co[:-1]): v=v*t+c
                    remainder=sum((w.abs_upper()*MOM[ax+bx+N+1] for w,ax,ks in terms),arb(0))*halfwidth**(N+1)/factorial(N+1)
                    ans[i,bx,ay]=add_radius(v,remainder)
        return ans

    def cell_mixed(self, xst, ycenter, halfwidth):
        yy=kernel_all(arb(ycenter)+arb(0,arb(halfwidth)),7)
        return {(i,bx,by):sum((xst[i,bx,ay]*yy[ay+by] for ay in range(3)),arb(0))
                for i in range(13) for bx in range(6) for by in range(6-bx)}

    def bounds(self, mixed):
        b,f,z=[],[],[]
        muY=[]
        def mat(i0,nn,dx,dy):
            return arb_mat([[mixed[i,dx+g[0],dy+g[1]] for g in JETS] for i in range(i0,i0+nn)])
        for n in range(5):
            bn,fn,zn,mun=arb(0),arb(0),arb(0),arb(0)
            for dx in range(n+1):
                dy=n-dx
                bn+=comb(n,dx)*frob(mat(0,6,dx,dy))
                fn+=comb(n,dx)*frob(mat(6,6,dx,dy))
                mz=-mat(12,1,dx,dy).transpose()
                mun+=comb(n,dx)*frob(mz)
                if n==0: mz[0,0]+=VMARK
                zn+=comb(n,dx)*frob(mz)
            b.append(bn.upper()); f.append(fn.upper()); z.append(zn.upper())
            muY.append(mun.upper())
        dmin=(self.yy_floor-b[0]**2).lower()
        need(dmin>0, 'conditional y covariance not certified positive')
        d=[arb(0)]+[sum((comb(n,j)*b[j]*b[n-j] for j in range(n+1)),arb(0)).upper() for n in range(1,5)]
        inv=[(1/dmin).upper()]
        for n in range(1,5): inv.append((inv[0]*sum((comb(n,j)*d[j]*inv[n-j] for j in range(1,n+1)),arb(0))).upper())
        fj=[sum((comb(n,j)*f[j]*inv[n-j] for j in range(n+1)),arb(0)).upper() for n in range(5)]
        a=[sum((comb(n,j)*fj[j]*f[n-j] for j in range(n+1)),arb(0)).upper() for n in range(5)]
        m=[sum((comb(n,j)*fj[j]*z[n-j] for j in range(n+1)),arb(0)).upper() for n in range(5)]
        need(a[0]<1, 'finite chi-square domain not certified')
        return dict(B=b,F=f,z=z,A=a,m=m,muY=muY,D_floor=dmin)

    def pair_weight(self):
        # Gaussian raw-moment recurrence, det = Hxx*Hyy-Hxy^2.
        def det4(offset):
            from functools import lru_cache
            mu=[self.Hmean[offset+i,0] for i in range(3)]
            cov=[[self.S0[offset+i,offset+j] for j in range(3)] for i in range(3)]
            @lru_cache(None)
            def moment(alpha):
                if sum(alpha)==0: return arb(1)
                i=next(i for i,v in enumerate(alpha) if v)
                beta=list(alpha);beta[i]-=1
                value=mu[i]*moment(tuple(beta))
                for j in range(3):
                    if beta[j]:
                        gamma=beta[:];gamma[j]-=1
                        value+=beta[j]*cov[i][j]*moment(tuple(gamma))
                return value
            val=sum(((-1)**j*comb(4,j)*moment((4-j,4-j,2*j)) for j in range(5)),arb(0))
            need(val>0, 'det fourth moment positive')
            return val
        dm,ds=det4(0),det4(3)
        return (dm*ds).sqrt().sqrt(),dm,ds

def norm_derivative_envelope(a,c):
    # Same raw-derivative identities as prior rnu_white.norm_envelope, now Arb.
    u=[sum((comb(n,j)*a[j]*a[n-j] for j in range(n+1)),arb(0)) for n in range(5)]
    h,b=[1/(1-a[0]**2)],[1/(1-a[0])]
    for n in range(1,5):
        h.append(h[0]*sum((comb(n,j)*u[j]*h[n-j] for j in range(1,n+1)),arb(0)))
        b.append(b[0]*sum((comb(n,j)*a[j]*b[n-j] for j in range(1,n+1)),arb(0)))
    ell=[3*a[0]**2/(1-a[0]**2)+c[0]**2/(1-a[0])]
    for n in range(1,5):
        det=3*sum((comb(n-1,j)*h[j]*u[n-j] for j in range(n)),arb(0))
        mean=sum((factorial(n)//(factorial(i)*factorial(j)*factorial(n-i-j))*c[i]*b[j]*c[n-i-j]
                  for i in range(n+1) for j in range(n-i+1)),arb(0))
        ell.append(det+mean)
    e=ell[0].exp()
    X=[e-1,e*ell[1],e*(ell[2]+ell[1]**2),
       e*(ell[3]+3*ell[1]*ell[2]+ell[1]**3),
       e*(ell[4]+4*ell[1]*ell[3]+3*ell[2]**2+6*ell[1]**2*ell[2]+ell[1]**4)]
    return ell,X

def density_ratio_bounds(bounds, dmin, a2):
    """Uniform density/window ratios and orders 1..4, no CDF subtraction."""
    b,u=bounds['B'],bounds['muY']
    c0=b[0]**2
    gmin=a2.lower()-c0
    need(gmin>0 and dmin>0,'density denominator floor')
    d=[arb(0)]+[sum((comb(n,j)*b[j]*b[n-j] for j in range(n+1)),arb(0)) for n in range(1,5)]
    c=[c0]+d[1:]
    j=[1/gmin]
    for n in range(1,5):j.append(j[0]*sum((comb(n,k)*d[k]*j[n-k] for k in range(1,n+1)),arb(0)))
    def triple(X,Y,Z,n):
        return sum((factorial(n)//(factorial(i)*factorial(k)*factorial(n-i-k))*X[i]*Y[k]*Z[n-i-k]
                    for i in range(n+1) for k in range(n-i+1)),arb(0))
    mu=[u[n]+triple(c,j,u,n) for n in range(5)]
    sig=[arb(1)]+[d[n]+triple(c,j,c,n) for n in range(1,5)]
    inv=[1/dmin]
    for n in range(1,5):inv.append(inv[0]*sum((comb(n,k)*sig[k]*inv[n-k] for k in range(1,n+1)),arb(0)))
    lp=[arb(0)]
    lw=[arb(0)]
    w=[HEIGHT+mu[0]]+mu[1:]
    for n in range(1,5):
        lp.append(sum((comb(n-1,k)*j[k]*d[n-k] for k in range(n)),arb(0))+triple(u,j,u,n)/2)
        lw.append(sum((comb(n-1,k)*inv[k]*sig[n-k] for k in range(n)),arb(0))/2+triple(w,inv,w,n)/2)
    rp0=a2/gmin
    rw0=(HEIGHT*mu[0]).exp()/dmin.sqrt()
    rp_lower=(-u[0]**2/(2*gmin)).exp()
    rw_lower=(-(HEIGHT**2*(1-dmin)+2*HEIGHT*mu[0]+mu[0]**2)/(2*dmin)).exp()
    def bell(base,L):
        out=[base]
        for n in range(1,5):out.append(sum((comb(n-1,k)*L[k+1]*out[n-1-k] for k in range(n)),arb(0)))
        return out
    return dict(Rpg_lower=rp_lower.str(40),Rpg_upper=rp0.str(40),
                Rwm_lower=rw_lower.str(40),Rwm_upper=rw0.str(40),
                Rpg_raw_directional_derivative_absolute_bounds=[x.str(40) for x in bell(rp0,lp)],
                Rwm_raw_directional_derivative_absolute_bounds=[x.str(40) for x in bell(rw0,lw)],
                scope='all y in the covered annulus; Rwm integrates the whole mark window; derivatives at fixed pins')

def cell_intersects(ix,iy,den):
    # Exact rational arithmetic: cells [i/den,(i+1)/den] in both axes.
    lo=lambda i: 0 if i<=0<=i+1 else min(i*i,(i+1)*(i+1))
    hi=lambda i: max(i*i,(i+1)*(i+1))
    return lo(ix)+lo(iy)<=289*den*den and hi(ix)+hi(iy)>=25*den*den

def certified_cover(den=4):
    start=time.monotonic()
    pins=source_contract()
    field=Field()
    global_max={k:[arb(0)]*5 for k in ('B','F','z','A','m','muY')}
    floor=arb(1)
    count=0
    witnesses={}
    h=arb(1)/(2*den)
    for ix in range(-17*den,17*den):
        eligible=[iy for iy in range(-17*den,17*den) if cell_intersects(ix,iy,den)]
        if not eligible: continue
        xst=field.x_stencils(arb(2*ix+1)/(2*den),h)
        for iy in eligible:
            bnd=field.bounds(field.cell_mixed(xst,arb(2*iy+1)/(2*den),h))
            count+=1
            floor=min(floor,bnd['D_floor'])
            for k in global_max:
                for n in range(5):
                    if bnd[k][n]>global_max[k][n]:
                        global_max[k][n]=bnd[k][n]
                        witnesses[k+str(n)]=[ix,iy,den]
        if ix%8==0: print(json.dumps(dict(progress_x=ix,cells=count,elapsed=round(time.monotonic()-start,2))),flush=True)
    need(count==sum(cell_intersects(i,j,den) for i in range(-17*den,17*den) for j in range(-17*den,17*den)), 'coverage count mismatch')
    ell,chi=norm_derivative_envelope(global_max['A'],global_max['m'])
    sys.path.insert(0,str(ROOT/'output/rn_modulus'))
    from sqrt_chi2_modulus import uniform_speed_bound
    # A_j were bounded in Frobenius as well as spectral norm: the product
    # bounds used F_j Frobenius and D^-1 spectral norms throughout.
    speed=uniform_speed_bound(6,global_max['A'][0],global_max['m'][0],global_max['A'][1],global_max['m'][1])
    weight,dm,ds=field.pair_weight()
    scale=weight/arb('0.0077592917375327855')
    result=dict(status='PASS',scope='actual infinite-periodized pair Gaussian RN sector; fixed r=1/20, b=6/5; all y with 5<=|y|<=17 and all v in [b-r^3/6,b]',
                full_RN_UNIF_closed=False,all_small_r_closed=False,provider_independence=0,
                precision_bits=ctx.prec,Taylor_degree=N,grid_denominator=den,accepted_cells=count,
                arithmetic='python-flint Arb ball arithmetic',fixed=field.fixed,
                uniform_norms={k:[v.str(40) for v in vals] for k,vals in global_max.items()},
                conditional_y_covariance_floor=floor.str(40),
                logQ_raw_directional_derivative_absolute_bounds=[v.str(40) for v in ell],
                chi2_raw_directional_derivative_absolute_bounds=[v.str(40) for v in chi],
                sqrt_chi2_upper=chi[0].sqrt().str(40),sqrt_chi2_path_Lipschitz_upper=speed.str(40),
                mark_window=VMARK.str(40),derivatives='raw y-directional derivatives, each fixed mark v in the window',
                pair_det_fourth_moments=[dm.str(40),ds.str(40)],pair_weight_hL2=weight.str(40),
                kappa_pair_upper=(scale*chi[0].sqrt()).str(40),kappa_pair_path_Lipschitz_upper=(scale*speed).str(40),
                kappa_pair_condition='uses the separately pinned H3 lower bound Z_r>=0.0077592917375327855',
                density_ratios=density_ratio_bounds(global_max,floor,field.YY[1,1]),
                worst_cells=witnesses,elapsed_seconds=round(time.monotonic()-start,3),
                source_contract_sha256=pins,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'RN_FIELD_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    if '--cover' in sys.argv: certified_cover()
    else:
        f=Field()
        for x,y,h in [('5','0','0'),('5','0','1/4'),('17/4','11/4','1/4')]:
            b=f.bounds(f.cell_mixed(f.x_stencils(x,h),y,h))
            print(x,y,h,{k:([v.str(12) for v in val] if isinstance(val,list) else val.str(12)) for k,val in b.items()})
