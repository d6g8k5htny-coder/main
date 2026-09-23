# w6_analysis.py -- W6 analysis helpers: combo moments, limit jet, limit law.
import mpmath as mp
from w6_kernel import *

d1, d2 = mp.mpf('-1.26'), mp.mpf('0.24')

# limit functional definitions (basis of S0 = ann(n))
F_DEFS = {
 'f':   {(0,0):1},
 'fx':  {(1,0):1},
 'fy':  {(0,1):1},
 'fxx': {(2,0):1},
 'fxy': {(1,1):1},
 'fyy': {(0,2):1},
 'fxxx':{(3,0):1},
 'A3':  {(3,0):d1**2,(2,1):2*d1*d2,(1,2):d2**2},
 'D3':  {(2,1):d1**2+mp.mpf('0.75'),(1,2):2*d1*d2,(0,3):d2**2},
}
F_NAMES = ['f','fx','fy','fxx','fxy','fyy','fxxx','A3','D3']

def combo_var_0(coefs):
    """Prior variance of a jet combo at 0."""
    v0 = mp.mpf(0)
    for al,ca in coefs.items():
        for be,cb in coefs.items():
            v0 += ca*cb*((-1)**(be[0]+be[1]))*K2d(0,0,al[0]+be[0],al[1]+be[1])
    return v0

def combo_E_Var_pins(r, coefs, vals=None):
    """E and Var of a jet combo at 0 given the 9 pins at rung r."""
    pins = pin_list(r)
    if vals is None:
        vals = pin_values(r, mu_t(r))
    G = gram(pins); Gi = G**-1
    v = mp.zeros(len(pins),1)
    for i,vi in enumerate(vals): v[i]=vi
    cc = mp.zeros(len(pins),1)
    for j,(sj,aj) in enumerate(pins):
        for al,co in coefs.items():
            cc[j] += co*cov_der(sj[0],sj[1],aj,al)
    mean = (cc.transpose()*Gi*v)[0]
    var = combo_var_0(coefs) - (cc.transpose()*Gi*cc)[0]
    return mean, var

def limit_gram():
    """Gram G0 of the 9 limit functionals."""
    defs = [F_DEFS[k] for k in F_NAMES]
    n = len(defs)
    G = mp.zeros(n,n)
    for i in range(n):
        for j in range(i,n):
            G[i,j] = combo_var_0_cross(defs[i], defs[j])
            G[j,i] = G[i,j]
    return G

def combo_var_0_cross(ca_, cb_):
    tot = mp.mpf(0)
    for al,ca in ca_.items():
        for be,cb in cb_.items():
            tot += ca*cb*((-1)**(be[0]+be[1]))*K2d(0,0,al[0]+be[0],al[1]+be[1])
    return tot

def kappa_vec(y, target_al):
    """kappa_k(y) = Cov(F_k, d^target_al f(y)) for the limit functionals."""
    kk = mp.zeros(9,1)
    for k,name in enumerate(F_NAMES):
        tot = mp.mpf(0)
        for al,co in F_DEFS[name].items():
            # Cov(d^al f(0), d^ta f(y)) = (-1)^|ta| d^{al+ta} K(0 - y) = d^{al+ta}K(y) * (-1)^|al|
            tot += co*cov_der(0 - y[0], 0 - y[1], al, target_al)
        kk[k] = tot
    return kk

def limit_mean_cov(y, targets, vstar):
    """Conditional mean/cov of derivatives `targets` at y given limit functionals = vstar."""
    G0 = limit_gram()
    G0i = G0**-1
    nt = len(targets)
    Szz = mp.zeros(nt,nt)
    KAP = mp.zeros(9,nt)
    for a,ta in enumerate(targets):
        for b,tb in enumerate(targets):
            Szz[a,b] = cov_der(0,0,ta,tb) if a<=b else Szz[b,a]
        KAP[:,a] = kappa_vec(y, ta)
    for a in range(nt):
        for b in range(a+1,nt):
            Szz[b,a] = Szz[a,b]
    v = mp.zeros(9,1)
    for i,vi in enumerate(vstar): v[i]=vi
    mean = (KAP.transpose()*G0i)*v
    cov = Szz - (KAP.transpose()*G0i)*KAP
    return mean, cov, G0, G0i
