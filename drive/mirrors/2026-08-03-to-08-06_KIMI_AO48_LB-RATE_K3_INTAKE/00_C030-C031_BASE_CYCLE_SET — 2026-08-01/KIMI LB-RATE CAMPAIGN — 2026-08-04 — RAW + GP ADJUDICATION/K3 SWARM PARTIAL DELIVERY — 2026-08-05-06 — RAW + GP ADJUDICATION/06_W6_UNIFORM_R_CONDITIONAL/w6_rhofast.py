# w6_rhofast.py -- vectorized fast rho(y;r): E_win = P(window) * G(mid-window) * (1+O(ell^2)).
# G(u) varies by O(ell) over the window; midpoint-rule error O(ell^2) ~ 1e-12 relative.
import sys
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import erf

_HGx, _HGw = None, None
def _hg(nH):
    global _HGx, _HGw
    if _HGx is None or len(_HGx) != nH:
        _HGx, _HGw = hermgauss(nH)
    return _HGx, _HGw

def Phi_vec(x):
    return 0.5*(1+np.vectorize(erf)(x/np.sqrt(2)))

try:
    from scipy.special import ndtr as _ndtr
    def Phi_vec(x): return _ndtr(x)
except Exception:
    pass

def Eq_minus_T2_vec(mu, sg, q):
    """E[(q - T^2)_+] elementwise; mu,q arrays, sg scalar (sg>0)."""
    phi = lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
    out = np.zeros_like(q)
    pos = q > 0
    t = np.sqrt(np.where(pos, q, 1.0))
    al = (-t-mu)/sg; be = (t-mu)/sg
    Pin = Phi_vec(be) - Phi_vec(al)
    ET2 = (mu*mu+sg*sg)*Pin + 2*mu*sg*(phi(al)-phi(be)) + sg*sg*(al*phi(al) - be*phi(be))
    val = q*Pin - ET2
    return np.where(pos, val, 0.0)

def G_of_u_fast(u, mF, sF2, mH, SH, SFH, nH=24):
    cH = SFH.flatten()/sF2
    mHu = mH + cH*(u - mF)
    SHF = SH - np.outer(SFH.flatten(), SFH.flatten())/sF2
    M = np.array([[0.5,0.0,0.5],[0.5,0.0,-0.5],[0.0,1.0,0.0]])
    mTDC = M@mHu; STDC = M@SHF@M.T
    mD, mC = mTDC[1], mTDC[2]
    SDDCC = STDC[1:3,1:3]
    mT = mTDC[0]
    STD = STDC[0,1:3]
    SDDCCi = np.linalg.inv(SDDCC)
    sgT2 = STDC[0,0] - STD@SDDCCi@STD
    sgT = np.sqrt(max(sgT2, 1e-300))
    ev, Q = np.linalg.eigh(SDDCC)
    ev = np.maximum(ev, 1e-300)
    xg, wg = _hg(nH)
    X, Y = np.meshgrid(xg*np.sqrt(2), xg*np.sqrt(2), indexing='ij')
    W1, W2 = np.meshgrid(wg, wg, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)          # (nH^2, 2)
    dc = np.array([mD, mC]) + pts@(Q*np.sqrt(ev)).T          # (nH^2, 2)
    d, c = dc[:,0], dc[:,1]
    muT = mT + (dc - np.array([mD,mC]))@(SDDCCi@STD)
    R2 = d*d + c*c
    gvals = Eq_minus_T2_vec(muT, sgT, R2)
    return float(np.sum(W1.ravel()*W2.ravel()*gvals)/np.pi)

def E_win_fast(mF, sF2, mH, SH, SFH, pgrad, r_float, nH=24):
    ell = r_float**3/6; b = 1.2
    sF = np.sqrt(sF2)
    z1 = (b-mF)/sF; z0 = (b-ell-mF)/sF
    Pwin = float(Phi_vec(np.array(z1)) - Phi_vec(np.array(z0)))
    Gmid = G_of_u_fast(b-ell/2, mF, sF2, mH, SH, SFH, nH)
    return pgrad*Pwin*Gmid, pgrad, Pwin, Gmid
