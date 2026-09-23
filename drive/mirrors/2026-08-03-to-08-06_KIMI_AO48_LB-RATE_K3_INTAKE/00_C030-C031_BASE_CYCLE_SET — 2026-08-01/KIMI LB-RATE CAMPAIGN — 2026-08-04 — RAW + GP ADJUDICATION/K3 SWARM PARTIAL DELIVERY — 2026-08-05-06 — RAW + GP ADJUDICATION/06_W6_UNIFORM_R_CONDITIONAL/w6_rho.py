# w6_rho.py -- WP intensity rho(y;r) = p_grad(0) * E_win(y;r) with the G1-frozen exact
# integrand: E_win = E[ |det H| 1{det H<0} 1{b-ell < F < b} | 9 pins, grad f(y) = 0 ].
# Decomposition: F-slice outer (Gauss-Legendre on the window), (A,B) Gauss-Hermite outer,
# C-inner closed form E[(C^2-q)_+]. mpmath conditioning; numpy quadrature.
import sys
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from w6_kernel import *
from w6_analysis import *

TGTS6 = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]  # f, g1, g2, H11, H12, H22

def law_FH(y, r):
    """4-dim law of (F, H11, H12, H22) at y given 9 pins and grad f(y) = 0.
    Returns (mF, sF2, mH(3), SH(3x3)) as numpy floats."""
    pins = pin_list(r); vals = pin_values(r, mu_t(r))
    mean, cov, G, Gi = cond_mean_cov(pins, vals, y, TGTS6)
    # condition on grad = 0 (indices 1,2)
    idxF, idxH = 0, [3,4,5]
    idxG = [1,2]
    mG = np.array([mean[1], mean[2]], float)
    SGG = np.array([[cov[1,1],cov[1,2]],[cov[2,1],cov[2,2]]], float)
    mF = float(mean[0]); mH = np.array([mean[3],mean[4],mean[5]], float)
    SFG = np.array([[cov[0,1],cov[0,2]]], float)              # 1x2
    SHG = np.array([[cov[3,1],cov[3,2]],[cov[4,1],cov[4,2]],[cov[5,1],cov[5,2]]], float)  # 3x2
    SFF = float(cov[0,0])
    SHH = np.array([[cov[3,3],cov[3,4],cov[3,5]],[cov[4,3],cov[4,4],cov[4,5]],[cov[5,3],cov[5,4],cov[5,5]]], float)
    SGGi = np.linalg.inv(SGG)
    # p_grad(0)
    detSG = float(np.linalg.det(SGG))
    pgrad = np.exp(-0.5*mG@SGGi@mG)/(2*np.pi*np.sqrt(detSG))
    # conditional law given grad = 0
    mF2 = mF - float(SFG@SGGi@mG)
    mH2 = mH - SHG@SGGi@mG
    SFH = np.array([[cov[0,3],cov[0,4],cov[0,5]]], float)     # 1x3
    SFF2 = SFF - float(SFG@SGGi@SFG.T)
    SHH2 = SHH - SHG@SGGi@SHG.T
    SFH2 = SFH - SFG@SGGi@SHG.T                               # Cov(F,H | grad=0), 1x3
    return mF2, SFF2, mH2, SHH2, SFH2, pgrad

def Eq_minus_T2(mu, sg, q):
    """E[(q - T^2)_+] for T ~ N(mu, sg^2), q >= 0."""
    from math import erf, exp, pi as _pi, sqrt as _sqrt
    Phi = lambda x: 0.5*(1+erf(x/_sqrt(2)))
    phi = lambda x: exp(-x*x/2)/_sqrt(2*_pi)
    if q <= 0:
        return 0.0
    t = _sqrt(q)
    al = (-t-mu)/sg; be = (t-mu)/sg
    Pin = Phi(be) - Phi(al)
    ET2 = (mu*mu+sg*sg)*Pin + 2*mu*sg*(phi(al)-phi(be)) + sg*sg*(al*phi(al) - be*phi(be))
    return q*Pin - ET2

def G_of_u(u, mF, sF2, mH, SH, SFH, nH=48):
    """G(u) = E[(D^2+C^2-T^2)_+ | F=u], H|F=u ~ N(mH(u), SHF).
    Coordinates: H = (A,B,C) -> T=(A+B)/2, D=(A-B)/2, C=C; det = T^2-D^2-C^2."""
    cH = SFH.flatten()/sF2            # Cov(H,F)/Var(F)
    mHu = mH + cH*(u - mF)
    SHF = SH - np.outer(SFH.flatten(), SFH.flatten())/sF2
    # transform to (T,D,C); mHu ordering is (f_xx, f_xy, f_yy) = (A, C, B)
    M = np.array([[0.5,0.0,0.5],[0.5,0.0,-0.5],[0.0,1.0,0.0]])
    mTDC = M@mHu
    STDC = M@SHF@M.T
    # marginal of (D,C): indices 1,2 ; conditional T | D,C
    mD, mC = mTDC[1], mTDC[2]
    SDD, SDC, SCC = STDC[1,1], STDC[1,2], STDC[2,2]
    mT = mTDC[0]
    STD = np.array([STDC[0,1], STDC[0,2]])
    SDDCC = np.array([[SDD, SDC],[SDC, SCC]])
    SDDCCi = np.linalg.inv(SDDCC)
    def givDC(d, c):
        muT = mT + STD@SDDCCi@np.array([d-mD, c-mC])
        sgT2 = STDC[0,0] - STD@SDDCCi@STD
        sgT = np.sqrt(max(sgT2, 1e-300))
        return Eq_minus_T2(muT, sgT, d*d + c*c)
    # Gauss-Hermite over (D,C) ~ N((mD,mC), SDDCC): whiten
    ev, Q = np.linalg.eigh(SDDCC)
    ev = np.maximum(ev, 1e-300)
    xg, wg = hermgauss(nH)
    tot = 0.0
    sq = np.sqrt(ev)
    for i in range(nH):
        for j in range(nH):
            z1, z2 = xg[i]*np.sqrt(2), xg[j]*np.sqrt(2)
            pt = np.array([mD,mC]) + Q@(sq*np.array([z1,z2]))
            tot += wg[i]*wg[j]*givDC(pt[0], pt[1])
    return tot/np.pi

def E_win(y, r, nH=32, nL=16):
    mF, sF2, mH, SH, SFH, pgrad = law_FH(y, r)
    ell = float(r)**3/6
    b = 1.2
    xg, wg = leggauss(nL)
    us = (b-ell) + ell*(xg+1)/2
    sF = np.sqrt(sF2)
    tot = 0.0
    for k in range(nL):
        u = float(us[k])
        phi_u = np.exp(-0.5*((u-mF)/sF)**2)/(np.sqrt(2*np.pi)*sF)
        tot += wg[k]*phi_u*G_of_u(u, mF, sF2, mH, SH, SFH, nH)
    Ew = tot*ell/2
    return pgrad*Ew, pgrad, Ew

if __name__ == '__main__':
    set_dps(80)
    r = mp.mpf('0.025')
    y = (mp.mpf(0), mp.mpf('0.5672'))
    rho, pgrad, Ew = E_win(y, r)
    print(f"probe y=(0,0.5672), r=0.025: E_win={Ew:.12e}  rho={rho:.12e}")
    print("W4: E_win=2.020901410980e-05  rho_max~1.053e-4")
