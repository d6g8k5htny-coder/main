# w6_cs.py -- corrected global CS upper bound (W2 Thm W2-7, valid):
#   rho(y;r) <= rho_CS = p_grad(0) * sqrt(E[det^2 H | grad=0]) * sqrt(P_W),
# I_CS(r) = int_{B3} rho_CS dy.  Validity: CS chain with the square root present.
import sys, time
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from w6_zone import RungLaw
from w6_rhofast import Phi_vec
from w6_kernel import set_dps

def Edet2(mH, SH):
    """E[det^2 H] for H=(A,C,B) = (f_xx, f_xy, f_yy), det = A*B - C^2. Isserlis (W2-9)."""
    mA,mC,mB = mH
    S = SH
    SAA,SAB_ = S[0,0],S[0,2]
    SAC,SBC = S[0,1],S[1,2]
    SBB,SCC = S[2,2],S[1,1]
    EAABB = mA*mA*mB*mB + (SAA*mB*mB + 4*SAB_*mA*mB + SBB*mA*mA) + (SAA*SBB + 2*SAB_**2)
    EC4 = mC**4 + 6*SCC*mC*mC + 3*SCC*SCC
    EABC2 = (mA*mB*mC*mC + (SAB_*mC*mC + 2*SAC*mB*mC + 2*SBC*mA*mC + SCC*mA*mB)
             + (2*SAC*SBC + SAB_*SCC))
    return EAABB - 2*EABC2 + EC4

def point_rhoCS(law, y):
    mean, cov = law.law6(y)
    m = np.array([float(mean[i]) for i in range(6)])
    S = np.array([[float(cov[i,j]) for j in range(6)] for i in range(6)])
    mG = m[1:3]; SGG = S[1:3,1:3]
    detSG = SGG[0,0]*SGG[1,1]-SGG[0,1]**2
    if detSG <= 0: return 0.0
    SGGi = np.linalg.inv(SGG)
    pgrad = np.exp(-0.5*mG@SGGi@mG)/(2*np.pi*np.sqrt(detSG))
    idxFH = [0,3,4,5]; idxG=[1,2]
    mFH = m[idxFH]; SFG = S[np.ix_(idxFH,idxG)]
    mFH2 = mFH - SFG@SGGi@mG
    SFH2 = S[np.ix_(idxFH,idxFH)] - SFG@SGGi@SFG.T
    mF, sF2 = mFH2[0], SFH2[0,0]
    if sF2 <= 0: return 0.0
    mH = mFH2[1:4]; SH = SFH2[1:4,1:4]
    e2 = Edet2(mH, SH)
    if e2 < 0: e2 = 0.0
    ell = float(law.r)**3/6; b = 1.2
    sF = np.sqrt(sF2)
    PW = float(Phi_vec(np.array((b-mF)/sF)) - Phi_vec(np.array((b-ell-mF)/sF)))
    return pgrad*np.sqrt(e2)*np.sqrt(max(PW,0.0))

def I_CS(r, nth=48, nr=46, r_lo=0.06, r_hi=3.0):
    law = RungLaw(r)
    ths = np.linspace(0, 2*np.pi, nth, endpoint=False)
    rs = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), nr))
    tot = 0.0
    for i,th in enumerate(ths):
        ct, st = mp.cos(mp.mpf(th)), mp.sin(mp.mpf(th))
        for j,rr in enumerate(rs):
            rho = point_rhoCS(law, (mp.mpf(rr)*ct, mp.mpf(rr)*st))
            wth = 2*np.pi/nth
            if j == 0: wr = (rs[1]-rs[0])/2
            elif j == nr-1: wr = (rs[-1]-rs[-2])/2
            else: wr = (rs[j+1]-rs[j-1])/2
            tot += rho*rr*wth*wr
    return tot

if __name__ == '__main__':
    set_dps(45)
    import sys
    for r in (sys.argv[1:] or ['0.05','0.025','0.0125']):
        t0=time.time()
        I = I_CS(r)
        rf = float(r)
        print(f"r={r}: I_CS={I:.10e}  I_CS/r^1.5={I/rf**1.5:.6e}  ({time.time()-t0:.0f}s)", flush=True)
