# w6_zone.py -- I_WP(r) over the wedge via polar trapezoid, several rungs (optimized).
import sys, time
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from w6_kernel import *
from w6_analysis import *
from w6_rho import G_of_u

set_dps(35)
TGTS = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
STAS = ['M','S','Y']

class RungLaw:
    def __init__(self, r):
        self.r = mp.mpf(r)
        self.pins = pin_list(self.r)
        self.vals = pin_values(self.r, mu_t(self.r))
        G = gram(self.pins)
        self.Gi = G**-1
        self.v = mp.zeros(9,1)
        for i,vi in enumerate(self.vals): self.v[i] = vi
        self.Szz = mp.zeros(6,6)
        for a in range(6):
            for b in range(a,6):
                self.Szz[a,b] = cov_der(0,0,TGTS[a],TGTS[b]); self.Szz[b,a] = self.Szz[a,b]
        self.M, self.S, self.Y = stations(self.r)
    def law6(self, y):
        # per-station kernel derivative table K1^(p)(s1) K1^(q)(s2), p,q in 0..3
        tabs = []
        for st in (self.M, self.S, self.Y):
            s1, s2 = st[0]-y[0], st[1]-y[1]
            k1 = [K1d(s1, p) for p in range(4)]
            k2 = [K1d(s2, q) for q in range(4)]
            tabs.append((k1,k2))
        C = mp.zeros(9,6)
        for j,(sj,aj) in enumerate(self.pins):
            k1,k2 = tabs[j//3]
            for a,ta in enumerate(TGTS):
                p, q = aj[0]+ta[0], aj[1]+ta[1]
                C[j,a] = ((-1)**(ta[0]+ta[1]))*k1[p]*k2[q]
        W = self.Gi*C
        mean = (W.transpose()*self.v)
        cov = self.Szz - (C.transpose()*W)
        return mean, cov

def point_rho(law, y, nH=24):
    from w6_rhofast import E_win_fast
    mean, cov = law.law6(y)
    m = np.array([float(mean[i]) for i in range(6)])
    S = np.array([[float(cov[i,j]) for j in range(6)] for i in range(6)])
    mG = m[1:3]; SGG = S[1:3,1:3]
    detSG = SGG[0,0]*SGG[1,1]-SGG[0,1]**2
    if detSG <= 0: return 0.0
    SGGi = np.linalg.inv(SGG)
    pgrad = np.exp(-0.5*mG@SGGi@mG)/(2*np.pi*np.sqrt(detSG))
    idxFH = [0,3,4,5]; idxG = [1,2]
    mFH = m[idxFH]; SFG = S[np.ix_(idxFH,idxG)]
    mFH2 = mFH - SFG@SGGi@mG
    SFH2 = S[np.ix_(idxFH,idxFH)] - SFG@SGGi@SFG.T
    mF, sF2 = mFH2[0], SFH2[0,0]
    if sF2 <= 0: return 0.0
    mH = mFH2[1:4]; SH = SFH2[1:4,1:4]; SFH = SFH2[0,1:4].reshape(1,3)
    rho,_,_,_ = E_win_fast(mF, sF2, mH, SH, SFH, pgrad, float(law.r), nH)
    return rho

def zone_integral(r, th_lo=55, th_hi=125, nth=36, r_lo=0.2, r_hi=2.6, nr=49):
    law = RungLaw(r)
    ths = np.linspace(np.radians(th_lo), np.radians(th_hi), nth)
    rs = np.linspace(r_lo, r_hi, nr)
    A = np.zeros((nth, nr))
    t0 = time.time()
    for i,th in enumerate(ths):
        ct, st = mp.cos(mp.mpf(th)), mp.sin(mp.mpf(th))
        for j,rr in enumerate(rs):
            y = (mp.mpf(rr)*ct, mp.mpf(rr)*st)
            A[i,j] = point_rho(law, y)
    I = np.trapezoid(np.trapezoid(A*rs[None,:], rs, axis=1), ths, axis=0)
    print(f"r={r}: I_wedge={I:.10e}  ({time.time()-t0:.0f}s)  peak={A.max():.3e}", flush=True)
    return I

if __name__ == '__main__':
    r = sys.argv[1] if len(sys.argv)>1 else '0.025'
    zone_integral(r)
