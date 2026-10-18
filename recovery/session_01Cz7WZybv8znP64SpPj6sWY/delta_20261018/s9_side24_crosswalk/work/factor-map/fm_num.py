"""factor-map lens, numeric: a deliberately NON-reference, anisotropic, even, stationary covariance.

K(z) = w1 exp(-z'M1 z/2) + w2 exp(-z'M2 z/2), K(0)=1, non-axis direction t=(1,2,2)/3.
Each source's objects are computed from that source's OWN definitions:
  chain  LS-DER-056 (2)-(13),(21)-(25): g_t, u_t=-f_ttt/12, h_t, q_t, sigma_t^2 (10), Gamma_t (13), B_3 (25), integrand of (24)
  P      Section 15 (15.1)-(15.2):        G, t_u=f_uuu, V_u, A_u, tau_u^2 (15.1), p_G(0), p_Vu(0), integrand of (15.2)
and the pre-integration (b,k) / (b,kappa) integrands with FULL 8-dim joint contact densities:
  P      (10.3),(11.3),(13.6): 4 k^(-2/3) pi_0(R;v_0) z_0 , v_0=(b,0,0,12k,0,0,0,0) on U0*=(f,f_x,f_xx,f_xxx,f_y1,f_xy1,f_y2,f_xy2)
  chain  LS-DER-046 (14),(19): (6^(2/3)/3) kappa^(-2/3) J_0, J_0=(1/6) p_0 kappa^2 E ; Y=(f,u_t,g_t,h_t) at (b,-kappa/6,0,..)
"""
import itertools
import mpmath as mp
from fractions import Fraction as Fr

mp.mp.dps = 30

w = [Fr(3, 5), Fr(2, 5)]
M1 = [[Fr(1), Fr(1, 5), Fr(0)], [Fr(1, 5), Fr(3, 2), Fr(1, 10)], [Fr(0), Fr(1, 10), Fr(4, 5)]]
M2 = [[Fr(2), Fr(-1, 3), Fr(1, 4)], [Fr(-1, 3), Fr(1), Fr(0)], [Fr(1, 4), Fr(0), Fr(3, 5)]]
Ms = [M1, M2]
t = [Fr(1, 3), Fr(2, 3), Fr(2, 3)]
n1 = [Fr(2, 3), Fr(1, 3), Fr(-2, 3)]
n2 = [Fr(-2, 3), Fr(2, 3), Fr(-1, 3)]

def qf(M, x, y):
    return sum(x[i]*M[i][j]*y[j] for i in range(3) for j in range(3))

def matchings(lst):
    if not lst:
        yield []
        return
    a = lst[0]
    for i in range(1, len(lst)):
        rest = lst[1:i] + lst[i+1:]
        for mt in matchings(rest):
            yield [(a, lst[i])] + mt

def DK(vecs):
    """directional derivative of K at 0 along the listed vectors (exact)."""
    if len(vecs) % 2:
        return Fr(0)
    n = len(vecs)//2
    tot = Fr(0)
    for wi, M in zip(w, Ms):
        s = Fr(0)
        for mt in matchings(list(range(len(vecs)))):
            p = Fr(1)
            for (i, j) in mt:
                p *= qf(M, vecs[i], vecs[j])
            s += p
        tot += wi*(-1)**n*s
    return tot

def cov(al, be):
    # E[d_al f(x) d_be f(x)] = (-1)^|be| D^{al+be} K(0); al,be are (coef, vector list)
    return al[0]*be[0]*(-1)**len(be[1])*DK(al[1] + be[1])

def V(*vs, c=Fr(1)):
    return (c, list(vs))

def covmat(A, B):
    return mp.matrix([[mp.mpf(cov(a, b).numerator)/cov(a, b).denominator for b in B] for a in A])

# sanity: K(0)=1, check SPD of M's
print('K(0) =', DK([]))
for M in Ms:
    MM = mp.matrix([[mp.mpf(x.numerator)/x.denominator for x in row] for row in M])
    print('eig M =', [mp.nstr(e, 6) for e in mp.eigsy(MM)[0]])

# ---------- chain objects (LS-DER-056)
g = [V(t), V(n1), V(n2)]
u = V(t, t, t, c=Fr(-1, 12))
h = [V(t, t), V(t, n1), V(t, n2)]
q = [V(n1, n1), V(n1, n2), V(n2, n2)]
Sg = covmat(g, g); Sh = covmat(h, h); Sq = covmat(q, q); Cqh = covmat(q, h)
su = covmat([u], [u])[0, 0]; cug = covmat([u], g)
sig2 = su - (cug*mp.inverse(Sg)*cug.T)[0, 0]                         # (10)
Gam = Sq - Cqh*mp.inverse(Sh)*Cqh.T                                   # (13)

# ---------- P objects (Section 15)
G = g
tu = V(t, t, t)
Vu = h
A = q
SG = covmat(G, G); SV = covmat(Vu, Vu)
tau2 = covmat([tu], [tu])[0, 0] - (covmat([tu], G)*mp.inverse(SG)*covmat(G, [tu]))[0, 0]   # (15.1)
pG = (2*mp.pi)**mp.mpf(-1.5)/mp.sqrt(mp.det(SG))
pV = (2*mp.pi)**mp.mpf(-1.5)/mp.sqrt(mp.det(SV))
print('tau^2 (P 15.1) =', mp.nstr(tau2, 20), '  144*sigma_t^2 (chain (10)) =', mp.nstr(144*sig2, 20),
      '  ref values would be 6 and 6')
print('det Sigma_g =', mp.nstr(mp.det(Sg), 15), ' det Sigma_h =', mp.nstr(mp.det(Sh), 15), ' (ref: 1 and 3)')

# ---------- cone moment D (same object in both: chain (18), P (15.1)); q=(p,w,s), Q=[[p,w],[w,s]]
def trunc_moments(alpha, beta, jmax=4):
    phi = lambda z: mp.npdf(z)
    Mo = [mp.ncdf(beta) - mp.ncdf(alpha), phi(alpha) - phi(beta)]
    for j in range(2, jmax + 1):
        Mo.append((j - 1)*Mo[j - 2] + alpha**(j - 1)*phi(alpha) - beta**(j - 1)*phi(beta))
    return Mo

def cone_moment(mu, C):
    """E[(ps-w^2)^2 1{Q<0}] for (p,w,s)~N(mu,C), via analytic w-integral and 2D quad over p<0,s<0."""
    ip, iw, is_ = 0, 1, 2
    Cps = mp.matrix([[C[ip, ip], C[ip, is_]], [C[is_, ip], C[is_, is_]]])
    Cps_inv = mp.inverse(Cps)
    cw = mp.matrix([[C[iw, ip], C[iw, is_]]])
    bw = cw*Cps_inv
    vw = C[iw, iw] - (bw*cw.T)[0, 0]
    sdw = mp.sqrt(vw)
    detc = mp.det(Cps)
    def integrand(p, s):
        dp, ds = p - mu[ip], s - mu[is_]
        quad = Cps_inv[0, 0]*dp*dp + 2*Cps_inv[0, 1]*dp*ds + Cps_inv[1, 1]*ds*ds
        dens = mp.exp(-quad/2)/(2*mp.pi*mp.sqrt(detc))
        P0 = p*s
        c = mp.sqrt(P0)
        mw = mu[iw] + bw[0, 0]*dp + bw[0, 1]*ds
        al, be = (-c - mw)/sdw, (c - mw)/sdw
        Mo = trunc_moments(al, be)
        # E[w^j 1{|w|<c}] with w = mw + sdw z
        Ew = []
        for j in (0, 2, 4):
            Ew.append(sum(mp.binomial(j, i)*mw**(j - i)*sdw**i*Mo[i] for i in range(j + 1)))
        return dens*(P0*P0*Ew[0] - 2*P0*Ew[1] + Ew[2])
    return mp.quad(integrand, [-mp.inf, 0], [-mp.inf, 0])

mp.mp.dps = 20
Dt = cone_moment([0, 0, 0], Gam)
print('D_t = D_u (anisotropic) =', mp.nstr(Dt, 15))
# reference-law control: Gamma_ref = (2/3)dd + (dd+dd) in (p,w,s) coords: Var p = Var s = 2+2/3, Var w = 1, Cov(p,s)=2/3
Gref = mp.matrix([[mp.mpf(8)/3, 0, mp.mpf(2)/3], [0, 1, 0], [mp.mpf(2)/3, 0, mp.mpf(8)/3]])
Dref = cone_moment([0, 0, 0], Gref)
print('control D_2 ref via same routine =', mp.nstr(Dref, 15), ' exact 29/6-sqrt(6) =', mp.nstr(mp.mpf(29)/6 - mp.sqrt(6), 15))

g76 = mp.gamma(mp.mpf(7)/6)
B3 = 3*g76/(mp.mpf(2)**(mp.mpf(4)/3)*mp.pi**mp.mpf(3.5))
I24 = B3*sig2**(mp.mpf(2)/3)*Dt/mp.sqrt(mp.det(Sg)*mp.det(Sh))
I152 = g76/(mp.mpf(24)**(mp.mpf(1)/3)*mp.sqrt(mp.pi))*pG*pV*tau2**(mp.mpf(2)/3)*Dt
print('integrand chain (24) =', mp.nstr(I24, 18), '  P (15.2) =', mp.nstr(I152, 18), '  ratio =', mp.nstr(I24/I152, 18))
I24ref = B3*(mp.mpf(1)/24)**(mp.mpf(2)/3)*Dref/mp.sqrt(3)
print('(for scale) reference-law integrand =', mp.nstr(I24ref, 12), ' -> anisotropic law is genuinely non-reference')

# ---------- full 8-dim joint contact densities, pre-integration factor map
U0s = [V(), V(t), V(t, t), V(t, t, t), V(n1), V(t, n1), V(n2), V(t, n2)]       # P (3.4)
Ych = [V(), u, V(t), V(n1), V(n2), V(t, t), V(t, n1), V(t, n2)]               # LS-DER-046/047/056 Y=(f,u,g,h)
SU = covmat(U0s, U0s); SY = covmat(Ych, Ych)
SUi = mp.inverse(SU); SYi = mp.inverse(SY)
def gdens(x, S, Si):
    xv = mp.matrix(x)
    return mp.exp(-(xv.T*Si*xv)[0]/2)/((2*mp.pi)**(len(x)/mp.mpf(2))*mp.sqrt(mp.det(S)))
def pi0(bb, kk):
    return gdens([bb, 0, 0, 12*kk, 0, 0, 0, 0], SU, SUi)
def p0(bb, ka):
    return gdens([bb, -ka/6, 0, 0, 0, 0, 0, 0], SY, SYi)
print('pointwise p_0(b,kappa=6k) / [12 pi_0(b,k)] at sample (b,k):')
for bb, kk in [(mp.mpf('-0.7'), mp.mpf('0.05')), (mp.mpf('0.3'), mp.mpf('0.2')), (mp.mpf('1.9'), mp.mpf('0.013'))]:
    print('   ', mp.nstr(bb, 3), mp.nstr(kk, 3), mp.nstr(p0(bb, 6*kk)/(12*pi0(bb, kk)), 20))
# factorization pi_0 = p_(f,V)(b,0) p_G(0) phi_tau(12k)   (P line 460)
SfV = covmat([V()] + Vu, [V()] + Vu)
for bb, kk in [(mp.mpf('0.3'), mp.mpf('0.2'))]:
    fac = gdens([bb, 0, 0, 0], SfV, mp.inverse(SfV))*pG*mp.npdf(12*kk, 0, mp.sqrt(tau2))
    print('P line 460 factorization ratio pi_0/(p_(f,V) p_G phi_tau) =', mp.nstr(pi0(bb, kk)/fac, 20))
# k- and kappa-integrals at fixed b (cone moment E is a common b-only factor; parity => no k-dependence)
for bb in (mp.mpf('-0.4'), mp.mpf('0.8')):
    IP = mp.quad(lambda kk: 4*kk**(-mp.mpf(2)/3)*pi0(bb, kk)*(6*kk)**2, [0, mp.inf])
    IC = mp.quad(lambda ka: (mp.mpf(6)**(mp.mpf(2)/3)/3)*ka**(-mp.mpf(2)/3)*mp.mpf(1)/6*p0(bb, ka)*ka**2, [0, mp.inf])
    closed = 144*gdens([bb, 0, 0, 0], SfV, mp.inverse(SfV))*pG*g76*tau2**(mp.mpf(2)/3)/(144*mp.mpf(24)**(mp.mpf(1)/3)*mp.sqrt(mp.pi))
    print(f'b={mp.nstr(bb,3)}: P int_k / chain int_kappa =', mp.nstr(IP/IC, 20),
          '; P int_k / [p_(f,V)(b,0) p_G(0) Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt pi)] =', mp.nstr(IP/closed, 20))

# ---------- birth integration (tower property, P line 463 / chain (17)) at modest precision
mp.mp.dps = 15
cf = covmat([V()] + Vu, A)   # Cov((f,V),A)
SfVi = mp.inverse(SfV)
SA = covmat(A, A)
CondA = SA - cf.T*SfVi*cf
def Eb(bb):
    mu = cf.T*SfVi*mp.matrix([bb, 0, 0, 0])
    return cone_moment([mu[0], mu[1], mu[2]], CondA)
pfV = lambda bb: gdens([bb, 0, 0, 0], SfV, SfVi)
vb = 1/SfVi[0, 0]   # Var(f | V=0)
nodes = mp.gauss_quadrature(24, 'hermite')
tot = mp.mpf(0)
for x, wt in zip([nodes[0][i] for i in range(nodes[0].rows)], [nodes[1][i] for i in range(nodes[1].rows)]):
    bb = x*mp.sqrt(2*vb)
    tot += wt*mp.exp(x**2)*pfV(bb)*Eb(bb)*mp.sqrt(2*vb)
print('int p_(f,V)(b,0) E[F|f=b,V=0] db =', mp.nstr(tot, 12), '  p_V(0) D_u =', mp.nstr(pV*Dt, 12),
      ' ratio =', mp.nstr(tot/(pV*Dt), 12))
