# Independent numeric crosswalk (own code; does not import lens scripts).
# Field: P's literal 3D image sum K_L (P l.13-14) built through Fourier/Poisson route (P l.58-59),
# i.e. a different representation from the chain's real-space 1D image sum (061 (1)).
# D_u: own route = trace/traceless decomposition A=[[s+x,y],[y,s-x]] with general (correlated)
# covariance, exact truncated Gaussian moments in s, smooth 2D polar quadrature in (x,y).
# Chain D_t: LS-DER-060 (16) spectral formula.
import itertools, mpmath as mp
mp.mp.dps = 20

def tensor_fourier(L, M=60):
    # d^gamma K(0) for K(z)=sum_n a_n exp(2 pi i n.z/L), a_n = prod exp(-2pi^2 n_i^2/L^2)/Z  (P l.58-59)
    w = lambda m: mp.e**(-2*mp.pi**2*m*m/L**2)
    Z1 = mp.fsum(w(m) for m in range(-M, M+1))
    def mom(j):  # 1D: sum_m (2 pi i m/L)^j w(m)/Z1
        return mp.re(mp.fsum((2j*mp.pi*m/L)**j*w(m) for m in range(-M, M+1))/Z1)
    m1 = [mom(j) for j in range(7)]
    return lambda g: m1[g[0]]*m1[g[1]]*m1[g[2]]

def tensor_image3d(L, N=4):
    # P l.13-14 literally: 3D lattice sum of derivatives of exp(-|z+Ln|^2/2) at z=0
    def He(j, x):
        a, bb = mp.mpf(1), x
        if j == 0: return a
        for n in range(1, j): a, bb = bb, x*bb - n*a
        return bb
    pts = list(itertools.product(range(-N, N+1), repeat=3))
    den = mp.fsum(mp.e**(-L*L*(p[0]**2+p[1]**2+p[2]**2)/2) for p in pts)
    cache = {}
    def T(g):
        if g not in cache:
            cache[g] = mp.fsum((-1)**sum(g)*He(g[0], L*p[0])*He(g[1], L*p[1])*He(g[2], L*p[2])
                               *mp.e**(-L*L*(p[0]**2+p[1]**2+p[2]**2)/2) for p in pts)/den
        return cache[g]
    return T

def BF():
    m1 = [1, 0, -1, 0, 3, 0, -15]
    return lambda g: mp.mpf(m1[g[0]]*m1[g[1]]*m1[g[2]])

def dirderiv(T, vecs):
    s = mp.mpf(0)
    for idx in itertools.product(range(3), repeat=len(vecs)):
        c = mp.mpf(1)
        for v, i in zip(vecs, idx): c *= v[i]
        if c == 0: continue
        g = [0, 0, 0]
        for i in idx: g[i] += 1
        s += c*T(tuple(g))
    return s

def C(T, a, bvecs):  # Cov(d_a f, d_b f) = (-1)^{|b|} d^{a+b} K(0)
    return (-1)**len(bvecs)*dirderiv(T, a+bvecs)

def frame(u, seed):
    u = mp.matrix(u); u = u/mp.norm(u)
    a = mp.matrix(seed); a = a - (a.T*u)[0]*u; a = a/mp.norm(a)
    c = mp.matrix([u[1]*a[2]-u[2]*a[1], u[2]*a[0]-u[0]*a[2], u[0]*a[1]-u[1]*a[0]])
    return [list(u), list(a), list(c)]

def cm(T, F1, F2):
    return mp.matrix([[C(T, a, bb) for bb in F2] for a in F1])

def schur(A, B, Cc):
    return A - B*mp.inverse(Cc)*B.T

def D_own(Gam):
    # A11=s+x, A22=s-x, A12=y ; (s,x,y) = M q, q=(A11,A12,A22)
    Mm = mp.matrix([[0.5, 0, 0.5], [0.5, 0, -0.5], [0, 1, 0]])
    S = Mm*Gam*Mm.T
    Sss = S[0, 0]; Ssw = mp.matrix([[S[0, 1], S[0, 2]]]); Sww = mp.matrix([[S[1, 1], S[1, 2]], [S[2, 1], S[2, 2]]])
    Wi = mp.inverse(Sww); beta = Ssw*Wi; v = Sss - (beta*Ssw.T)[0]; sv = mp.sqrt(v)
    detW = mp.det(Sww)
    def inner(xx, yy):
        mu = beta[0]*xx + beta[1]*yy; R2 = xx*xx + yy*yy; R = mp.sqrt(R2)
        c = (-R - mu)/sv
        ph = mp.npdf(c); Ph = mp.ncdf(c)
        Mn = [Ph, -ph]
        for n in range(2, 5): Mn.append(-c**(n-1)*ph + (n-1)*Mn[n-2])
        # (s^2-R^2)^2 with s = mu + sv w : expand in w
        # s^2 = mu^2 + 2 mu sv w + v w^2 ; g = s^2 - R2
        g0, g1, g2 = mu*mu - R2, 2*mu*sv, v
        coeff = [g0*g0, 2*g0*g1, g1*g1 + 2*g0*g2, 2*g1*g2, g2*g2]
        return mp.fsum(coeff[n]*Mn[n] for n in range(5))
    def pw(xx, yy):
        q = (Wi[0, 0]*xx*xx + 2*Wi[0, 1]*xx*yy + Wi[1, 1]*yy*yy)
        return mp.e**(-q/2)/(2*mp.pi*mp.sqrt(detW))
    f = lambda rho, th: inner(rho*mp.cos(th), rho*mp.sin(th))*pw(rho*mp.cos(th), rho*mp.sin(th))*rho
    # Gauss-Legendre product rule: rho on [0,14] split in 4 panels, theta periodic trapezoid
    import itertools as _it
    nodes, weights = gl_nodes()
    tot = mp.mpf(0)
    for (lo, hi) in [(0, 1.5), (1.5, 3.5), (3.5, 7), (7, 14)]:
        for xg, wg in zip(nodes, weights):
            rho = (hi - lo)/2*xg + (hi + lo)/2; wr = (hi - lo)/2*wg
            NT = 64
            s = mp.fsum(f(rho, 2*mp.pi*j/NT) for j in range(NT))*2*mp.pi/NT
            tot += wr*s
    return tot

_GL = {}
def gl_nodes(n=40):
    if n not in _GL:
        xs, ws = [], []
        for i in range(1, n+1):
            x = mp.cos(mp.pi*(i - mp.mpf(1)/4)/(n + mp.mpf(1)/2))
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for kk in range(2, n+1):
                    p0, p1 = p1, ((2*kk-1)*x*p1 - (kk-1)*p0)/kk
                dp = n*(x*p1 - p0)/(x*x - 1)
                dx = p1/dp; x -= dx
                if abs(dx) < mp.mpf(10)**(-mp.mp.dps+2): break
            xs.append(x); ws.append(2/((1 - x*x)*dp*dp))
        _GL[n] = (xs, ws)
    return _GL[n]

def D_060(Gam):
    H = mp.matrix([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])
    ev = sorted([mp.re(e) for e in mp.eig(Gam*H, left=False, right=False)])
    lam, mu, nu = ev[2], -ev[0], -ev[1]
    h = lambda p: mu*mp.cos(p)**2 + nu*mp.sin(p)**2
    I = mp.quad(lambda p: h(p)**2.5/mp.sqrt(lam + h(p)), [0, mp.pi/2])
    return 1.5*lam**2 - lam*(mu+nu) + (3*mu**2 + 2*mu*nu + 3*nu**2)/2 - 8/mp.pi*I

def integrands(T, u, seedP, seedC):
    # P side, frame seedP
    uu, y1, y2 = frame(u, seedP)
    G = [[uu], [y1], [y2]]; t = [uu, uu, uu]
    V = [[uu, uu], [uu, y1], [uu, y2]]; A = [[y1, y1], [y1, y2], [y2, y2]]
    CG = cm(T, G, G); CV = cm(T, V, V)
    tau2 = cm(T, [t], [t])[0, 0] - (cm(T, [t], G)*mp.inverse(CG)*cm(T, G, [t]))[0, 0]
    Gam = schur(cm(T, A, A), cm(T, A, V), CV)
    pG = (2*mp.pi)**-1.5/mp.sqrt(mp.det(CG)); pV = (2*mp.pi)**-1.5/mp.sqrt(mp.det(CV))
    DP = D_own(Gam)
    IP = mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi))*pG*pV*tau2**(mp.mpf(2)/3)*DP
    # chain side LS-DER-056 (2)-(13),(24)-(25), different transverse frame seedC; u_t = -f_ttt/12
    tt, n1, n2 = frame(u, seedC)
    g = [[tt], [n1], [n2]]; h = [[tt, tt], [tt, n1], [tt, n2]]; q = [[n1, n1], [n1, n2], [n2, n2]]
    Sg = cm(T, g, g); Sh = cm(T, h, h)
    s_u = C(T, [tt, tt, tt], [tt, tt, tt])/144
    c_ug = mp.matrix([[-C(T, [tt, tt, tt], gi)/12 for gi in g]])
    sig2 = s_u - (c_ug*mp.inverse(Sg)*c_ug.T)[0]
    Gc = schur(cm(T, q, q), cm(T, q, h), Sh)
    DC = D_060(Gc)
    B3 = 3*mp.gamma(mp.mpf(7)/6)/(2**(mp.mpf(4)/3)*mp.pi**3.5)
    IC = B3*sig2**(mp.mpf(2)/3)*DC/mp.sqrt(mp.det(Sg)*mp.det(Sh))
    # mirrored direction -u (P side) for evenness
    return dict(IP=IP, IC=IC, DP=DP, DC=DC, tau2=tau2, sig2=sig2, pGpV=pG*pV,
                detprod=mp.det(Sg)*mp.det(Sh))

if __name__ == '__main__':
    print('BF: D_own =', D_own(mp.matrix([[mp.mpf(8)/3, 0, mp.mpf(2)/3], [0, 1, 0], [mp.mpf(2)/3, 0, mp.mpf(8)/3]])),
          ' 29/6-sqrt6 =', mp.mpf(29)/6 - mp.sqrt(6))
    for L in (mp.mpf('2.5'), mp.mpf(3)):
        TF = tensor_fourier(L); TI = tensor_image3d(L)
        md = max(abs(TF(g) - TI(g)) for g in itertools.product(range(7), repeat=3) if sum(g) <= 6)
        print(f'L={L}: Fourier-route vs 3D-image-route max derivative diff (|g|<=6) = {mp.nstr(md, 3)}')
        for u, sP, sC in [((1, 0, 0), (0, 1, 0), (0, 0.6, 0.8)), ((1, 1, 1), (1, 0, 0), (0, 1, 0)),
                          ((0.2, -0.5, 0.9), (1, 0, 0), (0.3, 0.9, -0.1)), ((-0.7, 0.4, 0.1), (0, 0, 1), (0.5, 0.5, 0.2))]:
            R = integrands(TF, u, sP, sC)
            Rm = integrands(TF, tuple(-c for c in u), sC, sP)
            print(f'  u={u}: I_P={mp.nstr(R["IP"], 18)} I_C={mp.nstr(R["IC"], 18)} I_P/I_C-1={mp.nstr(R["IP"]/R["IC"]-1, 3)}'
                  f' D_own/D_060-1={mp.nstr(R["DP"]/R["DC"]-1, 3)} tau2/(144 sig2)-1={mp.nstr(R["tau2"]/(144*R["sig2"])-1, 3)}'
                  f' I_P(-u)/I_P(u)-1={mp.nstr(Rm["IP"]/R["IP"]-1, 3)}')
