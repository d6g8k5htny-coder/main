# Independent re-derivation (adversarial source-fidelity lens): build P (15.1)-(15.2) objects and
# LS-DER-056 (8)-(13),(18),(24)-(25) objects from their literal definitions at L=2.2 and BF,
# with D computed by a THIRD method (closed-form inner s-integral + 2D quadrature), not grid, not 060 spectral.
import itertools, mpmath as mp
mp.mp.dps = 30
def k1derivs(L, jmax=6, N=12):
    # j-th derivative at 0 of k_L(x)=sum_n exp(-(x+nL)^2/2)/sum_n exp(-(nL)^2/2)
    He = lambda j, a: mp.hermite(j, a/mp.sqrt(2)) * mp.mpf(2)**(-mp.mpf(j)/2)  # probabilists' He_j
    Z = mp.fsum(mp.e**(-(n*L)**2/2) for n in range(-N, N+1))
    return [mp.fsum((-1)**j * He(j, n*L) * mp.e**(-(n*L)**2/2) for n in range(-N, N+1))/Z for j in range(jmax+1)]
def bf_derivs(jmax=6):
    He = lambda j, a: mp.hermite(j, a/mp.sqrt(2)) * mp.mpf(2)**(-mp.mpf(j)/2)
    return [(-1)**j * He(j, 0) for j in range(jmax+1)]
def dK(m, gamma):  # product kernel derivative at 0 (LS-DER-061 (1)-(2); = P/PROOF Z^3 image sum by factorization)
    out = mp.mpf(1)
    for g in gamma: out *= m[g]
    return out
def functional(dirs):  # directional derivative partial_{dirs[0]}...partial_{dirs[-1]} f at 0 as {multi-index: coef}
    F = {}
    for combo in itertools.product(range(3), repeat=len(dirs)):
        c = mp.mpf(1)
        for d, i in zip(dirs, combo): c *= d[i]
        g = [0, 0, 0]
        for i in combo: g[i] += 1
        F[tuple(g)] = F.get(tuple(g), 0) + c
    return F
def cov(m, F, G):  # Cov(d^a f, d^b f) = (-1)^|b| d^(a+b) K(0)   (LS-DER-061 (3))
    s = mp.mpf(0)
    for a, ca in F.items():
        for b, cb in G.items():
            s += ca*cb*(-1)**sum(b)*dK(m, [a[i]+b[i] for i in range(3)])
    return s
def C(m, A, B): return mp.matrix([[cov(m, a, b) for b in B] for a in A])
def frame(u):
    u = mp.matrix(u); u = u/mp.norm(u)
    a = mp.matrix([1, 0, 0]) if abs(u[0]) < 0.9 else mp.matrix([0, 1, 0])
    n1 = a - (a.T*u)[0]*u; n1 = n1/mp.norm(n1)
    n2 = mp.matrix([u[1]*n1[2]-u[2]*n1[1], u[2]*n1[0]-u[0]*n1[2], u[0]*n1[1]-u[1]*n1[0]])
    return [list(u), list(n1), list(n2)]
def D_third(Gam):
    # q=(A11,A12,A22)~N(0,Gam); A=[[s+x,y],[y,s-x]], s=(A11+A22)/2, x=(A11-A22)/2, y=A12.
    T = mp.matrix([[0.5, 0, 0.5], [0.5, 0, -0.5], [0, 1, 0]]); Cs = T*Gam*T.T
    # condition s on (x,y): s|(x,y) ~ N(mu, v)
    Cxy = mp.matrix([[Cs[1,1], Cs[1,2]], [Cs[2,1], Cs[2,2]]]); cs = mp.matrix([[Cs[0,1], Cs[0,2]]])
    w = cs*mp.inverse(Cxy); v = Cs[0,0] - (w*cs.T)[0]
    detC = mp.det(Cxy); Ci = mp.inverse(Cxy)
    def inner(x, y):
        R2 = x*x+y*y; R = mp.sqrt(R2); mu = w[0]*x + w[1]*y
        # E[(s^2-R^2)^2 1{s<-R}] for s~N(mu,v): integrate polynomial against normal tail exactly
        sd = mp.sqrt(v); a = (-R-mu)/sd
        # moments M_k = E[z^k 1{z<a}] for standard z
        Phi = mp.ncdf(a); phi = mp.npdf(a)
        M = [Phi, -phi]
        for kk in range(2, 5): M.append((kk-1)*M[kk-2] - a**(kk-1)*phi)
        # s = mu + sd z ; (s^2-R^2)^2 expand in z
        p = [mu*mu-R2, 2*mu*sd, sd*sd]  # s^2-R^2 = p0+p1 z+p2 z^2
        sq = [p[0]**2, 2*p[0]*p[1], p[1]**2+2*p[0]*p[2], 2*p[1]*p[2], p[2]**2]
        val = mp.fsum(sq[i]*M[i] for i in range(5))
        dens = mp.e**(-0.5*(Ci[0,0]*x*x+2*Ci[0,1]*x*y+Ci[1,1]*y*y))/(2*mp.pi*mp.sqrt(detC))
        return val*dens
    return mp.quad(lambda rr, th: inner(rr*mp.cos(th), rr*mp.sin(th))*rr, [0, 2, 5, 12], [0, mp.pi/2, mp.pi, 3*mp.pi/2, 2*mp.pi])
def D_060(Gam):
    H = mp.matrix([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]]); ev = sorted([mp.re(e) for e in mp.eig(Gam*H)[0]])
    lam = ev[2]; mu, nu = -ev[0], -ev[1]
    h = lambda ph: mu*mp.cos(ph)**2 + nu*mp.sin(ph)**2
    return 1.5*lam**2 - lam*(mu+nu) + 0.5*(3*mu**2+2*mu*nu+3*nu**2) - 8/mp.pi*mp.quad(lambda ph: h(ph)**2.5/mp.sqrt(lam+h(ph)), [0, mp.pi/2])
def objects(m, u):
    t, n1, n2 = frame(u)
    G = [functional([t]), functional([n1]), functional([n2])]
    tu = functional([t, t, t])
    uu = {kk: -c/12 for kk, c in tu.items()}  # LS-DER-056 (2): u_t=-f_ttt/12
    V = [functional([t, t]), functional([t, n1]), functional([t, n2])]
    A = [functional([n1, n1]), functional([n1, n2]), functional([n2, n2])]
    CG = C(m, G, G); CV = C(m, V, V)
    # P (15.1): tau^2 = Var(t_u|G=0)
    ctg = C(m, [tu], G); tau2 = C(m, [tu], [tu])[0] - (ctg*mp.inverse(CG)*ctg.T)[0]
    # 056 (9)-(10): sigma^2 = s_u - c_ug Sigma_g^-1 c_gu
    cug = C(m, [uu], G); sig2 = C(m, [uu], [uu])[0] - (cug*mp.inverse(CG)*cug.T)[0]
    CAV = C(m, A, V); Gam = C(m, A, A) - CAV*mp.inverse(CV)*CAV.T   # P: Cov(A|V=0); 056 (13): Gamma_t
    odd_even = max(abs(cov(m, a, b)) for a in G+[tu] for b in V+A)
    return CG, CV, tau2, sig2, Gam, odd_even
def integrands(m, u):
    CG, CV, tau2, sig2, Gam, oe = objects(m, u)
    D3 = D_third(Gam); D6 = D_060(Gam)
    pG = (2*mp.pi)**-1.5/mp.sqrt(mp.det(CG)); pV = (2*mp.pi)**-1.5/mp.sqrt(mp.det(CV))
    IP = mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi)) * pG*pV*tau2**(mp.mpf(2)/3)*D3     # P (15.2) integrand
    B3 = 3*mp.gamma(mp.mpf(7)/6)/(mp.mpf(2)**(mp.mpf(4)/3)*mp.pi**3.5)                       # 056 (25)
    I56 = B3*sig2**(mp.mpf(2)/3)*D6/mp.sqrt(mp.det(CG)*mp.det(CV))                          # 056 (24) integrand (D via 060)
    return IP, I56, D3, D6, tau2/(144*sig2), oe
for label, m in [('L=2.2', k1derivs(mp.mpf('2.2'))), ('BF', bf_derivs())]:
    for u in [(0.3, -0.7, 0.2), (1, 1, 1), (1, 0, 0)]:
        IP, I56, D3, D6, rt, oe = integrands(m, [mp.mpf(x) for x in u])
        print(label, u, 'P(15.2)=', mp.nstr(IP, 20), ' 056(24)=', mp.nstr(I56, 20), ' ratio-1=', mp.nstr(IP/I56-1, 3),
              ' D_third=', mp.nstr(D3, 18), ' D_060=', mp.nstr(D6, 18), ' tau2/144sig2=', mp.nstr(rt, 18), ' odd-even max=', mp.nstr(oe, 3))
print('29/6-sqrt6 =', mp.nstr(mp.mpf(29)/6-mp.sqrt(6), 18))
B3 = 3*mp.gamma(mp.mpf(7)/6)/(mp.mpf(2)**(mp.mpf(4)/3)*mp.pi**3.5)
print('B3 vs Gamma(7/6)/(24^(1/3)sqrt pi)(2pi)^-3 144^(2/3): ratio-1 =', mp.nstr(B3/(mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi))*(2*mp.pi)**-3*mp.mpf(144)**(mp.mpf(2)/3))-1, 3))
