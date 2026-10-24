"""Objects-and-conventions lens: build each source's Gaussian objects from its OWN
definitions and compare.  Read-only crosswalk aid; not a review of either source.

P / PROOF side: K_L = 3D image sum (P lines 13-14, PROOF lines 18-19); objects of
P Sec. 15: G=grad f, t_u=d_u^3 f, V_u=H u, A_u transverse block, tau_u^2=Var(t_u|G=0),
D_u=E[(det A_u)^2 1{A_u<0}|V_u=0]; contact pins U0*=(f,f_x,f_xx,f_xxx,f_y,f_xy) (3.4),
target v_0=(b,0,0,12k,0..) (line 103), Jacobian 12 r^-(d+3) (3.2).

Chain side: k_L 1D normalized kernel, K_L = product (LS-DER-041 (1)-(2), 061 (1)-(3));
LS-DER-056 (2)-(13): u_t=-f_ttt/12, g_t, h_t, q_t, Sigma_g, sigma_t^2, Sigma_h, Gamma_t;
D_t by LS-DER-060 (16); pins Y=(f,-f_ttt/12,f_t,f_tt,f_n,f_tn) LS-DER-047 (2), target
(b,-kappa/6,0..) 047 (6)/056 (7), Jacobian r^-(d+3) (046 (4)).
"""
import itertools, math, random
import mpmath as mp

mp.mp.dps = 40

# ---------------------------------------------------------------- field derivative tensors
def He(j, x):
    # probabilists' Hermite: d^j/dx^j e^{-x^2/2} = (-1)^j He_j(x) e^{-x^2/2}
    if j == 0:
        return mp.mpf(1)
    a, b = mp.mpf(1), x
    for n in range(1, j):
        a, b = b, x*b - n*a
    return b

def chain_moments(L, N=8):
    # LS-DER-061 (1)-(2): m_j = k_L^{(j)}(0), 1D normalized periodized kernel
    den = mp.fsum(mp.e**(-(n*L)**2/2) for n in range(-N, N+1))
    return [mp.fsum((-1)**j*He(j, n*L)*mp.e**(-(n*L)**2/2) for n in range(-N, N+1))/den
            for j in range(7)]

def chain_tensor(L):
    m = chain_moments(L)
    return {g: m[g[0]]*m[g[1]]*m[g[2]] for g in itertools.product(range(7), repeat=3) if sum(g) <= 6}

def P_tensor(L, N=6):
    # P line 13-14: K_L(z)=sum_{n in Z^3} exp(-|z+Ln|^2/2)/sum_{n in Z^3} exp(-|Ln|^2/2),
    # summed over the 3D lattice cube directly (no use of the 1D product).
    pts = list(itertools.product(range(-N, N+1), repeat=3))
    den = mp.fsum(mp.e**(-(L*L)*(a*a+b*b+c*c)/2) for a, b, c in pts)
    out = {}
    for g in itertools.product(range(7), repeat=3):
        if sum(g) > 6:
            continue
        s = mp.fsum(((-1)**sum(g))*He(g[0], L*a)*He(g[1], L*b)*He(g[2], L*c)
                    * mp.e**(-(L*L)*(a*a+b*b+c*c)/2) for a, b, c in pts)
        out[g] = s/den
    return out

def BF_tensor():
    # K_infty = exp(-|x|^2/2): d^j at 0 = (-1)^j He_j(0)
    m = [(-1)**j*He(j, mp.mpf(0)) for j in range(7)]
    return {g: m[g[0]]*m[g[1]]*m[g[2]] for g in itertools.product(range(7), repeat=3) if sum(g) <= 6}

# ---------------------------------------------------------------- directional covariances
def T(D, dirs):
    if not dirs:
        return D[(0, 0, 0)]
    s = mp.mpf(0)
    for idx in itertools.product(range(3), repeat=len(dirs)):
        c = mp.mpf(1)
        for v, i in zip(dirs, idx):
            c *= v[i]
            if c == 0:
                break
        if c == 0:
            continue
        g = [0, 0, 0]
        for i in idx:
            g[i] += 1
        s += c*D[tuple(g)]
    return s

def cov(D, A, B):
    # A,B: linear functionals = list of (coef, [direction vectors]).  Cov(d^a f, d^b f) = (-1)^|b| d^{a+b}K(0)
    return mp.fsum(ca*cb*(-1)**len(db)*T(D, da+db) for ca, da in A for cb, db in B)

def covmat(D, Fs, Gs=None):
    Gs = Fs if Gs is None else Gs
    return mp.matrix([[cov(D, a, b) for b in Gs] for a in Fs])

def cond(C11, C12, C22):
    return C11 - C12*mp.inverse(C22)*C12.T

def frame(u):
    u = [mp.mpf(x) for x in u]
    nrm = mp.sqrt(mp.fsum(x*x for x in u)); u = [x/nrm for x in u]
    a = [1, 0, 0] if abs(u[0]) < 0.8 else [0, 1, 0]
    d = mp.fsum(x*y for x, y in zip(a, u)); n1 = [x - d*y for x, y in zip(a, u)]
    nn = mp.sqrt(mp.fsum(x*x for x in n1)); n1 = [x/nn for x in n1]
    n2 = [u[1]*n1[2]-u[2]*n1[1], u[2]*n1[0]-u[0]*n1[2], u[0]*n1[1]-u[1]*n1[0]]
    return u, n1, n2

F = lambda *dirs: [(mp.mpf(1), list(dirs))]

# ---------------------------------------------------------------- P Sec. 15 objects (own definitions)
def P_objects(D, u):
    u, y1, y2 = frame(u)
    G = [F(u), F(y1), F(y2)]                         # G = grad f in frame R (P line 449)
    t = F(u, u, u)                                   # t_u = d_u^3 f
    V = [F(u, u), F(u, y1), F(u, y2)]                # V_u = H u in frame: (f_xx, f_xy1, f_xy2)
    A = [F(y1, y1), F(y1, y2), F(y2, y2)]            # A_u transverse block entries (A11, A12, A22)
    CG = covmat(D, G); CV = covmat(D, V)
    tau2 = cond(covmat(D, [t]), covmat(D, [t], G), CG)[0, 0]   # (15.1)
    CA = cond(covmat(D, A), covmat(D, A, V), CV)                # Cov(A_u | V_u=0)
    pG = (2*mp.pi)**(-1.5)/mp.sqrt(mp.det(CG)); pV = (2*mp.pi)**(-1.5)/mp.sqrt(mp.det(CV))
    # parity: odd (G,t) vs even (f,V,A) cross-covariances
    odd, even = G+[t], [F()]+V+A
    par = max(abs(x) for x in covmat(D, odd, even))
    return dict(tau2=tau2, CA=CA, pG=pG, pV=pV, detG=mp.det(CG), detV=mp.det(CV), parity=par,
                frame=(u, y1, y2))

# ---------------------------------------------------------------- LS-DER-056 objects (own definitions)
def chain_objects(D, t, sign=-1):
    t, n1, n2 = frame(t)
    ut = [(mp.mpf(sign)/12, [t, t, t])]              # (2): u_t = -d_ttt f/12  (sign=-1)
    g = [F(t), F(n1), F(n2)]                         # (2)
    h = [F(t, t), F(t, n1), F(t, n2)]                # (3)
    q = [F(n1, n1), F(n1, n2), F(n2, n2)]            # (4)
    Sg = covmat(D, g); Sh = covmat(D, h)             # (8),(11)
    s_u = covmat(D, [ut])[0, 0]; c_ug = covmat(D, [ut], g)
    sig2 = s_u - (c_ug*mp.inverse(Sg)*c_ug.T)[0, 0]  # (10)
    Gam = cond(covmat(D, q), covmat(D, q, h), Sh)    # (13)
    return dict(sig2=sig2, Gam=Gam, Sg=Sg, Sh=Sh, c_ug=c_ug)

def D_spectral(Gam):
    # LS-DER-060 (16)
    H = mp.matrix([[0, 0, mp.mpf(1)/2], [0, -1, 0], [mp.mpf(1)/2, 0, 0]])
    ev = mp.eig(Gam*H, left=False, right=False)
    ev = sorted([mp.re(e) for e in ev])
    lam, mu, nu = ev[2], -ev[0], -ev[1]
    hfun = lambda p: mu*mp.cos(p)**2 + nu*mp.sin(p)**2
    I = mp.quad(lambda p: hfun(p)**2.5/mp.sqrt(lam+hfun(p)), [0, mp.pi/2])
    return mp.mpf(3)/2*lam**2 - lam*(mu+nu) + (3*mu**2+2*mu*nu+3*nu**2)/2 - 8/mp.pi*I

def D_direct(CA, Nz=1500, Nphi=3000):
    # P (15.1) definition read literally: E[(det A)^2 1{A<0}], A ~ N(0, CA) in (A11,A12,A22);
    # polar reduction E r^4 = 15 for standard 3D normal; midpoint grid on S^2 (dz dphi area).
    Lc = mp.cholesky(CA); Lf = [[float(Lc[i, j]) for j in range(3)] for i in range(3)]
    tot = 0.0
    for i in range(Nz):
        z = -1 + (i+0.5)*2.0/Nz; rho = math.sqrt(1-z*z)
        for j in range(Nphi):
            p = (j+0.5)*2*math.pi/Nphi
            v = (rho*math.cos(p), rho*math.sin(p), z)
            a11 = Lf[0][0]*v[0]
            a12 = Lf[1][0]*v[0] + Lf[1][1]*v[1]
            a22 = Lf[2][0]*v[0] + Lf[2][1]*v[1] + Lf[2][2]*v[2]
            det = a11*a22 - a12*a12
            if a11 < 0 and det > 0:
                tot += det*det
    area = (2.0/Nz)*(2*math.pi/Nphi)
    return 15.0/(4*math.pi)*tot*area   # E r^4 / |S^2| * integral = 15/(4 pi) * integral

# ---------------------------------------------------------------- pin densities and ledger
def gauss_density(C, x):
    n = C.rows
    xv = mp.matrix(x)
    q = (xv.T*mp.inverse(C)*xv)[0, 0]
    return mp.e**(-q/2)/mp.sqrt((2*mp.pi)**n*mp.det(C))

def pin_check(D, u, b, k):
    u, y1, y2 = frame(u)
    kap = 6*k                                         # h=b-kappa r^3/6 (075 (1)) vs s=b-k r^3 (P line 21)
    Ustar = [F(), F(u), F(u, u), F(u, u, u), F(y1), F(u, y1), F(y2), F(u, y2)]          # P (3.4)
    Y = [F(), [(-mp.mpf(1)/12, [u, u, u])], F(u), F(u, u), F(y1), F(u, y1), F(y2), F(u, y2)]  # 047 (2)
    piP = gauss_density(covmat(D, Ustar), [b, 0, 0, 12*k, 0, 0, 0, 0])
    pY = gauss_density(covmat(D, Y), [b, -kap/6, 0, 0, 0, 0, 0, 0])
    return 12*piP, pY

def integrands(D, u):
    Po = P_objects(D, u); Co = chain_objects(D, u)
    Dp = D_spectral(Po['CA']); Dc = D_spectral(Co['Gam'])
    g76 = mp.gamma(mp.mpf(7)/6)
    IP = g76/(mp.cbrt(24)*mp.sqrt(mp.pi))*Po['pG']*Po['pV']*Po['tau2']**(mp.mpf(2)/3)*Dp     # P (15.2)
    B3 = 3*g76/(2**(mp.mpf(4)/3)*mp.pi**3.5)                                                  # 056 (25)
    IC = B3*Co['sig2']**(mp.mpf(2)/3)*Dc/mp.sqrt(mp.det(Co['Sg'])*mp.det(Co['Sh']))           # 056 (24)
    return IP, IC, Po, Co, Dp, Dc

if __name__ == '__main__':
    random.seed(20261018)
    print('== (1) FIELD: P 3D image sum vs chain product of 1D k_L, derivative tensor through order 6')
    for L in (mp.mpf('2.2'), mp.mpf(24)):
        DP, DC = P_tensor(L), chain_tensor(L)
        err = max(abs(DP[g]-DC[g]) for g in DP)
        print(f'  L={L}: max |dK_P - dK_chain| over all |gamma|<=6 = {mp.nstr(err, 3)};  K(0): P={mp.nstr(DP[(0,0,0)],20)} chain={mp.nstr(DC[(0,0,0)],20)}')
    DB = BF_tensor(); D24 = chain_tensor(mp.mpf(24))
    print('  L=24 vs K_infty, max derivative diff:', mp.nstr(max(abs(D24[g]-DB[g]) for g in DB), 5))

    dirs = [(1, 0, 0), (1, 1, 1), (0.3, -0.7, 0.2), (-0.9, 0.15, 0.4)]
    for label, D in (('L=2.2', chain_tensor(mp.mpf('2.2'))), ('L=24', D24), ('BF', DB)):
        print(f'== (4) OBJECTS at {label}')
        for u in dirs:
            Po = P_objects(D, u); Co = chain_objects(D, u); Cop = chain_objects(D, u, sign=+1)
            dG = mp.det(Co['Sg']) - Po['detG']; dV = mp.det(Co['Sh']) - Po['detV']
            dGam = max(abs(x) for x in (Co['Gam'] - Po['CA']))
            r = Po['tau2']/(144*Co['sig2'])
            print(f'  u={u}: detSg-detCovG={mp.nstr(dG,3)} detSh-detCovV={mp.nstr(dV,3)} '
                  f'max|Gamma_t-Cov(A|V=0)|={mp.nstr(dGam,3)} tau2/(144 sig2)={mp.nstr(r,25)} '
                  f'sig2(sign+)/sig2(sign-)={mp.nstr(Cop["sig2"]/Co["sig2"],20)} parity max={mp.nstr(Po["parity"],3)}')
        if label == 'BF':
            Po = P_objects(D, (1, 0, 0)); Co = chain_objects(D, (1, 0, 0))
            print('  BF: tau2 =', mp.nstr(Po['tau2'], 20), ' sig2 =', mp.nstr(Co['sig2'], 20), '(056 (35): 1/24)',
                  ' detSh =', mp.nstr(mp.det(Co['Sh']), 20), ' Cov(u_t,g_t) =', [mp.nstr(x, 10) for x in Co['c_ug']])
            print('  BF: Cov(A|V=0) =', [[mp.nstr(Po['CA'][i, j], 12) for j in range(3)] for i in range(3)])

    print('== (4b) D object: P literal definition (grid, cone indicator) vs LS-DER-060 spectral (chain)')
    for label, D, u in (('L=2.2', chain_tensor(mp.mpf('2.2')), (0.3, -0.7, 0.2)),
                        ('L=2.2', chain_tensor(mp.mpf('2.2')), (1, 1, 1)),
                        ('BF', DB, (1, 0, 0))):
        Po = P_objects(D, u); Co = chain_objects(D, u)
        dd = D_direct(Po['CA']); ds = D_spectral(Co['Gam'])
        print(f'  {label} u={u}: D_P(direct grid)={dd:.8f}  D_t(060 spectral)={mp.nstr(ds, 15)}  rel diff={(dd-float(ds))/float(ds):.2e}')
    print('  BF target D_2 = 29/6 - sqrt(6) =', mp.nstr(mp.mpf(29)/6 - mp.sqrt(6), 15))

    print('== (2/3) PIN LEDGER: 12*pi_0(P, f_xxx coordinate, target 12k) vs p_Y(chain, -f_ttt/12, target -kappa/6), kappa=6k')
    for label, D in (('L=2.2', chain_tensor(mp.mpf('2.2'))), ('BF', DB)):
        for (u, b, k) in (((0.3, -0.7, 0.2), mp.mpf('0.4'), mp.mpf('0.7')), ((1, 1, 1), mp.mpf('-1.1'), mp.mpf('0.25'))):
            a, c = pin_check(D, u, b, k)
            extra = ''
            if label == 'BF':
                kap = 6*k
                ref = 2*mp.sqrt(3)/(2*mp.pi)**4*mp.e**(-3*b**2/4 - kap**2/3)   # 047 (7), d=3
                extra = f'  047(7) closed form ratio={mp.nstr(c/ref, 20)}'
            print(f'  {label} u={u} b={b} k={k}: 12 pi_0 / p_Y = {mp.nstr(a/c, 25)}{extra}')

    print('== (3) DIRECTED SPHERE: integrand evenness and full-integrand equality at L=2.2')
    D = chain_tensor(mp.mpf('2.2'))
    for u in ((0.3, -0.7, 0.2), (1, 1, 1), (-0.9, 0.15, 0.4)):
        IP, IC, *_ = integrands(D, u)
        IPm, ICm, *_ = integrands(D, tuple(-x for x in u))
        print(f'  u={u}: P(15.2) integrand={mp.nstr(IP, 18)}  056(24) integrand={mp.nstr(IC, 18)}  ratio={mp.nstr(IP/IC, 20)}  I(-u)/I(u)={mp.nstr(IPm/IP, 20)}')
