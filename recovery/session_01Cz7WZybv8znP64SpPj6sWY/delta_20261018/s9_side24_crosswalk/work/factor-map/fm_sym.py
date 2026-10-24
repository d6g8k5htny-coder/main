"""factor-map lens: independent sympy algebra mapping every chain normalization factor onto P / PROOF.

Chain sources: LS-DER-046 (2)-(19), LS-DER-075 (1)-(21), LS-DER-056 (7),(21)-(25),(28)-(30), LS-DER-050 (4),(16)-(20).
Landing sources: P (1) setup line 21, (3.1)-(3.3), (5.2),(5.4), (10.1)-(10.2), (11.1)-(11.3), (13.6), (15.1)-(15.2);
PROOF.md (1).
"""
import sympy as sp

R = sp.Rational
r, b, k, kap, ell, tau, sig = sp.symbols('r b k kappa ell tau sigma', positive=True)
pi = sp.pi
g76 = sp.gamma(R(7, 6))
out = []
def rep(label, val):
    out.append(f"{label}: {val}")

# ---------------------------------------------------------------- 1. kappa vs k from the definitions
# P line 21: f(S)=b-k r^3 ; chain LS-DER-046 (2) / LS-DER-075 (1): f(S)=b-kappa r^3/6
rep('1a saddle value: solve b-k r^3 = b-kappa r^3/6 for kappa', sp.solve(sp.Eq(b - k*r**3, b - kap*r**3/6), kap))
# cubic axial model through both pin sets (M=-r/2, S=+r/2, zero axial slope at both)
x = sp.symbols('x')
c0, c1, c2, c3 = sp.symbols('c0:4')
F = c0 + c1*x + c2*x**2 + c3*x**3
solP = sp.solve([F.subs(x, -r/2) - b, F.subs(x, r/2) - (b - k*r**3),
                 sp.diff(F, x).subs(x, -r/2), sp.diff(F, x).subs(x, r/2)], [c0, c1, c2, c3], dict=True)[0]
FP = F.subs(solP)
rep("1b P cubic: f''' , f''(M)/r , f''(S)/r", [sp.simplify(sp.diff(FP, x, 3)),
     sp.simplify(sp.diff(FP, x, 2).subs(x, -r/2)/r), sp.simplify(sp.diff(FP, x, 2).subs(x, r/2)/r)])
# chain: f(S)=b-kappa r^3/6
solC = sp.solve([F.subs(x, -r/2) - b, F.subs(x, r/2) - (b - kap*r**3/6),
                 sp.diff(F, x).subs(x, -r/2), sp.diff(F, x).subs(x, r/2)], [c0, c1, c2, c3], dict=True)[0]
FC = F.subs(solC)
rep("1c chain cubic: f''' , f''(M)/r (LS-DER-046 (8) says -kappa) , f''(S)/r (+kappa)",
    [sp.simplify(sp.diff(FC, x, 3)), sp.simplify(sp.diff(FC, x, 2).subs(x, -r/2)/r), sp.simplify(sp.diff(FC, x, 2).subs(x, r/2)/r)])

# ---------------------------------------------------------------- 2. pin frames: P (3.1) vs LS-DER-046 (3)
fa, fc, fxa, fxc = sp.symbols('fa fc fxa fxc')  # f(M), f(S), d_t f(M), d_t f(S)
U0 = (fa + fc)/2; U1 = (fc - fa)/r; U2 = (fxc - fxa)/r; U3 = 6/r**2*(fxa + fxc - 2*(fc - fa)/r)
Vp = (fc + fa)/2; Vm = (fc - fa - r*(fxc + fxa)/2)/r**3; Gp = (fxc + fxa)/2; Gm = (fxc - fxa)/r
rep('2a LS-DER-046 V^-_corr + U3/12 (P) == 0 ?', sp.simplify(Vm + U3/12))
vars4 = [fa, fxa, fc, fxc]
JP = sp.Matrix([[sp.diff(e, v) for v in vars4] for e in (U0, U1, U2, U3)])
JC = sp.Matrix([[sp.diff(e, v) for v in vars4] for e in (Vp, Vm, Gp, Gm)])
rep('2b |det| axial 4x4: P (3.1) , chain LS-DER-046 (3)', [sp.simplify(abs(JP.det())), sp.simplify(abs(JC.det()))])
for d in (2, 3):
    m = d - 1
    rep(f'2c d={d}: P |det T_r| (3.2) = 12 r^-(d+3)? , chain (4) = r^-(d+3)?',
        [sp.simplify(abs(JP.det())*(1/r)**m / (12*r**(-(d + 3)))), sp.simplify(abs(JC.det())*(1/r)**m / r**(-(d + 3)))])
# targets on the P pins (f(M)=b, f(S)=b-k r^3, slopes 0) and on chain pins with kappa=6k
tgt = {fa: b, fc: b - k*r**3, fxa: 0, fxc: 0}
rep('2d P target (3.3) components U0,U1,U2,U3', [sp.simplify(e.subs(tgt)) for e in (U0, U1, U2, U3)])
rep('2e chain target LS-DER-046 (5) V+,V-corr at f(S)=b-kappa r^3/6',
    [sp.simplify(e.subs({fa: b, fc: b - kap*r**3/6, fxa: 0, fxc: 0})) for e in (Vp, Vm)])
# Taylor limit of V^-_corr: f(S)-f(M) = r f1 + r^3 f3/24 + ..., slopes avg = f1 + r^2 f3/8 + ...
f1, f3, f5 = sp.symbols('f1 f3 f5')
taylor = {fc - fa: None}
diffval = r*f1 + r**3*f3/24 + r**5*f5/1920
slope_sum = 2*f1 + r**2*f3/4 + r**4*f5/192
Vm_lim = sp.limit((diffval - r*slope_sum/2)/r**3, r, 0)
rep('2f lim V^-_corr (even-part Taylor) = -f_ttt/12 ?', sp.simplify(Vm_lim + f3/12) == 0)
# density relation: u=-f_ttt/12 => density of u at -kappa/6 equals 12 * density of f_ttt at 2 kappa
tt = sp.symbols('t_')
phi = lambda z, s: sp.exp(-z**2/(2*s**2))/sp.sqrt(2*pi*s**2)
rep('2g phi_{tau/12}(-kappa/6) / [12 phi_tau(2 kappa)] at kappa=6k vs P phi_tau(12k)',
    sp.simplify(phi(-kap/6, tau/12).subs(kap, 6*k) / (12*phi(12*k, tau))))

# ---------------------------------------------------------------- 3. full intensity before r->0 (same measure?)
piP, zz = sp.symbols('pi_r z_r', positive=True)   # P's pi_r(R;v_r) at k ; Z_r/r^2 (identical weight W_r in both)
for d in (2, 3):
    # P: polar r^(d-1) * heights r^3 (db dk) * pins 12 r^-(d+3) pi_r * Z_r = r A_r  (10.1)-(10.2)
    P_int = r**(d - 1) * r**3 * 12*r**(-(d + 3))*piP * r**2*zz          # per dr db dk dsigma
    # chain: polar r^(d-1) * gap Jacobian r^3/6 (db dkappa) * pins r^-(d+3) p_Y * Z_r ; p_Y = 12 pi_r (frame relation 2a/2b)
    C_int = r**(d - 1) * r**3/6 * r**(-(d + 3))*(12*piP) * r**2*zz      # per dr db dkappa dsigma
    # pushforward of chain measure to dk: dkappa = 6 dk
    rep(f'3 d={d}: P density (per dk) / [chain density (per dkappa) * dkappa/dk]', sp.simplify(P_int / (C_int*6)))
    rep(f'3 d={d}: P r A_r form check (10.2) r*12*pi*z', sp.simplify(P_int/(r*12*piP*zz)))

# ---------------------------------------------------------------- 4. lifetime pushforward
rr = (ell/k)**R(1, 3)
P_life = sp.simplify(rr*sp.diff(rr, ell))  # P (11.1)
rrc = (6*ell/kap)**R(1, 3)
C_life = sp.simplify(rrc*sp.diff(rrc, ell))  # LS-DER-075 (7)-(9)
rep('4a P (11.1) / [(1/3) k^(-2/3) ell^(-1/3)]', sp.simplify(P_life/(k**R(-2, 3)*ell**R(-1, 3)/3)))
rep('4b chain (9) / [(6^(2/3)/3) kappa^(-2/3) ell^(-1/3)]', sp.simplify(C_life/(6**R(2, 3)/3*kap**R(-2, 3)*ell**R(-1, 3))))
rep('4c chain (9) at kappa=6k / P (11.1)', sp.simplify(C_life.subs(kap, 6*k)/P_life))
rep('4d ell definitions: chain kappa r^3/6 at kappa=6k minus P k r^3', sp.simplify(kap*r**3/6 - k*r**3).subs(kap, 6*k))

# ---------------------------------------------------------------- 5. limiting combined scalar
E = sp.symbols('E', positive=True)  # common cone moment E[(det A)^2 1{A<0} | f=b, V=0]
pi0 = sp.Function('pi0')
# P (11.2)-(11.3)/(13.6): A_0/(3k^(2/3)) dk with A_0 = 12 pi_0 z_0, z_0=(6k)^2 E
P_lim = 12*pi0(b, k)*(6*k)**2*E/(3*k**R(2, 3))
rep('5a P integrand / [4 k^(-2/3) pi0 z0]  (the 4 in (11.3),(13.6))', sp.simplify(P_lim/(4*k**R(-2, 3)*pi0(b, k)*(6*k)**2*E)))
# chain LS-DER-046 (14),(19): (6^(2/3)/3) kappa^(-2/3) J_0, J_0=(1/6) p_0 kappa^2 E with p_0(b,kappa)=12 pi0(b,kappa/6)
C_lim = 6**R(2, 3)/3*kap**R(-2, 3)*R(1, 6)*12*pi0(b, kap/6)*kap**2*E
rep('5b chain integrand (per dkappa) at kappa=6k, times dkappa/dk=6, / P integrand (per dk)',
    sp.simplify(C_lim.subs(kap, 6*k)*6/P_lim))
rep('5c LS-DER-075 (13) (1/6)(6^(2/3)/3) == 4 * 6^(2/3)/(12*6) ?', sp.simplify(R(1, 6)*6**R(2, 3)/3 - 4*6**R(2, 3)/(12*6)) == 0)

# ---------------------------------------------------------------- 6. kappa integral (LS-DER-056 (23)) vs P 144-kernel
s2 = sig**2
dens_u = sp.exp(-kap**2/(72*s2))/sp.sqrt(2*pi*s2)                 # LS-DER-056 (21) without p_g(0)
I23 = sp.integrate(dens_u*kap**R(4, 3), (kap, 0, sp.oo))
rep('6a LS-DER-056 (23) evaluated / stated RHS', sp.simplify(sp.gammasimp(I23/(g76*72**R(7, 6)*s2**R(2, 3)/(2*sp.sqrt(2*pi))))))
Pker = 144*sp.integrate(k**R(4, 3)*phi(12*k, tau), (k, 0, sp.oo))
rep('6b P kernel 144 int k^(4/3) phi_tau(12k) dk / [Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt pi)]',
    sp.simplify(sp.gammasimp(Pker/(g76*tau**R(4, 3)/(24**R(1, 3)*sp.sqrt(pi))))))
rep('6c (1/6)(6^(2/3)/3) * I23 at sigma=tau/12  /  P kernel', sp.simplify(sp.gammasimp((R(1, 6)*6**R(2, 3)/3*I23).subs(sig, tau/12)/Pker)))

# ---------------------------------------------------------------- 7. B_3 decomposition
B3 = 3*g76/(2**R(4, 3)*pi**R(7, 2))
rep('7a B_3 / [Gamma(7/6)/(24^(1/3) sqrt pi) * (2pi)^-3 * 144^(2/3)]',
    sp.nsimplify(sp.simplify(B3/(g76/(24**R(1, 3)*sp.sqrt(pi))*(2*pi)**-3*144**R(2, 3)))))
rep('7b B_3 from (1/6)(6^(2/3)/3) Gamma(7/6) 72^(7/6)/(2 sqrt(2pi)) (2pi)^-3 / B_3',
    sp.simplify(R(1, 6)*6**R(2, 3)/3*g76*72**R(7, 6)/(2*sp.sqrt(2*pi))*(2*pi)**-3/B3))
rep('7c (30) prefactor = B_3*15/(4pi) ?', sp.simplify(B3*15/(4*pi) - 45*g76/(2**R(10, 3)*pi**R(9, 2))) == 0)
rr_ = sp.symbols('rho', positive=True)
rep('7d LS-DER-056 (29) radial factor', sp.simplify((2*pi)**R(-3, 2)*sp.integrate(rr_**6*sp.exp(-rr_**2/2), (rr_, 0, sp.oo))))
# integrand-level identity for arbitrary covariance objects (pointwise in direction)
dg, dh, Dt, ta2 = sp.symbols('detSigma_g detSigma_h D tau2', positive=True)
L056 = B3*(ta2/144)**R(2, 3)*Dt/sp.sqrt(dg*dh)
P152 = g76/(24**R(1, 3)*sp.sqrt(pi))*(2*pi)**R(-3, 2)/sp.sqrt(dg)*(2*pi)**R(-3, 2)/sp.sqrt(dh)*ta2**R(2, 3)*Dt
rep('7e (24) integrand / (15.2) integrand, arbitrary detSigma_g, detSigma_h, tau^2=144 sigma^2, D', sp.simplify(L056/P152))
# amplitude scaling: f -> A f
A = sp.symbols('A', positive=True)
rep('7f (24) integrand scaling under f->A f (P says A^(-2/3))',
    sp.simplify(L056.subs({dg: A**6*dg, dh: A**6*dh, ta2: A**2*ta2, Dt: A**4*Dt})/L056))

# ---------------------------------------------------------------- 8. isotropic family, general covariance params (a, m4, chi), d=2..7
a, m4, chi = sp.symbols('a m4 chi', positive=True)
Om = a*chi - m4**2; beta = m4/3
Dn = sp.symbols('D_n', positive=True)
for d in range(2, 8):
    n = d - 1
    S_area = 2*pi**R(d, 2)/sp.gamma(R(d, 2))
    pG = (2*pi*a)**R(-d, 2)
    pV = (2*pi)**R(-d, 2)*(m4*beta**n)**R(-1, 2)          # V=(f_tt, f_tn...) var (m4, beta,...)
    tau2_iso = chi - m4**2/a                                # Var(f_ttt | G=0) = Omega/a
    Du = beta**n*Dn                                         # A|V=0 = sqrt(beta) M_n  (LS-DER-047 (11),(14))
    P_iso = g76/(24**R(1, 3)*sp.sqrt(pi))*S_area*pG*pV*tau2_iso**R(2, 3)*Du
    Ajet = S_area*Dn*sp.gamma(R(1, 6))/(2**(d + 2)*3**(R(3*d + 5, 6))*pi**(d + R(1, 2)))
    C_iso = Ajet*Om**R(2, 3)*m4**R(d - 2, 2)/a**R(3*d + 4, 6)                     # LS-DER-075 (18)-(19)
    Bd = Dn*(2*pi*m4)**R(-1, 2)*(2*pi)**(-n)*a**R(-n, 2)*beta**R(n, 2)             # LS-DER-075 (15)
    Kd = 3*g76/(pi*sp.sqrt(Om))*(Om/(2*a))**R(7, 6)                              # LS-DER-075 (16)
    C_rec = R(1, 6)*6**R(2, 3)/3*S_area*Bd*Kd
    rep(f'8 d={d}: P(15.2)_iso / LS-DER-075(18)-(19) ; LS-DER-075 recombination(1/6)(6^(2/3)/3)|S|B_d K_d / (18)-(19)',
        [sp.simplify(sp.gammasimp(P_iso/C_iso)), sp.simplify(sp.gammasimp(C_rec/C_iso))])
# LS-DER-050 (17) odd-block density check against u=-f_ttt/12, Cov(f_t,f_ttt)=-m4
Cod = sp.Matrix([[a, m4/12], [m4/12, chi/144]])
yv = sp.Matrix([0, -kap/6])
dens_odd = sp.exp(-(yv.T*Cod.inv()*yv)[0]/2)/(2*pi*sp.sqrt(Cod.det()))
rep('8b LS-DER-050 (17) density / evaluated', sp.simplify(12/(2*pi*sp.sqrt(Om))*sp.exp(-2*a*kap**2/Om)/dens_odd))
# Kd = (17)-integral
rep('8c LS-DER-075 (16) K_d / int (17) kappa^(4/3)',
    sp.simplify(sp.gammasimp(3*g76/(pi*sp.sqrt(Om))*(Om/(2*a))**R(7, 6) /
                             sp.integrate(12/(2*pi*sp.sqrt(Om))*sp.exp(-2*a*kap**2/Om)*kap**R(4, 3), (kap, 0, sp.oo), conds='none'))))

# ---------------------------------------------------------------- 9. PROOF.md (1) from P (15.2) at reference law (independent redo)
for d, Dval in ((2, R(4, 3)), (3, R(29, 6) - sp.sqrt(6))):
    S_area = 2*pi**R(d, 2)/sp.gamma(R(d, 2))
    P_ref = g76/(24**R(1, 3)*sp.sqrt(pi))*S_area*(2*pi)**(-d)/sp.sqrt(3)*6**R(2, 3)*Dval
    proof1 = g76*R(3, 2)**R(1, 3)*Dval/(2*sp.sqrt(3)*pi**(d - 1)*sp.sqrt(pi))
    rep(f'9 d={d}: P(15.2)@ref / PROOF (1)', sp.simplify(P_ref/proof1))
cinf077 = 2**R(-23, 6)*3**R(-8, 3)*(29*sp.sqrt(6) - 36)*sp.gamma(R(1, 6))*pi**R(-5, 2)
c3ref = g76*R(3, 2)**R(1, 3)*(R(29, 6) - sp.sqrt(6))/(2*sp.sqrt(3)*pi**2*sp.sqrt(pi))
rep('9b PROOF(1) d=3 / LS-CLS-077 closed form', sp.simplify(sp.gammasimp(c3ref/cinf077)))
rep('9c numeric c_3,ref', sp.N(c3ref, 30))

print("\n".join(out))
