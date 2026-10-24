"""LS-DER-056 (24)-(25) vs parent P (15.2) at d=3, pointwise in the direction, for arbitrary covariance objects."""
import sympy as sp
r = sp.Rational
tau2, dg, dh, Dt = sp.symbols('tau2 detSigma_g detSigma_h D', positive=True)
g76 = sp.gamma(r(7, 6)); pi = sp.pi
# P (15.2) integrand, d=3: Gamma(7/6)/(24^(1/3) sqrt(pi)) * p_G(0) p_V(0) tau^(4/3) D
pG = (2*pi)**r(-3, 2) / sp.sqrt(dg)
pV = (2*pi)**r(-3, 2) / sp.sqrt(dh)
P152 = g76 / (24**r(1, 3)*sp.sqrt(pi)) * pG * pV * tau2**r(2, 3) * Dt
# LS-DER-056: u_t = -f_ttt/12 so sigma_t^2 = Var(u_t|g=0) = tau^2/144
sig2 = tau2 / 144
B3 = 3*g76 / (2**r(4, 3) * pi**r(7, 2))
L056 = B3 * sig2**r(2, 3) * Dt / sp.sqrt(dg*dh)
print('LS-DER-056 integrand / P(15.2) integrand =', sp.simplify(L056 / P152))
# LS-DER-056 (23): kappa integral of the odd pin density
kap, s2 = sp.symbols('kappa s2', positive=True)
dens = (2*pi*s2)**r(-1, 2) * sp.exp(-kap**2/(72*s2))
I23 = sp.integrate(dens * kap**r(4, 3), (kap, 0, sp.oo))
print('056 (23) / stated (pG factored out) =', sp.simplify(sp.gammasimp(I23 / (g76*72**r(7, 6)*s2**r(2, 3)/(2*sp.sqrt(2*pi))))))
# assembled: (1/6)(6^(2/3)/3) * p_h(0) D * p_g(0) * I23  ->  B3 (s2)^(2/3) D / sqrt(dg dh)
asm = r(1, 6) * 6**r(2, 3) / 3 * (2*pi)**r(-3, 2)/sp.sqrt(dh) * Dt * (2*pi)**r(-3, 2)/sp.sqrt(dg) * I23
print('assembled (1/6)(6^(2/3)/3) p_h p_g I23 / (B3 s2^(2/3) D/sqrt(dg dh)) =', sp.simplify(sp.gammasimp(asm / (B3*s2**r(2, 3)*Dt/sp.sqrt(dg*dh)))))
# P's own kernel: 144 int k^(4/3) phi_tau(12k) dk with t=12k, vs chain kappa: target u=-kappa/6 means f_ttt = 2 kappa, i.e. t = 2 kappa = 12 k  => kappa = 6k
k = sp.symbols('k', positive=True)
lhs = 144*sp.integrate(k**r(4, 3)*sp.exp(-(12*k)**2/(2*tau2))/sp.sqrt(2*pi*tau2), (k, 0, sp.oo))
print('P kernel / [Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt pi)] =', sp.simplify(sp.gammasimp(lhs/(g76*tau2**r(2, 3)/(24**r(1, 3)*sp.sqrt(pi))))))
