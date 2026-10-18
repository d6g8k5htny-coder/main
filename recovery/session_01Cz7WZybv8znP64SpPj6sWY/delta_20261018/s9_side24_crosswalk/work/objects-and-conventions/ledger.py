"""Symbolic measure ledger: P (3.2),(10.1)-(10.2),(11.1)-(11.2),(13.6) vs chain
LS-DER-046 (4)-(6),(10), LS-DER-075 (1)-(13), LS-DER-076 (10),(15), with kappa = 6k."""
import sympy as sp
r, k, b, ell, kap = sp.symbols('r k b ell kappa', positive=True)
piU, E = sp.symbols('pi_U E_b', positive=True)   # P contact density at v_0 (f_xxx coordinate), cone moment given f=b,V=0
d = 3
# ---- gap variable: P s=b-k r^3 (line 21); chain h=b-kappa r^3/6 (075 (1)) ; same saddle value iff kappa=6k
print('saddle-value identity kappa=6k:', sp.simplify((b - k*r**3) - (b - kap*r**3/6)).subs(kap, 6*k))
# ---- f_ttt at contact: P target U3 -> 12k (3.3)/line 103; chain target u=-f_ttt/12=-kappa/6 (056 (7)) => f_ttt = 2kappa
print('f_ttt: P 12k vs chain 2kappa at kappa=6k ->', sp.simplify(12*k - 2*kap.subs(kap, 6*k)))
# ---- endpoint axial Hessian: P alpha_M -> -6k (5.2); chain A_M -> -kappa (046 (8))
print('soft factor: P (6k)^2 vs chain kappa^2 ->', sp.simplify((6*k)**2 - kap**2).subs(kap, 6*k))
# ---- pin Jacobians: P 12 r^-(d+3) pi_U (10.1); chain r^-(d+3) p_Y (046 (6)), p_Y = 12 pi_U (u = -U3/12, numerically verified)
pY = 12*piU
print('pin Jacobian x density, P/chain:', sp.simplify((12*r**-(d+3)*piU)/(r**-(d+3)*pY)))
# ---- full contact measure on (r, b, gap, direction) before lifetime pushforward
# P: r^(d-1) dr dsigma * r^3 db dk * 12 r^-(d+3) pi_U * r^2 (6k)^2 E   (10.2: r A_r dr db dk dsigma)
P_meas = r**(d-1) * r**3 * 12*r**-(d+3)*piU * r**2*(6*k)**2*E          # per dr db dk dsigma
# chain: r^(d-1) dr dsigma * (r^3/6) db dkappa * r^-(d+3) p_Y * r^2 kappa^2 E  (046/075/076 (15))
C_meas = r**(d-1) * (r**3/6) * r**-(d+3)*pY * r**2*kap**2*E             # per dr db dkappa dsigma
# compare as measures: dkappa = 6 dk
print('contact measure P(dk) / chain(dkappa=6dk):', sp.simplify(P_meas/(C_meas.subs(kap, 6*k)*6)))
print('  radial power P:', sp.simplify(P_meas/(piU*E*k**2)), ' chain:', sp.simplify(C_meas/(pY*E*kap**2)))
# ---- lifetime pushforward: P ell = k r^3 (r=(ell/k)^(1/3)), chain ell = kappa r^3/6 (075 (6)-(9))
rP = (ell/k)**sp.Rational(1, 3); rC = (6*ell/kap)**sp.Rational(1, 3)
jP = sp.simplify(rP*sp.diff(rP, ell)); jC = sp.simplify(rC*sp.diff(rC, ell))
print('r dr/dell P:', jP, '  P (11.1) stated:', sp.Rational(1, 3)*k**sp.Rational(-2, 3)*ell**sp.Rational(-1, 3))
print('r dr/dell chain:', jC, '  075 (9) stated:', sp.Rational(1, 3)*6**sp.Rational(2, 3)*kap**sp.Rational(-2, 3)*ell**sp.Rational(-1, 3))
# ---- coefficient integrands per db dsigma after pushforward
cP = sp.simplify(jP*ell**sp.Rational(1, 3) * 12*piU*(6*k)**2*E)            # P (13.6): 4 k^-2/3 pi_0 z_0 dk
cC = sp.simplify(jC*ell**sp.Rational(1, 3) * sp.Rational(1, 6)*pY*kap**2*E)   # chain: (6^(2/3)/3) kappa^-2/3 J_0 dkappa
print('P coefficient integrand:', cP, '  P (13.6) 4 k^(-2/3) pi_0 z_0 =', sp.simplify(4*k**sp.Rational(-2, 3)*piU*36*k**2*E))
print('ratio P dk / chain dkappa(=6dk):', sp.simplify(cP/(cC.subs(kap, 6*k)*6)))
# ---- combined chain scalar 6^(2/3)/18 (075 (13)) vs P's 4/(12)... show both equal 1/3 per dk with pi_U units
print('chain (1/6)(6^(2/3)/3) =', sp.nsimplify(sp.Rational(1, 6)*6**sp.Rational(2, 3)/3), '; 075 (13) 6^(2/3)/18 equal:',
      sp.simplify(sp.Rational(1, 6)*6**sp.Rational(2, 3)/3 - 6**sp.Rational(2, 3)/18) == 0)
# ---- odd-variance convention: tau^2 = Var(f_ttt|G=0); sigma^2 = Var(-f_ttt/12|g=0)
tau2 = sp.symbols('tau2', positive=True)
print('tau^(4/3) / (144 sigma^2)^(2/3) =', sp.simplify(tau2**sp.Rational(2, 3)/((144*(tau2/144))**sp.Rational(2, 3))))
# ---- alternative-convention hazards (what a mismatch WOULD cost): hemisphere, probability sphere, unordered pairs
print('hazard multipliers (not present in either source): hemisphere 1/2, probability-normalized S^2 1/(4pi)=',
      sp.nsimplify(1/(4*sp.pi)), ', unordered-pair 1/2')
