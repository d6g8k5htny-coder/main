import sympy as sp
R = sp.Rational; pi = sp.pi; g76 = sp.gamma(R(7,6))
r = sp.symbols('r', positive=True)
fa, fc, fxa, fxc = sp.symbols('fa fc fxa fxc')
# P (3.1) frame and chain LS-DER-046 (3) frame on the same raw axial pins
U = sp.Matrix([(fa+fc)/2, (fc-fa)/r, (fxc-fxa)/r, 6/r**2*(fxa+fxc-2*(fc-fa)/r)])
Y = sp.Matrix([(fc+fa)/2, (fc-fa-r*(fxc+fxa)/2)/r**3, (fxc+fxa)/2, (fxc-fxa)/r])
JU = U.jacobian([fa,fxa,fc,fxc]); JY = Y.jacobian([fa,fxa,fc,fxc])
M = sp.simplify(JY*JU.inv())   # linear map U -> Y
print('map U->Y:', M.tolist(), ' |det| =', sp.simplify(abs(M.det())))
# isotropic family: P (15.2) vs LS-DER-050 (19)-(20), deriving A|V=0 cov from LS-DER-050 (8)
a, m4, chi = sp.symbols('a m4 chi', positive=True)
beta = m4/3; Om = a*chi - m4**2; Dn = sp.symbols('D_n', positive=True)
# check conditional covariance of transverse Hessian given H_tt=0 for n=2 entries (Q11,Q12,Q22)
Cqq = beta*sp.Matrix([[3,0,1],[0,1,0],[1,0,3]]); Cqh = beta*sp.Matrix([1,0,1]); Vh = 3*beta
cond = sp.simplify(Cqq - Cqh*Cqh.T/Vh)
Mn = sp.Matrix([[2+R(2,3),0,R(2,3)],[0,1,0],[R(2,3),0,2+R(2,3)]])
print('Cov(Q|Htt=0) - beta*Cov(M_2) =', sp.simplify(cond - beta*Mn))
for d in range(2, 8):
    n = d-1
    S = 2*pi**R(d,2)/sp.gamma(R(d,2))
    pG = (2*pi*a)**R(-d,2)
    pV = (2*pi)**R(-d,2)/sp.sqrt(m4*beta**n)
    tau2 = chi - m4**2/a
    P152 = g76/(24**R(1,3)*sp.sqrt(pi))*S*pG*pV*tau2**R(2,3)*beta**n*Dn
    A = S*Dn*sp.gamma(R(1,6))/(2**(d+2)*3**R(3*d+5,6)*pi**(d+R(1,2)))
    C050 = A*Om**R(2,3)*m4**R(d-2,2)/a**R(3*d+4,6)
    print('d=%d P(15.2)/LS-DER-050(19)-(20) =' % d, sp.simplify(sp.gammasimp(sp.powsimp(P152/C050, force=True))))
# BF reference d=3 numeric & LS-DER-047/050/056 closed form & LS-CLS-077 closed form
c050 = 2**R(1,6)*3**R(1,3)*(29*sp.sqrt(6)-36)*sp.gamma(R(1,6))/(432*pi**R(5,2))
c077 = 2**R(-23,6)*3**R(-8,3)*(29*sp.sqrt(6)-36)*sp.gamma(R(1,6))*pi**R(-5,2)
print('LS-DER-075(21)/LS-CLS-077 closed form:', sp.nsimplify(sp.simplify(c050/c077)), sp.N(c050, 25))
# PROOF interval check vs c_inf and vs chain-corrected value
cinf = sp.N(c050, 60)
P3 = sp.Rational(-620813376, 35)
c24 = cinf*(1 + P3*sp.exp(-288))
lo, hi = sp.Rational('0.04177593184059834334'), sp.Rational('0.04177593184059834335')
print('c_inf in PROOF interval:', bool(lo < cinf < hi), ' chain c24 in interval:', bool(lo < sp.N(c24,60) < hi))
print('relative P3*e^-288 =', sp.N(P3*sp.exp(-288), 5))
# 24 in 24^(1/3) is not L: P kernel symbolic in tau, no L
tau, k = sp.symbols('tau k', positive=True)
phi = sp.exp(-(12*k)**2/(2*tau**2))/sp.sqrt(2*pi*tau**2)
ker = sp.simplify(144*sp.integrate(k**R(4,3)*phi, (k, 0, sp.oo)))
print('P kernel / [Gamma(7/6) tau^(4/3)/(24^(1/3) sqrt(pi))] =', sp.simplify(sp.gammasimp(ker/(g76*tau**R(4,3)/(24**R(1,3)*sp.sqrt(pi))))))
