# Isotropic d=3 field with directional moments a=E f_1^2, m4=E f_11^2, chi=E f_111^2:
# evaluate the P (15.2) integrand structure symbolically and compare to LS-DER-050 (16) at d=3.
import sympy as sp
a,m4,chi=sp.symbols('a m4 chi',positive=True)
# isotropic 4th-order tensor: E[H_ij H_kl]=(m4/3)(dd+dd+dd); E[f_111 f_1]=-m4; Var f_111=chi
SG=a*sp.eye(3)
def c4(i,j,k,l): return sp.Rational(1,3)*m4*((i==j)*(k==l)+(i==k)*(j==l)+(i==l)*(j==k))
V=[(0,0),(0,1),(0,2)]; A=[(1,1),(2,2),(1,2)]
SV=sp.Matrix(3,3,lambda i,j:c4(*V[i],*V[j]))
SAV=sp.Matrix(3,3,lambda i,j:c4(*A[i],*V[j]))
SAA=sp.Matrix(3,3,lambda i,j:c4(*A[i],*A[j]))
Gam=sp.simplify(SAA-SAV*SV.inv()*SAV.T)
Gam0=Gam.subs(m4,3)
print('Gamma/(m4/3) == Gamma at m4=3 :', sp.simplify(Gam-(m4/3)*Gam0)==sp.zeros(3,3), ' Gamma0=',Gam0.tolist())
tau2=chi-m4**2/a
D0=sp.Rational(29,6)-sp.sqrt(6)
D=(m4/3)**2*D0           # degree-2 homogeneity of D in Gamma (h has degree 4)
pG=(2*sp.pi)**sp.Rational(-3,2)/sp.sqrt(SG.det()); pV=(2*sp.pi)**sp.Rational(-3,2)/sp.sqrt(SV.det())
integrand=pG*pV*tau2**sp.Rational(2,3)*D
Omega=a*chi-m4**2
law=Omega**sp.Rational(2,3)*m4**sp.Rational(1,2)*a**sp.Rational(-13,6)
ratio=sp.simplify(integrand/law)
print('integrand / [Omega^(2/3) m4^(1/2) a^(-13/6)] =',ratio,' free symbols:',ratio.free_symbols)
