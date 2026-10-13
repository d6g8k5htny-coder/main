# Fast checks using the eta-form of the density (no nested root finding):
#   nu(t) = c0 t^(1/4) int_0^{eta_c} kappa(eta)^(-5/4) 1{t < u0^2 kappa(eta)} d eta    (from (25) by u=sqrt(l/kappa))
#   E N_t = 2 c0 int_0^{eta_c} (2/5) min(u0, sqrt(t/kappa))^(5/2) d eta
from mpmath import mp, mpf, sqrt, quad, findroot, pi, asin, acos, cos, diff, sin
mp.dps = 25
eta_c = 2/(3*sqrt(3)); a_c = mpf(3)**(-mpf(1)/4); K_c = 4/mpf(3)**(mpf(5)/4); Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)
def Phi(z,eta): return -z**4/4 + z**2/2 + eta*z
def roots(eta):
    th = acos(eta/eta_c); return sorted([(2/sqrt(3))*cos((th-2*pi*k)/3) for k in range(3)])
def kappa(eta):
    zm,z0,zp = roots(eta); return Phi(zm,eta)-Phi(z0,eta)
c0=mpf(1); u0=mpf(1)
C_edge=(8*c0/7)*K_c**(-mpf(2)/3)*u0**(mpf(7)/6)
def eta_cut(t):  # kappa(eta)=t/u0^2
    return findroot(lambda e: kappa(e)-t/u0**2,(mpf(0),eta_c),solver='bisect',tol=mpf(10)**(-22))
def nu(t):
    ec=eta_cut(t); 
    pts=[mpf(0)]+[ec*(1-mpf(10)**(-k)) for k in (1,2,3,4)]+[ec]
    return c0*t**(mpf(1)/4)*quad(lambda e: kappa(e)**(-mpf(5)/4), pts)
def EN(t):
    ec=eta_cut(t)
    return 2*c0*(mpf(2)/5)*( quad(lambda e: (t/kappa(e))**(mpf(5)/4), [mpf(0),ec*(1-mpf('1e-2')),ec]) + u0**(mpf(5)/2)*(eta_c-ec) )
print("C_edge (c0=u0=1) =",C_edge, "  (3/2)C_edge =",1.5*C_edge)
for t in [mpf('1e-4'),mpf('1e-6'),mpf('1e-8'),mpf('1e-10')]:
    print("t",t," t^(1/3) nu(t) =",t**(mpf(1)/3)*nu(t)," t^(-2/3) E N_t =",t**(-mpf(2)/3)*EN(t))
# (26) total mass via E N_t at t=u0^2/4
print("(26) E N_{u0^2/4} =",EN(u0**2/4)," expected 4 c0 eta_c/5 =",4*c0*eta_c/5)
# (9): restricted tip density exact for t < u0^2 kappa(eta_c-eps)
eps=mpf('0.1'); C_tip=c0*quad(lambda e: kappa(e)**(-mpf(5)/4),[0,eta_c-eps])
print("(9) C_tip,eps(0.1) =",C_tip,"; kappa_min u0^2 =",kappa(eta_c-eps)*u0**2)
# small-u contribution (28): 2 c0 int_0^r u^(1/6)[(t/u^2)^(1/3) q] du <= (12 c0 A_q/7) r^(7/6); limiting (12 c0 Q_c/7) r^(7/6)
print("(28) limiting coeff for r=0.1:",12*c0*Q_c/7*mpf('0.1')**(mpf(7)/6))
# (31) physical cubic mark, L=1
L=mpf(1); om=2*pi/L; w1=mpf(1)/3
def y1(s): return (2/om)*asin(s/sqrt(2*w1))
for u in [mpf('0.01'),mpf('0.2')]:
    g=2/(om*sqrt(2*w1))*(1-u/(6*w1))**(-mpf(1)/2); k_u=sqrt(u)/(2*sqrt(3)*g**3)
    print("u",u,"g(u)",g," y1'(s_c)",diff(y1,-sqrt(u/3))," k(u)",k_u)
    for d in [mpf('1e-4'),mpf('1e-8')]:
        eta=eta_c-d; zm,z0,zp=roots(eta); sm,s0=sqrt(u)*zm,sqrt(u)*z0
        r=y1(s0)-y1(sm); ell=u**2*kappa(eta)
        print("   d",d," ell/r^3 =",ell/r**3," r/(2g sqrt(u) a_c sqrt(d)) =",r/(2*g*sqrt(u)*a_c*sqrt(d)))
    v=eta_c*u**mpf(1.5)
    def Pphys(y): s=sqrt(2*w1)*sin(om*y/2); return -s**4/4+u*s**2/2+v*s
    yc=y1(-sqrt(u/3))
    print("   phys P',P'',P''' at y_c:",diff(Pphys,yc,1),diff(Pphys,yc,2),diff(Pphys,yc,3)," 12k(u)=",12*k_u)
    print("   older max height - b = u^2 Phi(z_+) =",u**2*Phi(roots(eta_c)[2],eta_c)," 2u^2/3=",2*u**2/3, " younger at edge:",u**2*Phi(roots(eta_c)[0],eta_c)," -u^2/12=",-u**2/12)
# crossover sanity: C_tip,eps ~ eps^(-7/8) scaling
for e in [mpf('0.1'),mpf('0.01'),mpf('0.001')]:
    Ct=c0*quad(lambda x: kappa(x)**(-mpf(5)/4),[0,eta_c-e]); print("eps",e,"C_tip*eps^(7/8) =",Ct*e**(mpf(7)/8), " predicted limit c0 K_c^(-5/4)(8/7) =",c0*K_c**(-mpf(5)/4)*mpf(8)/7)
