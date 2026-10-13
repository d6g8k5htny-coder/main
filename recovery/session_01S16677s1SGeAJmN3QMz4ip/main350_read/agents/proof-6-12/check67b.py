from mpmath import mp, mpf, sqrt, quad, findroot, pi, asin, acos, cos, diff
mp.dps = 40
eta_c = 2/(3*sqrt(3)); a_c = mpf(3)**(-mpf(1)/4); K_c = 4/mpf(3)**(mpf(5)/4); Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)
def Phi(z,eta): return -z**4/4 + z**2/2 + eta*z
def roots(eta):  # z^3 - z - eta = 0, trig form, eta in [0,eta_c]
    th = acos(eta/eta_c)
    r = sorted([(2/sqrt(3))*cos((th-2*pi*k)/3) for k in range(3)])
    return r
def kappa(eta):
    zm,z0,zp = roots(eta); return Phi(zm,eta)-Phi(z0,eta)
def eta_of_h(h):
    return findroot(lambda e: kappa(e)-h, (mpf(0), eta_c), solver='bisect', tol=mpf(10)**(-36))
def q_of_h(h):
    eta=eta_of_h(h); zm,z0,zp=roots(eta); return 1/(z0-zm)
print("== roots at eta_c:",roots(eta_c), " z_c=-1/sqrt3=",-1/sqrt(3)," z_+=2/sqrt3=",2/sqrt(3))
print("Phi_{eta_c}(z_+)=",Phi(2/sqrt(3),eta_c)," expected 2/3")
print("== (23) h^(1/3) q(h) -> Q_c =",Q_c)
for h in [mpf('1e-3'),mpf('1e-6'),mpf('1e-9'),mpf('1e-12')]:
    print("  h",h,":",h**(mpf(1)/3)*q_of_h(h))
print("== q(h)->1 as h->1/4:",q_of_h(mpf('0.25')-mpf('1e-9')))
print("== (22) kappa/(K_c d^1.5), (z0-zm)/(2 a_c sqrt d):")
for d in [mpf('1e-2'),mpf('1e-4'),mpf('1e-6')]:
    zm,z0,zp=roots(eta_c-d); print("  d",d,kappa(eta_c-d)/(K_c*d**mpf(1.5)),(z0-zm)/(2*a_c*sqrt(d)))
# (24) global bound: sup h^(1/3) q(h) over (0,1/4]
import itertools
sup=max(h**(mpf(1)/3)*q_of_h(h) for h in [mpf(x)/1000 for x in range(1,250)]+[mpf('1e-4'),mpf('1e-6')])
print("== (24) sup h^(1/3)q(h) sampled:",sup, " (A_q finite)")
# (6)/(7)/(8): nu(t) t^(1/3) -> C_edge, choose c0=1,u0=1
c0=mpf(1); u0=mpf(1)
C_edge=(8*c0/7)*K_c**(-mpf(2)/3)*u0**(mpf(7)/6)
def nu(t):
    return 2*c0*quad(lambda u: u**(-mpf(1)/2)*q_of_h(t/u**2), [2*sqrt(t), u0])
for t in [mpf('1e-4'),mpf('1e-6'),mpf('1e-8')]:
    print("== t",t,"t^(1/3) nu(t) =",t**(mpf(1)/3)*nu(t)," C_edge=",C_edge)
# (26) total mass = (4 c0 eta_c/5) u0^(5/2): check via integral of nu
tot=quad(nu,[mpf(0),mpf('1e-6'),mpf('1e-3'),mpf('0.05'),mpf('0.25')])
print("== (26) int nu =",tot," expected",4*c0*eta_c/5*u0**mpf(2.5))
# (9) tip density: direct from (25) with f = 1_[t,t+dt] restricted to eta<=eta_c-eps, compare C_tip t^(1/4)
eps=mpf('0.1'); 
C_tip=c0*quad(lambda e: kappa(e)**(-mpf(5)/4),[0,eta_c-eps])
def nu_tip_direct(t):
    # density = 2 c0 int_0^{eta_c-eps} u^{3/2} * (du/dl) d eta, u = sqrt(l/kappa), du/dl = 1/(2 sqrt(l kappa))
    return 2*c0*quad(lambda e: (t/kappa(e))**(mpf(3)/4)/(2*sqrt(t*kappa(e))), [0,eta_c-eps])
for t in [mpf('1e-3'),mpf('1e-5')]:
    print("== (9) t",t,"direct",nu_tip_direct(t)," formula C_tip t^(1/4) =",C_tip*t**(mpf(1)/4), " kappa_min u0^2 =",kappa(eta_c-eps)*u0**2)
print("   ratio tip/edge at t=1e-5:", C_tip*mpf('1e-5')**(mpf(1)/4)/(C_edge*mpf('1e-5')**(-mpf(1)/3)))
# (31) physical cubic mark, L=1 omega=2pi w1=1/3
L=mpf(1); om=2*pi/L; w1=mpf(1)/3
def y1(s): return (2/om)*asin(s/sqrt(2*w1))
u=mpf('0.01')
g=2/(om*sqrt(2*w1))*(1-u/(6*w1))**(-mpf(1)/2)
k_u=sqrt(u)/(2*sqrt(3)*g**3)
print("== (30) g(u) =",g," direct y1'(s_c)=",diff(y1,-sqrt(u/3)))
for d in [mpf('1e-3'),mpf('1e-5'),mpf('1e-7')]:
    eta=eta_c-d; zm,z0,zp=roots(eta)
    sm,s0=sqrt(u)*zm,sqrt(u)*z0
    r=y1(s0)-y1(sm); ell=u**2*kappa(eta)
    print("  d",d,"ell/r^3 =",ell/r**3," k(u)=",k_u, " r/(2 g sqrt(u) a_c sqrt d)=",r/(2*g*sqrt(u)*a_c*sqrt(d)))
# physical third derivative at coalescing point: P(s(y)), s(y)=sqrt(2w1) sin(om y/2), v=eta_c u^{3/2}
v=eta_c*u**mpf(1.5)
def Pphys(y):
    s=sqrt(2*w1)*mp.sin(om*y/2); return -s**4/4+u*s**2/2+v*s
yc=y1(-sqrt(u/3))
print("== (31) phys P'(yc),P''(yc),P'''(yc):",diff(Pphys,yc,1),diff(Pphys,yc,2),diff(Pphys,yc,3)," 12k(u)=",12*k_u," 2sqrt3 sqrt(u)/g^3=",2*sqrt(3)*sqrt(u)/g**3)
# (32)
print("== (32) B_c(u)=b-u^2/12: Phi_{eta_c}(z_c)=",Phi(-1/sqrt(3),eta_c))
# (34) normalization: 2 c0 Q_c (6/7) u0^(7/6) == C_edge
print("== (7)/(34): 2 c0 Q_c (6/7) u0^(7/6) =",2*c0*Q_c*(mpf(6)/7)*u0**(mpf(7)/6)," C_edge=",C_edge)
# k(u) ~ const sqrt(u): g(0)=2/(om sqrt(2w1))
print("== k(u)/sqrt(u) at u->0:",1/(2*sqrt(3)*(2/(om*sqrt(2*w1)))**3))
