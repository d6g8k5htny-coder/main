from mpmath import mp, mpf, sqrt, polyroots, quad, findroot, pi, asin, diff, log
mp.dps = 40
eta_c = 2/(3*sqrt(3))
a_c = mpf(3)**(-mpf(1)/4)
K_c = 4/mpf(3)**(mpf(5)/4)
Q_c = (mpf(2)/3)*K_c**(-mpf(2)/3)
print("eta_c",eta_c,"a_c",a_c,"K_c",K_c,"(4/3)a_c",(4*a_c/3),"Q_c",Q_c)
def Phi(z,eta): return -z**4/4 + z**2/2 + eta*z
def roots(eta):
    # -z^3 + z + eta = 0  -> z^3 - z - eta = 0
    r = polyroots([1,0,-1,-eta])
    r = sorted([x.real for x in r if abs(x.imag)<mpf(10)**(-30)])
    return r
def kappa(eta):
    zm,z0,zp = roots(eta)
    return Phi(zm,eta)-Phi(z0,eta)
def q_of_h(h):
    eta = findroot(lambda e: kappa(e)-h, (mpf('1e-6'), eta_c*(1-mpf('1e-12'))), solver='bisect', tol=mpf(10)**(-35))
    zm,z0,zp = roots(eta)
    return 1/(z0-zm), eta
print("kappa(0)=",kappa(mpf(0)))
# kappa' = z_- - z_0 check
e=mpf('0.2'); h=mpf('1e-10')
print("kappa'(0.2) numeric",(kappa(e+h)-kappa(e-h))/(2*h)," z_- - z_0 =",roots(e)[0]-roots(e)[1])
# root intervals
zm,z0,zp=roots(e); print("roots at eta=0.2:",zm,z0,zp, " expected in (-1,-1/sqrt3),(-1/sqrt3,0),(1,inf)")
# edge values
zc=-1/sqrt(3); print("Phi_{eta_c}(z_c) =",Phi(zc,eta_c),"expected -1/12 =",-mpf(1)/12)
print("z_+ at eta_c:",roots(eta_c*(1-mpf('1e-20')))[2]," 2/sqrt3=",2/sqrt(3)," Phi =",Phi(2/sqrt(3),eta_c)," expected 2/3")
# (23): h^(1/3) q(h) -> Q_c
for h in [mpf('1e-3'),mpf('1e-5'),mpf('1e-7'),mpf('1e-9')]:
    q,eta = q_of_h(h); print("h",h,"h^(1/3)q(h)",h**(mpf(1)/3)*q,"Q_c",Q_c, " delta_eta",eta_c-eta)
# q(h)->1 as h->1/4
q,eta=q_of_h(mpf('0.2499999')); print("q near 1/4:",q)
# (22): kappa(eta) ~ K_c delta^(3/2)
for d in [mpf('1e-2'),mpf('1e-4'),mpf('1e-6')]:
    print("delta",d,"kappa/(K_c delta^1.5)",kappa(eta_c-d)/(K_c*d**mpf(1.5)), " (z0-zm)/(2 a_c sqrt(delta))", (roots(eta_c-d)[1]-roots(eta_c-d)[0])/(2*a_c*sqrt(d)))
