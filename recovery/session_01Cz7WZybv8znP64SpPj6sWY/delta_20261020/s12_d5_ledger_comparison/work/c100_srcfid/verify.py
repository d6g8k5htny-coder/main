import sympy as sp
x,y,a,Q,w,z,rho,c,s,b = sp.symbols('x y a Q w z rho c s b', real=True)
P = x**3/3 - x/4 - sp.Rational(1,12) + a/2*(x**2-sp.Rational(1,4))*y + Q/2*y**2 + w/2*x*y**2 + z/6*y**3
print("P(M)=",P.subs({x:-sp.Rational(1,2),y:0}), " P(S)=",P.subs({x:sp.Rational(1,2),y:0}), " Pxxx=",sp.diff(P,x,3))
def H(pt):
    return sp.Matrix([[sp.diff(P,x,2),sp.diff(P,x,y)],[sp.diff(P,x,y),sp.diff(P,y,2)]]).subs({x:pt[0],y:pt[1]}).det()
for name,(X0,sgn) in {"M":(-sp.Rational(1,2),1),"S":(sp.Rational(1,2),1)}.items():
    X = X0 + rho*c; Y = rho*s
    Px = sp.diff(P,x).subs({x:X,y:Y}); Py = sp.diff(P,y).subs({x:X,y:Y})
    sol = sp.solve([Px,Py],[a,Q],dict=True)[0]
    dM = sp.simplify(H((-sp.Rational(1,2),0)).subs(sol))
    dS = sp.simplify(H((sp.Rational(1,2),0)).subs(sol))
    dy = sp.simplify(H((X,Y)).subs(sol))
    prod = sp.simplify(dM*dS*dy)
    ser = sp.series(prod, rho, 0, 3).removeO()
    lead2 = sp.factor(sp.simplify(ser.coeff(rho,2)))
    print(name, "order rho^0 coeff:", sp.simplify(ser.coeff(rho,0)), " rho^1:", sp.simplify(ser.coeff(rho,1)))
    print(name, "rho^2 coeff:", lead2)
    target = -(-2*c**2+s**2*w)*(-4*c**3+3*c*s**2*w+s**3*z)**2/(4*s**6)
    print(name, "equals displayed leading factor?", sp.simplify(lead2-target)==0, "; with c->-c?", sp.simplify(lead2-target.subs(c,-c))==0)
    for nm,d in (("detM",dM),("detS",dS),("dety",dy)):
        sr = sp.series(d,rho,0,2).removeO()
        print("   ",name,nm," rho^0:",sp.factor(sp.simplify(sr.coeff(rho,0))),"  rho^1:",sp.factor(sp.simplify(sr.coeff(rho,1))))
    print("   forced a at rho=0:", sp.simplify(sp.limit(sol[a],rho,0)))
# coalesced six-pin jet law for K=exp(-|h|^2/2)
hx,hy=sp.symbols('hx hy')
K=sp.exp(-(hx**2+hy**2)/2)
def cov(al,be):
    return (-1)**(be[0]+be[1])*sp.diff(K,hx,al[0]+be[0],hy,al[1]+be[1]).subs({hx:0,hy:0}) if (al[0]+be[0]+al[1]+be[1])>0 else 1
obs=[(0,0),(1,0),(0,1),(2,0),(3,0),(1,1)]
free=[(0,2),(2,1),(1,2),(0,3),(4,0)]
A=sp.Matrix(len(obs),len(obs),lambda i,j: cov(obs[i],obs[j]))
Bm=sp.Matrix(len(free),len(obs),lambda i,j: cov(free[i],obs[j]))
C=sp.Matrix(len(free),len(free),lambda i,j: cov(free[i],free[j]))
k=sp.symbols('k')
e=sp.Matrix([b,0,0,0,12*k,0])
mean=Bm*A.inv()*e; condcov=C-Bm*A.inv()*Bm.T
print("free jets (f_yy,f_xxy,f_xyy,f_yyy,f_xxxx): mean", list(sp.simplify(mean)), "\ncondcov", condcov)
# normalizer limit
qq=sp.symbols('qq',real=True)
dens=sp.exp(-(qq+b)**2/4)/sp.sqrt(4*sp.pi)
z0=sp.integrate(qq**2*dens,(qq,-sp.oo,0))
print("z0(6/5)=",sp.N(z0.subs(b,sp.Rational(6,5)),12))
# diagnostic value
print("rho^2 phat0 at c=0,s=1,rho=0.02,b=6/5:", sp.N(sp.exp(-sp.Rational(36,100))/(2*sp.pi*sp.sqrt(1+sp.Rational(4,10000))),8))
# Omega inner radius along y axis
print("Omega inner radius:", sp.sqrt(4-sp.Rational(1,4)), float(sp.sqrt(4-sp.Rational(1,4))))
# P20 absorption
cc,r=sp.symbols('cc r',positive=True)
xx=cc/(2*r**2)
print("6! (2/c)^6 r^12 == 6!/x^6:", sp.simplify(sp.factorial(6)*(2/cc)**6*r**12 - sp.factorial(6)/xx**6)==0)
