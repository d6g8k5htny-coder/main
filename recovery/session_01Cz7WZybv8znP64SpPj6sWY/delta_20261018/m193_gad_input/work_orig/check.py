import sympy as sp, itertools, math
a,k4,k6,b,c,V,l,u,r,d = sp.symbols('a kappa4 kappa6 b c V ell u r d', positive=True)

print("== constants ==")
# proof (19) absolute moment
y=sp.symbols('y',real=True)
absmom = 2*sp.integrate(y**sp.Rational(4,3)*sp.exp(-y**2/(2*V))/sp.sqrt(2*sp.pi*V),(y,0,sp.oo))
print("E|N(0,V)|^(4/3) =", sp.simplify(absmom), " proof(19) diff:", sp.simplify(absmom - 2**sp.Rational(2,3)*sp.gamma(sp.Rational(7,6))*V**sp.Rational(2,3)/sp.sqrt(sp.pi)))
onesided = absmom/2
print("ORIGINAL(23) diff:", sp.simplify(onesided - 2**sp.Rational(-1,3)*sp.gamma(sp.Rational(7,6))*V**sp.Rational(2,3)/sp.sqrt(sp.pi)))
print("proof: 2^(2/3)/(2*12^(1/3)) - 24^(-1/3) =", sp.nsimplify(sp.simplify(2**sp.Rational(2,3)/(2*12**sp.Rational(1,3)) - 24**sp.Rational(-1,3))), float(2**(2/3)/(2*12**(1/3))-24**(-1/3)))
print("ORIGINAL: 12^(-1/3)*2^(-1/3) - 24^(-1/3) =", float(12**(-1/3)*2**(-1/3)-24**(-1/3)))

# ORIGINAL (21) radial delta integral: root r0
r0 = (12*l/u)**sp.Rational(1,3)
val = (u**2/4)*r0/ (u*r0**2/4)
print("ORIGINAL(21) diff:", sp.simplify(val - 12**sp.Rational(-1,3)*u**sp.Rational(4,3)*l**sp.Rational(-1,3)))
# ORIGINAL (24) substitution
t=sp.symbols('t',positive=True)
p=sp.Function('p')
U=sp.symbols('U',positive=True)
tt=(12/U)**sp.Rational(1,3)
integrand = 3*tt**-2*(12/tt**3)**2*sp.Abs(sp.diff(tt,U))
print("ORIGINAL(24) jac factor:", sp.simplify(integrand), "vs 12^(-1/3)U^(4/3)")
# Jacobian (18): per-coordinate gradient map det
M = sp.Matrix([[sp.Rational(1,2),sp.Rational(1,2)],[-1/r,1/r]])
print("per-coord det:", sp.simplify(M.det()), " -> total 12 r^-(d+3)")

print("== tensor identities (sympy, d=3, generic moments) ==")
D=3
# symbolic moments of a single symmetric coordinate: E Q^2=a, EQ^4=b, EQ^6=c
def mom(n):
    if n%2: return 0
    return {0:1,2:a,4:b,6:c}[n]
def Emon(powers):
    out=1
    for pw in powers: out*=mom(pw)
    return out
e=sp.symbols('e1:4',real=True)
def E_prod(idx):
    pw=[0]*D
    for i in idx: pw[i]+=1
    return Emon(pw)
K4=b-3*a**2; K6=c-15*a*b+30*a**3
delta=lambda i,j: 1 if i==j else 0
ok=True
for i,j,k,m in itertools.product(range(D),repeat=4):
    lhs=E_prod((i,j,k,m))
    rhs=a**2*(delta(i,j)*delta(k,m)+delta(i,k)*delta(j,m)+delta(i,m)*delta(j,k))+K4*sum(delta(i,s)*delta(j,s)*delta(k,s)*delta(m,s) for s in range(D))
    if sp.expand(lhs-rhs)!=0: ok=False
print("proof(8)=ORIGINAL(5)=E[QiQjQkQl]:", ok)
# polynomial expectations via expansion
def Epoly(expr):
    Qs=sp.symbols('Q1:4')
    expr=sp.expand(expr)
    poly=sp.Poly(expr,*Qs)
    tot=0
    for monom,coef in poly.terms():
        tot+=coef*Emon(monom)
    return sp.expand(tot)
Qs=sp.symbols('Q1:4')
Qe=sum(Qs[i]*e[i] for i in range(D))
sub1 = {e[2]: sp.sqrt(1-e[0]**2-e[1]**2)}
S4=sum(x**4 for x in e); S6=sum(x**6 for x in e)
v=[Epoly(Qs[i]*Qe**3) for i in range(D)]
print("v_i (9)/(7):", all(sp.simplify((v[i]-(3*a**2*e[i]+K4*e[i]**3)).subs(sub1))==0 for i in range(D)))
m6=Epoly(Qe**6)
print("E(Q.e)^6 proof/ORIGINAL(9):", sp.simplify((m6-(15*a**3+15*a*K4*S4+K6*S6)).subs(sub1))==0)
vv=sum(x**2 for x in v)
print("|v|^2:", sp.simplify((vv-(9*a**4+6*a**2*K4*S4+K4**2*S6)).subs(sub1))==0)
Vexpr=m6-vv/a
print("V(e) proof(10)=ORIGINAL(8):", sp.simplify((Vexpr-(6*a**3+9*a*K4*S4+(K6-K4**2/a)*S6)).subs(sub1))==0)
Vc=(6*a**3+9*a*K4+(K6-K4**2/a))
print("coord dir c-b^2/a:", sp.simplify(Vc-(c-b**2/a))==0)
# B(e)
Bm=sp.Matrix(D,D,lambda i,k: Epoly(Qs[i]*Qs[k]*Qe**2))
Bf=a**2*(sp.eye(D)+2*sp.Matrix(e)*sp.Matrix(e).T)+K4*sp.diag(*[x**2 for x in e])
print("B(e) proof(11)=ORIGINAL(6):", sp.simplify((Bm-Bf).subs(sub1))==sp.zeros(D,D))
