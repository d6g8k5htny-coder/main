import sympy as sp
p,t,q,v,e,r=sp.symbols('p tau q v epsilon r')
nu=t+e*v; pe=p+e*q; c=p*v-q*t; g=v/t; d=q-g*p
# (4): G=[B(eta)/nu-B(xi)/tau]/(r e) == [tau*dB - v*B(xi)]/(r tau nu), dB=(B(eta)-B(xi))/e
Bx,By=sp.symbols('Bx By')
print('(4):',sp.simplify((By/nu-Bx/t)/(r*e)-(t*(By-Bx)/e-v*Bx)/(r*t*nu))==0)
# B/r quartic: B(s,t)=[DvF-V0-r s V1]/r ; build from scalar jets J_ab up to order 4 symbolically
J={(a,b):sp.Symbol(f'J{a}{b}') for a in range(6) for b in range(6) if a+b<=5}
s_,t_=sp.symbols('s t')
def F(x,y,deg=5): return sum(J[(a,b)]*x**a*y**b/(sp.factorial(a)*sp.factorial(b)) for (a,b) in J if a+b<=deg)
X,Y=sp.symbols('X Y')
Fp=F(X,Y)
Du=sp.diff(Fp,X); Dv=sp.diff(Fp,Y)
def at(expr,x,y): return expr.subs({X:x,Y:y})
g_=lambda S_: at(Fp,S_,0); gp=lambda S_: at(Du,S_,0)
U1=(g_(r/2)-g_(-r/2))/r; U2=(gp(r/2)-gp(-r/2))/r; U3=6*(gp(-r/2)+gp(r/2)-2*U1)/r**2
V0=(at(Dv,-r/2,0)+at(Dv,r/2,0))/2; V1=(at(Dv,r/2,0)-at(Dv,-r/2,0))/r; A1=U1-r**2*U3/24
Bf=lambda s,tt: (at(Dv,r*s,r*tt)-V0-r*s*V1)/r
Mf=lambda s,tt: (at(Du,r*s,r*tt)-A1-r*s*U2-r*tt*V1)/r**2-(s**2/2)*U3
M1=Mf(p,t); M2=Mf(pe,nu); N=sp.cancel(sp.expand((M2-M1)/e))
G=sp.cancel(sp.expand((Bf(pe,nu)/nu-Bf(p,t)/t)/(r*e)))
def coef(expr,key,order):  # coefficient of J_key at r^order
    ex=sp.expand(expr); return sp.simplify(ex.coeff(J[key]).series(r,0,order+1).removeO().coeff(r,order))
KH=G-(2*q/t**2)*M1-((-p+e*q/2)/t**2)*N
KR=N-(2*g+e*g**2)*M1
bH=(-p+e*q/2)/t**2
L13=(p+e*q)/2-bH*(t**2/2+e*t*v/2+e**2*v**2/6)
print('(8) [Q_H]13:',sp.simplify(coef(KH,(1,3),1)-(q*t/6+v*L13))==0)
print('(8) [Q_R]13:',sp.simplify(coef(KR,(1,3),1)-v*nu**2/6)==0)
print('    K_R r^0 J21:',sp.simplify(coef(KR,(2,1),0)-nu*d)==0, ' J12,J03:',coef(KR,(1,2),0),coef(KR,(0,3),0))
print('    K_H r^0 J03:',coef(KH,(0,3),0))
FR=-6*(2*p*q+e*q**2)-(2*g+e*g**2)*(sp.Rational(3,2)-6*p**2)
print('(10):',sp.simplify(sp.diff(FR,e)-(-sp.Rational(3,2)*g**2-12*g*p*d-6*d**2))==0, ' affine:',sp.diff(FR,e,2)==0)
print('(9) F_R at d=0 zero iff eps*gamma=-2:',sp.factor(sp.simplify(FR.subs(q,g*p))))
# minors
rows9H=[M1,N]; 
cub=lambda ex:[coef(ex,k_,0) for k_ in [(2,1),(1,2),(0,3)]]
m=sp.Matrix([cub(M1),cub(G)]); print('radial minor excl J21 (cols J12,J03):',sp.factor(m[:,1:].det()),' expect tau^2 v/4')
m=sp.Matrix([cub(M1),cub(N)]); print('horizontal minor excl J03 (cols J21,J12):',sp.factor(m[:,:2].det()),' expect tau nu c/2 =',sp.factor(t*nu*c/2))
print('(13):',sp.simplify(t*nu*v*c/4+t**2*nu*v*d/4)==0)
