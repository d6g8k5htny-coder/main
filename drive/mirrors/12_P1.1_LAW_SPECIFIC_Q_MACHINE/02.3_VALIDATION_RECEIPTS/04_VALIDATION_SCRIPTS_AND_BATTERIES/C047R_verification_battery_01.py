# U-ID: U009-appendix | Claude / C047R | 2026-07-20
# Re-runnable verification battery for DEEP_DIVE_01 (C099/C100). Requires sympy.
import sympy as sp
x,y = sp.symbols('x y'); K = sp.exp(-(x**2+y**2)/2)
jets=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
C = sp.Matrix(10,10, lambda i,j: ((-1)**(jets[j][0]+jets[j][1]))*sp.diff(K,x,jets[i][0]+jets[j][0],y,jets[i][1]+jets[j][1]).subs({x:0,y:0}))
pins=[0,1,2,3,4,6]; free=[5,7,8,9]
Ccond = sp.simplify(C[free,free]-C[free,pins]*C[pins,pins].inv()*C[pins,free])
assert Ccond==sp.diag(2,2,2,6)
assert list(sp.simplify(C[free,pins]*C[pins,pins].inv()).row(0))==[-1,0,0,0,0,0]  # E[fyy|pins]=-b
c,s,t = sp.symbols('c s t', positive=True); ayy,axxy,axyy,ayyy = sp.symbols('ayy axxy axyy ayyy')
fl = ayy*y**2/2+axxy*x**2*y/2+axyy*x*y**2/2+ayyy*y**3/6
flx,fly = sp.diff(fl,x),sp.diff(fl,y); sub={x:t*c,y:t*s}
Y1=sp.limit(sp.expand((fl.subs(sub)-(t/2)*(c*flx.subs(sub)+s*fly.subs(sub)))/t**3),t,0)
Y2=sp.limit(sp.expand(flx.subs(sub)/t**2),t,0); Y3=sp.limit(sp.expand(fly.subs(sub)/t),t,0)
V={ayy:2,axxy:2,axyy:2,ayyy:6}
cov=lambda u,v: sp.simplify(sum(sp.diff(u,g)*sp.diff(v,g)*w_ for g,w_ in V.items()))
M=sp.Matrix(3,3,lambda i,j:cov([Y1,Y2,Y3][i],[Y1,Y2,Y3][j]))
claim=sp.Matrix([[s**2*(3*c**4+3*c**2*s**2+s**4)/24,-c*s**2*(2*c**2+s**2)/4,0],
                 [-c*s**2*(2*c**2+s**2)/4,s**2*(4*c**2+s**2)/2,0],[0,0,2*s**2]])
assert sp.simplify(M-claim)==sp.zeros(3,3)
assert sp.simplify(M.det()-s**8*(c**2+s**2)*(3*c**2+s**2)/24)==0
Vb=dict(V); Vb[ayyy]=5
assert sp.simplify(sum(sp.diff(Y1,g)**2*w_ for g,w_ in Vb.items())-claim[0,0])!=0  # control
X_,Y_,b = sp.symbols('X Y b',real=True); q,a,w,z,Q = sp.symbols('q a w z Q')
G1=X_**2-sp.Rational(1,4)+a*X_*Y_+w*Y_**2/2; G2=q*Y_; Vq={a:2,w:2,q:2}
v11=sp.simplify(sum(sp.diff(G1,g)**2*w_ for g,w_ in Vq.items())); v22=sp.simplify(sum(sp.diff(G2,g)**2*w_ for g,w_ in Vq.items()))
assert sp.simplify(v11-Y_**2*(4*X_**2+Y_**2)/2)==0 and v22==2*Y_**2
assert sp.simplify((X_**2-sp.Rational(1,4))**2/(2*v11)+(b*Y_)**2/(2*v22)-((X_**2-sp.Rational(1,4))**2/(Y_**2*(4*X_**2+Y_**2))+b**2/4))==0
rho=sp.symbols('rho',positive=True)
P=x**3/3-x/4-sp.Rational(1,12)+(a/2)*(x**2-sp.Rational(1,4))*y+(Q/2)*y**2+(w/2)*x*y**2+(z/6)*y**3
Px,Py=sp.diff(P,x),sp.diff(P,y); x0,y0=-sp.Rational(1,2)+rho*c,rho*s
sol=sp.solve([Px.subs({x:x0,y:y0}),Py.subs({x:x0,y:y0})],[Q,a],dict=True)[0]; Ps=P.subs(sol)
H=lambda pt: sp.det(sp.Matrix([[sp.diff(Ps,x,2),sp.diff(Ps,x,1,y,1)],[sp.diff(Ps,x,1,y,1),sp.diff(Ps,y,2)]]).subs(pt))
prod=sp.simplify(H({x:-sp.Rational(1,2),y:0})*H({x:sp.Rational(1,2),y:0})*H({x:x0,y:y0}))
ser=sp.series(prod,rho,0,3).removeO()
assert sp.simplify(ser.coeff(rho,0))==0 and sp.simplify(ser.coeff(rho,1))==0
assert sp.simplify(ser.coeff(rho,2)-(-(-2*c**2+s**2*w)*(-4*c**3+3*c*s**2*w+s**3*z)**2/(4*s**6)))==0
Psb=P.subs({a:sol[a]})  # control: partial station solve
Hb=lambda pt: sp.det(sp.Matrix([[sp.diff(Psb,x,2),sp.diff(Psb,x,1,y,1)],[sp.diff(Psb,x,1,y,1),sp.diff(Psb,y,2)]]).subs(pt))
assert sp.simplify(sp.series(sp.simplify(Hb({x:-sp.Rational(1,2),y:0})*Hb({x:sp.Rational(1,2),y:0})*Hb({x:x0,y:y0})),rho,0,1).removeO().coeff(rho,0))!=0
assert sp.simplify(sp.solve(Px.subs({x:0,y:Y_}),w)[0]-1/(2*Y_**2))==0
import math; assert abs((1-0.25)-4/60-41/60+4/15/4- (0))<1 and abs(math.sqrt(1-0.25)**2-4/15/4-41/60)<1e-12
print("ALL BATTERIES PASS (with controls)")
