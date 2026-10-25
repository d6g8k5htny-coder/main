import sympy as sp
q,eta,s,k,b=sp.symbols('q eta s k b',positive=True)
c,t,a,d=sp.symbols('c t a d',real=True); u,v=sp.symbols('u v')
C02,C12,C21,C03=sp.symbols('C02 C12 C21 C03')
r=q*s
C10=-sp.Rational(3,2)*k*r**2*s; C30=2*k*s**3; C01=-C21*q**2/4
P=(b-k*r**3/2)+C10*u+C30*u**3+C01*v+C02*v**2+C21*u**2*v+C12*u*v**2+C03*v**3
Pu,Pv=sp.diff(P,u),sp.diff(P,v)
e1=Pu.subs({u:c,v:t}); e2=Pv.subs({u:c,v:t})
e3=sp.expand((Pu.subs({u:c+eta*a,v:t+eta*d})-e1)/eta); e4=sp.expand((Pv.subs({u:c+eta*a,v:t+eta*d})-e2)/eta)
sol=sp.solve([e1,e2,e3,e4],[C02,C12,C21,C03],dict=True)[0]
val=sp.together(P.subs(sol).subs({u:c,v:t})-b)
num,den=sp.fraction(val); num=sp.factor(sp.expand(num)); den=sp.factor(den)
print('denominator:',den)
# expansions: divide by k s^3
expr=sp.expand(sp.cancel(val/(k*s**3)))
e00=sp.simplify(expr.subs({q:0,eta:0})); print('coef of k s^3 at q=eta=0:',e00)
dq2=sp.factor(sp.simplify(sp.diff(expr,q,2).subs({q:0,eta:0})/2)); print('coef of k s^3 q^2 (=k r^2 s) at eta=0:',dq2)
deta=sp.factor(sp.simplify(sp.diff(expr,eta).subs({q:0,eta:0}))); print('coef of k s^3 eta (=k e s^2) at q=0:',deta)
dq1=sp.simplify(sp.diff(expr,q).subs({q:0,eta:0})); print('linear-in-q coef:',dq1)
open('/tmp/claude-0/-home-user-main/4dd532d5-b4fa-5c8e-a137-a0ec5074f318/scratchpad/hm/coefs.txt','w').write(f'den={den}\ne00={e00}\nkappa_r2s={dq2}\nkappa_es2={deta}\n')
print('---- exact')
full=sp.factor(sp.cancel(val))
print('P_nu(z)-b =',full)
print('at q=0 (all eta):',sp.simplify(val.subs(q,0)))
kap=(a*t-c*d)/(2*d)
rest=sp.factor(sp.cancel(val-(k*r**2*s*kap - k*r**3/2)))
print('P_nu(z)-b - [k r^2 s kappa_hat - k r^3/2] =',rest)
print('---- zero set and leading form')
D=sp.Symbol('D'); 
lead=-(k*s/(2*d*t**2))*(c*d-a*t)*(r**2*t**2-(c*d-a*t)**2*(eta*s)**2)
diff=sp.simplify(sp.expand(sp.series(sp.series((val-lead)/(k*s**3),q,0,3).removeO(),eta,0,3).removeO()))
print('val - leading form, through second order in (q,eta):',sp.factor(diff))
print('positive roots in eta:',sp.solve(sp.numer(sp.together(val)),eta))
