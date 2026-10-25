import sympy as sp
q,eta,s,k,b,lam=sp.symbols('q eta s k b lam',positive=True)
c,t,a,d=sp.symbols('c t a d',real=True); u,v=sp.symbols('u v')
# independent re-solve of the full 10x10 cubic system (all ten coefficients unknown)
Cs={(i,j):sp.Symbol(f'C{i}{j}') for i in range(4) for j in range(4) if i+j<=3}
keys=list(Cs.keys())
P=sum(Cs[m]*u**m[0]*v**m[1] for m in keys); Pu,Pv=sp.diff(P,u),sp.diff(P,v)
r=q*s; A=q/2; h=P.subs(v,0); gv=Pv.subs(v,0)
rows=[h.subs(u,0)-(b-k*r**3/2), sp.diff(h,u).subs(u,0)+sp.Rational(3,2)*k*r**2*s, sp.diff(h,u,2).subs(u,0), sp.diff(h,u,3).subs(u,0)-12*k*s**3,
      (gv.subs(u,A)+gv.subs(u,-A))/2, (gv.subs(u,A)-gv.subs(u,-A))/(2*A),
      Pu.subs({u:c,v:t}), Pv.subs({u:c,v:t}),
      sp.expand((Pu.subs({u:c+eta*a,v:t+eta*d})-Pu.subs({u:c,v:t}))/eta), sp.expand((Pv.subs({u:c+eta*a,v:t+eta*d})-Pv.subs({u:c,v:t}))/eta)]
sol=sp.solve(rows,[Cs[m] for m in keys],dict=True)[0]
val=sp.together(P.subs(sol).subs({u:c,v:t})-b)
D=c*d-a*t
Pi=-k*s**3*(2*D+d*q)**2*(2*q*t+eta*(2*D+d*q))*(2*q*t-eta*(2*D-3*d*q))/(32*d*D*(t+d*eta)**2)
print('(2) exact formula matches full independent 10x10 solve:', sp.simplify(sp.cancel(val-Pi))==0)
ser=sp.series(Pi.subs({q:lam*q,eta:lam*eta}),lam,0,3).removeO()
o0=sp.simplify(ser.coeff(lam,0)); o1=sp.simplify(ser.coeff(lam,1)); lead=sp.expand(ser.coeff(lam,2))
target=-(k*s**3*D/(2*d*t**2))*(q**2*t**2-D**2*eta**2)
print('(3) order-0 and order-1 terms vanish:',o0==0,o1==0,'; order-2 term matches leading form:',sp.simplify(lead-target)==0)
print('(3a) eta=0 exact:',sp.factor(Pi.subs(eta,0)))
print('(3b) q=0 exact:',sp.factor(Pi.subs(q,0)))
roots=[sp.simplify(x) for x in sp.solve(sp.numer(sp.factor(Pi)),eta)]
print('(5) eta-roots:',roots)
