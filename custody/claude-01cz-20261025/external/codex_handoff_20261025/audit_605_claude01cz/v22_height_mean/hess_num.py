import sympy as sp
eta=sp.symbols('eta',positive=True); u,v=sp.symbols('u v'); R=sp.Rational
C02,C12,C21,C03=sp.symbols('C02 C12 C21 C03')
for (c,t,a,d,q) in [(1,1,0,1,R(1,50)),(R(1,2),R(3,2),R(3,5),R(4,5),R(1,40)),(-1,R(1,2),R(4,5),-R(3,5),R(1,30))]:
    s=R(1,100); k=1; b=0; r=q*s
    P=(b-k*r**3/2)-R(3,2)*k*r**2*s*u+2*k*s**3*u**3-C21*q**2/4*v+C02*v**2+C21*u**2*v+C12*u*v**2+C03*v**3
    Pu,Pv=sp.diff(P,u),sp.diff(P,v)
    e1=Pu.subs({u:c,v:t}); e2=Pv.subs({u:c,v:t})
    e3=sp.expand((Pu.subs({u:c+eta*a,v:t+eta*d})-e1)/eta); e4=sp.expand((Pv.subs({u:c+eta*a,v:t+eta*d})-e2)/eta)
    sol=sp.solve([e1,e2,e3,e4],[C02,C12,C21,C03],dict=True)[0]
    Pn=P.subs(sol); D=c*d-a*t
    roots=[-2*q*t/(2*D+d*q), 2*q*t/(2*D-3*d*q)]; es=[x for x in roots if x>0][0]
    val=sp.simplify(Pn.subs({u:c,v:t})-b)
    Hx=sp.hessian(Pn,(u,v)).subs({u:c,v:t}); Hy=sp.hessian(Pn,(u,v)).subs({u:c+eta*a,v:t+eta*d})
    dx=sp.simplify(Hx.det()); dy=sp.simplify(Hy.det())
    print('point',(c,t,a,d,q),' eta*=',sp.N(es,6),' Pi(eta*)=',sp.N(val.subs(eta,es),3))
    for e in [es/4, es/2, es, 2*es, 4*es]:
        print('   eta=%.5f  detHx=%.3e  detHy=%.3e  Pi=%.3e'%(float(e),float(dx.subs(eta,e)),float(dy.subs(eta,e)),float(val.subs(eta,e))))
