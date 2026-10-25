import sympy as sp
u,q,x,v,t=sp.symbols('u q x v t'); cs=sp.symbols('h0:7')
h=sum(cs[i]*u**i for i in range(7))  # generic univariate degree 6
a=q/2
# cubic Hermite interpolant at +-a
A,B,C,D=sp.symbols('A B C D'); Hq=A+B*u+C*u**2+D*u**3
sol=sp.solve([Hq.subs(u,-a)-h.subs(u,-a),Hq.subs(u,a)-h.subs(u,a),sp.diff(Hq,u).subs(u,-a)-sp.diff(h,u).subs(u,-a),sp.diff(Hq,u).subs(u,a)-sp.diff(h,u).subs(u,a)],[A,B,C,D])
H0,H1,H2,H3=sol[A],sol[B],2*sol[C],6*sol[D]
hp=sp.diff(h,u)
U0=(h.subs(u,-a)+h.subs(u,a))/2; U1=(h.subs(u,a)-h.subs(u,-a))/q; U2=(hp.subs(u,a)-hp.subs(u,-a))/q; U3=6*(hp.subs(u,-a)+hp.subs(u,a)-2*U1)/q**2
print('(7) H(0)=U0-q^2/8 U2:',sp.simplify(H0-(U0-q**2/8*U2))==0,'; H\'(0)=U1-q^2/24 U3:',sp.simplify(H1-(U1-q**2/24*U3))==0,'; H\'\'=U2:',sp.simplify(H2-U2)==0,'; H\'\'\'=U3:',sp.simplify(H3-U3)==0)
ker=sp.Rational(3,4)/a**3*sp.integrate((a**2-u**2)*sp.diff(h,u,3),(u,-a,a))
print('(9) H\'\'\'(0) kernel formula:',sp.simplify(H3-ker)==0,'; kernel mass:',sp.simplify(sp.Rational(3,4)/a**3*sp.integrate(a**2-u**2,(u,-a,a))))
# (8) targets with g(-q/2)=b, g(q/2)=b-k r^3, g'(+-)=0, r=q s
b,k,s=sp.symbols('b k s',positive=True); r=q*s
G=sp.Function('G')
tU0=(b+b-k*r**3)/2; tU1=(b-k*r**3-b)/q; tU2=0; tU3=6*(0+0-2*tU1)/q**2
print('(8) H(0)=b-kr^3/2:',sp.simplify(tU0-q**2/8*tU2-(b-k*r**3/2))==0,'; H\'(0)=-3/2 k r^2 s:',sp.simplify(tU1-q**2/24*tU3+sp.Rational(3,2)*k*r**2*s)==0,'; U3=12 k s^3:',sp.simplify(tU3-12*k*s**3)==0)
# (15) annihilator
ht=6*(v/t)**2-8*(v/t)**3+3*(v/t)**4
print('(15) h_t(t)=',sp.simplify(ht.subs(v,t)),' h_t\'(t)=',sp.simplify(sp.diff(ht,v).subs(v,t)),' h_t\'\'(t)=',sp.simplify(sp.diff(ht,v,2).subs(v,t)),' h_t(0),h_t\'(0)=',ht.subs(v,0),sp.diff(ht,v).subs(v,0))
# (3) counterpath
sv=sp.symbols('sv',positive=True); rr=sv**sp.Rational(5,4)
print('(3) sigma^2/(r^2 s+s^4)^2 with sigma^2=s^8:',sp.simplify(sv**8/(rr**2*sv+sv**4)**2),' -> limit',sp.limit(sv**8/(rr**2*sv+sv**4)**2,sv,0))
