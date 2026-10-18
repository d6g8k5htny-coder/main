import sympy as sp
a,b,d,k,p,C=sp.symbols('a b delta kappa p C',positive=True)
Rc=lambda t: k*t**p  # sample cumulative envelope
w=lambda lo,hi: sp.Rational(3,2)*(hi**sp.Rational(2,3)-lo**sp.Rational(2,3))
RIp=Rc(a-2*d)+Rc(b+2*d); RIm=Rc(a+2*d)+Rc(b-2*d)
P14=C*(w(a-2*d,b+2*d)-w(a+2*d,b-2*d))+RIp+RIm
shell=C*w(a-2*d,a+2*d)+Rc(a-2*d)+Rc(a+2*d)+C*w(b-2*d,b+2*d)+Rc(b-2*d)+Rc(b+2*d)
print('P14 - direct shells =',sp.simplify(P14-shell))
ER=Rc(a)+Rc(b)
print('lim P14 as delta->0 / E_R =',sp.simplify(sp.limit(P14,d,0)/ER))
# density envelope variant: R(B)=int_B r, r = k*l^(p-1)*p  (same cumulative)
l=sp.symbols('l',positive=True); r=k*p*l**(p-1)
Rd=lambda lo,hi: sp.integrate(r,(l,lo,hi))
P14d=RIp_d=Rd(a-2*d,b+2*d)+Rd(a+2*d,b-2*d)
print('density-form P14 remainder limit / R(I):',sp.simplify(sp.limit(P14d,d,0)/Rd(a,b)))
print('density direct shells remainder limit:',sp.limit(Rd(a-2*d,a+2*d)+Rd(b-2*d,b+2*d),d,0))
