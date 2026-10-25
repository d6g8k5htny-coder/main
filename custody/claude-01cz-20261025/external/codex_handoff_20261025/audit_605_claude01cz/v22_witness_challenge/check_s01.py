import sympy as sp
p,t,q,v,e,k,r=sp.symbols('p tau q v epsilon k r')
nu=t+e*v; h=p+e*q; c=p*v-q*t
# a_eps from B/t expansion vs stated
a_der=sp.simplify(((h**2/2-sp.Rational(1,8))*t-(p**2/2-sp.Rational(1,8))*nu)/(e*t*nu))
a_st=(t*p*q+e*t*q**2/2-v*(p**2/2-sp.Rational(1,8)))/(t*nu)
print('a_eps match:',sp.simplify(a_der-a_st)==0)
# N* from M*=s t J21 + t^2/2 J12
Nst21=sp.expand(((h*nu)-p*t)/e); Nst12=sp.expand((nu**2-t**2)/(2*e))
print('N* J21:',sp.simplify(Nst21-(q*t+p*v+e*q*v))==0,' J12:',sp.simplify(Nst12-(t*v+e*v**2/2))==0)
Mx=sp.Matrix([[p*t,t**2/2,0],[q*t+p*v+e*q*v,t*v+e*v**2/2,0],[a_st,q,v/2]])
print('(11) det=tau nu v c/4:',sp.simplify(Mx.det()-t*nu*v*c/4)==0)
# targets
M1=k*(sp.Rational(3,2)-6*p**2); M2=k*(sp.Rational(3,2)-6*h**2)
N=sp.expand((M2-M1)/e); print('(14) N target:',sp.simplify(N-(-6*k*(2*p*q+e*q**2)))==0)
U1=-k*r**2; U3=12*k; A1=U1-r**2*U3/24; print('(14) A1 target:',sp.simplify(A1+sp.Rational(3,2)*k*r**2)==0)
G=0
KH=G-(2*q/t**2)*M1-((-p+e*q/2)/t**2)*N
print('(19) K_H target:',sp.factor(sp.simplify(KH)),' stated diff:',sp.simplify(KH-(-3*k*q*(1-e**2*q**2)/t**2)))
# K_H cubic part on (J21,J12,J03) coefficients
row=lambda R: R
cub=[Mx[2,i]-(2*q/t**2)*Mx[0,i]-((-p+e*q/2)/t**2)*Mx[1,i] for i in range(3)]
print('(18) K_H cubic coeffs at v=0:',[sp.simplify(x.subs(v,0)) for x in cub])
print('     K_H cubic coeffs / v:',[sp.factor(sp.simplify(x/v)) for x in cub])
g=v/t; d=q-g*p
KR=N-(2*g+e*g**2)*M1
st=-sp.Rational(3,2)*k*g*(2+e*g)-12*k*p*(1+e*g)*d-6*k*e*d**2
print('(22) K_R target:',sp.simplify(KR-st)==0)
cubR=[sp.simplify(Mx[1,i]-(2*g+e*g**2)*Mx[0,i]) for i in range(3)]
print('(21) K_R cubic coeffs:',[sp.factor(x) for x in cubR],' expect nu*d*J21:',sp.simplify(cubR[0]-nu*d)==0)
