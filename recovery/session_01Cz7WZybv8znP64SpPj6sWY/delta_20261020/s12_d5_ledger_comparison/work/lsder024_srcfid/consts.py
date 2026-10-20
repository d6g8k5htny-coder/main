import mpmath as mp, sympy as sp
mu=mp.mpf('-1.2'); sd=mp.sqrt(2)
pdf=lambda x: mp.npdf(x,mu,sd)
E2neg=mp.quad(lambda x: x**2*pdf(x),[-mp.inf,0])
E3abs=mp.quad(lambda x: abs(x)**3*pdf(x),[-mp.inf,0,mp.inf])
print('E[S0^2 1{S0<0}] =',E2neg)
p=mp.mpf('0.15'); k=mp.mpf(1)/6
print('E|S0|^3 =',E3abs,' axis const (12k)^3 (1-p)^2/(1-2p)^3 E|S0|^3 =',(12*k)**3*(1-p)**2/(1-2*p)**3*E3abs)
# conditional law of S0=f_zz given (f,f_xx) for BF kernel; and Cov(S0,A4|f,f_xx)
S=sp.Matrix([[1,-1],[-1,3]]); c_S0=sp.Matrix([[-1,1]]); c_A4=sp.Matrix([[3,-15]])
print('mean coef S0 on (f,f_xx):',c_S0*S.inv(),' var S0|:',3-(c_S0*S.inv()*c_S0.T)[0],' Cov(S0,A4|):',-3-(c_S0*S.inv()*c_A4.T)[0])
# 024 (29) integral
r,t,d=sp.symbols('r t delta',positive=True)
I=sp.integrate(t**-5,(t,sp.sqrt(15)/2*r,d)); print('(29):',sp.expand(r**7*I))
# C100 axis cost at X=0 with kappa: w=kappa/(2Y^2), Var w=2
kap,Y=sp.symbols('kappa Y',positive=True); w=kap/(2*Y**2); print('cost exponent', sp.simplify(w**2/(2*2)), ' c at kappa=3/4:', sp.Rational(9,16)/16)
# P2 Region I factor bound rho^2/Delta^2 <= 1+t^2
p_,q_,rr=sp.symbols('p q r',positive=True)
print('rho^2/Delta^2 with p=tq:', sp.simplify(((p_**2+q_**2)/(q_**2+rr**2*p_**2)).subs(p_,sp.Symbol('t')*q_)))
