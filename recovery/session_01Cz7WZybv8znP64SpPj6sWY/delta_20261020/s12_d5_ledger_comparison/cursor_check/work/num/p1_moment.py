import sys; sys.path.insert(0, sys.argv[1])
import mpmath as mp
mp.mp.dps=40
Phi=lambda x: mp.ncdf(x); phi=lambda x: mp.npdf(x)
def m2neg(mu,sig):
    a=-mu/sig
    return (mu**2+sig**2)*Phi(a)-mu*sig*phi(a)
mu=mp.mpf(-6)/5
for lab,sig in [('variance 2 (sd sqrt2)',mp.sqrt(2)),('sd 2 (variance 4)',mp.mpf(2))]:
    v=m2neg(mu,sig)
    num=mp.quad(lambda x: x**2*mp.npdf(x,mu,sig),[-mp.inf,0])
    print(lab,'closed form =',mp.nstr(v,12),' quadrature =',mp.nstr(num,12))
# law of S0=f_yy(0) on BF kernel given f(0)=b,f_xx(0)=0 (f_xy, odd derivs uncorrelated)
K=mp.matrix([[1,-1],[-1,3]]); c=mp.matrix([[-1,1]]); b=mu*-1
G=c*mp.inverse(K)
print('S0 mean =',mp.nstr((G*mp.matrix([b,0]))[0],12),' var =',mp.nstr(3-(G*c.T)[0],12))
