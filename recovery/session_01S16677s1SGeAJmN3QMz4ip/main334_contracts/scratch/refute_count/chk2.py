import numpy as np
# mixture weight normalizability at (b,s)!=(0,0), d=1, p^H_O standard Gaussian on R^4
def pFO(alpha,b,s,d=1):
    a=np.sqrt((alpha+2)/alpha)
    # substitute ap = exp(t) to resolve the a'->0 end
    t=np.linspace(np.log(1e-6),np.log(a),400001); ap=np.exp(t)
    w=alpha*a**(-alpha)*ap**(alpha-1)*ap**(-(2*d+2))*(2*np.pi)**(-(d+1))*np.exp(-(b*b+s*s)/(2*ap*ap))*ap
    return np.trapezoid(w,t)
for alpha in [0.3,1,3]:
    for (b,s) in [(0.0,-0.01),(1.0,0.99),(0.0,0.0)]:
        print("alpha",alpha,"(b,s)",(b,s),"p^F_O ~",pFO(alpha,b,s))
# check the a'^-2 bookkeeping: density of A*O at o = E[A^-(2d+2) p(o/A)], and the kernel identity
# p^F_O(b,0,s,0) E_{Q^F}[Delta^F w] = int p_A(a') a'^-2 p^H_O((b,0,s,0)/a') E_H[Delta^H w(a'H) | O^H=(b,0,s,0)/a'] da'
# via scaling: Delta^F = a'^{2d} Delta^H, so a'^-(2d+2)*a'^{2d} = a'^-2. Pure algebra; verified symbolically:
import sympy as sp
ap,d=sp.symbols("ap d",positive=True)
print("exponent check:", sp.simplify(ap**(-(2*d+2))*ap**(2*d)) )
