import numpy as np
from scipy import integrate
for alpha in [0.5,2/3,0.8,1.0,1.5]:
    a=np.sqrt((alpha+2)/alpha)
    def nuF(l):
        lo=l/a
        if lo>=1: return 0.0
        return alpha*a**(-alpha)*l**(alpha-1)*integrate.quad(lambda s:s**(-alpha),lo,1)[0]
    print(round(alpha,3),[ (l, round(l**(1/3)*nuF(l),4)) for l in [1e-2,1e-4,1e-6,1e-8]])
