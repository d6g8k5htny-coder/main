import math
from float_jfail import J
# C_fail = |S1| * (4/sqrt2) (2pi)^-3 12^-1/2 (2pi)^-1/2 * int_0^1 e^{-b^2} db * int_{1/2}^2 e^{-12k^2} k^{-5/3} J(k) dk
const = 2*math.pi*(4/math.sqrt(2))*(2*math.pi)**-3/math.sqrt(12)/math.sqrt(2*math.pi)
Ib = math.sqrt(math.pi)/2*math.erf(1)
ks=[0.5+0.05*i for i in range(0,31)]
Js={}
for k in ks:
    Js[k]=J(k,24)
# Simpson on [0.5,2] with h=0.05 (30 intervals)
h=0.05
f=[math.exp(-12*k*k)*k**(-5/3)*Js[k] for k in ks]
S=f[0]+f[-1]+4*sum(f[1:-1:2])+2*sum(f[2:-1:2])
Ik=S*h/3
print('const',const,'Ib',Ib,'Ik',Ik)
print('C_fail (plane, band) ~', const*Ib*Ik)
for k in (0.5,0.6,0.7,0.8,1.0,1.5,2.0):
    kk=min(ks,key=lambda x:abs(x-k)); print(' J(%.2f)=%.4f integrand %.4e'%(kk,Js[kk],math.exp(-12*kk*kk)*kk**(-5/3)*Js[kk]))
