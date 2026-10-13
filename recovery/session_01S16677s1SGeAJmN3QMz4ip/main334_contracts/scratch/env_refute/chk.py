import numpy as np
from scipy import integrate
# toy H_SG(b,k)=C(1+|b|+k)^(2d) exp[-c(b^2+k^2)], d=2
d=2; C=1.0; c=0.5
H=lambda b,k: C*(1+abs(b)+k)**(2*d)*np.exp(-c*(b*b+k*k))
I_SG=integrate.dblquad(lambda b,k: k**(-2/3)*H(b,k), 0, 40, -40, 40)[0]
for alpha in [0.5, 2/3, 0.8, 1.0, 1.5, 3.0]:
    a=np.sqrt((alpha+2)/alpha)
    pA=lambda s: alpha*a**(-alpha)*s**(alpha-1)
    # E[A^{-2/3}] numeric vs formula
    EAm=integrate.quad(lambda s: pA(s)*s**(-2/3), 0, a, limit=200)[0] if alpha>2/3 else np.inf
    form=a**(-2/3)*alpha/(alpha-2/3) if alpha>2/3 else np.inf
    # weighted integral of H^F(b,k)=E[A^-2 H(b/A,k/A)] on a box, via Tonelli in (A,b,k)
    HF=lambda b,k: integrate.quad(lambda s: pA(s)*s**-2*H(b/s,k/s), 0, a, limit=200)[0]
    # compare on small k sample the b-integrated envelope ~ k^(alpha-1)
    ks=[1e-3,1e-2,1e-1]
    hk=[integrate.quad(lambda b: HF(b,k), -30, 30, limit=200)[0] for k in ks]
    slopes=np.diff(np.log(hk))/np.diff(np.log(ks))
    print(f"alpha={alpha:.3f} a={a:.3f} E[A^-2/3] num={EAm:.5f} formula={form:.5f}  log-slope of int H^F db near k->0: {slopes}")
print("I_SG=",I_SG)
