from fractions import Fraction as F
import random
random.seed(7)
bad=0
for _ in range(20000):
    Q=random.randint(1,60); V=F(random.randint(1,1000))
    k=random.randint(1,8); sup=[random.randint(0,200) for _ in range(k)]
    w=[random.randint(1,50) for _ in range(k)]; s=sum(w); p=[F(x,s) for x in w]
    EN=sum(a*b for a,b in zip(p,sup)); nu=EN/V
    tQ=sum(a*max(0,b-Q) for a,b in zip(p,sup))/V
    # any valid t_j >= tQ ; other D terms >=0 ; g=Q/(2V)
    D=tQ+F(Q,2*V)
    # null: choose C0,uC,R consistent with nu in [C_- w - R, C_+ w + R]
    wj=F(random.randint(1,100),37); C=nu/wj*F(random.randint(80,120),100)
    R=abs(nu-C*wj)+F(random.randint(0,5),1000)
    uC=F(random.randint(0,10),1000)*C; C0=C+F(random.randint(-10,10),10)*uC/10
    T=uC*wj+R
    assert abs(C-C0)<=uC and abs(nu-C*wj)<=R
    if not (T+2*D>=C0*wj): bad+=1
print("violations",bad)
