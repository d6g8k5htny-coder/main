import math
from fractions import Fraction as Fr
eta=1/20
# P22 route: (1-eta)^2 m^2/(2Q) + 2r <= eta m ; n >= Q^2 L/(2 r^2). m=1 wlog (scale-free)
def nfloor_P13(Q, m=1.0):
    A=(1-eta)**2*m*m
    rmax=(eta*m - A/(2*Q))/2
    if rmax<=0: return math.inf
    return Q*Q/(2*rmax*rmax)
best=min((nfloor_P13(q/100.0),q/100.0) for q in range(1,10000))
print("P13 floor coeff (n/L) brute:",best, " closed form:", 8*(1-eta)**4/eta**4, " Q*:", (1-eta)**2/eta)
for m in (1,7,50):
    b=min((nfloor_P13(q*m/100.0,m),q*m/100.0) for q in range(1,10000))
    print("  m=",m," coeff",round(b[0],1)," Q*/m",round(b[1]/m,2))
L4=math.log(4); L320=math.log(320)
print("ln4 floor", 8*19**4*L4, " J8 a.05", 8*19**4*L320)
# typical one-sided: r + t <= eta m
def nfloor_typ(Q):
    A=(1-eta)**2
    rmax=eta - A/(4*Q)
    return math.inf if rmax<=0 else Q*Q/(2*rmax*rmax)
bt=min((nfloor_typ(q/100.0),q/100.0) for q in range(1,10000))
print("typical one-sided coeff:",bt, " closed", (1-eta)**4/(2*eta**4), " *ln4", (1-eta)**4/(2*eta**4)*L4)
# fully optimistic: interval placed anywhere in tube width 2 eta m: 2r + t <= 2 eta m
def nfloor_opt(Q):
    A=(1-eta)**2
    rmax=(2*eta - A/(4*Q))/2
    return math.inf if rmax<=0 else Q*Q/(2*rmax*rmax)
bo=min((nfloor_opt(q/100.0),q/100.0) for q in range(1,10000))
print("tube-placement optimistic coeff:",bo," closed",(1-eta)**4/(8*eta**4)," *ln4",(1-eta)**4/(8*eta**4)*L4)
# line-172 cap route with P16 Hoeffding range C_K, t ~ 0: 2r <= eta m
for K in (24,):
    CK=16*K*K
    for m in (2,10,40,255):
        print("cap route K",K,"m",m," n/L >=", 2*CK**2/(eta**2*m**2))
# Exact integer check
print(8*19**4, 19**4/2)
