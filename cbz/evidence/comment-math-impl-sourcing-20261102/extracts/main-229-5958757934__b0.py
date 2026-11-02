from fractions import Fraction as F
from math import isqrt

checks = 0
negative_checks = 0

def check(x):
    global checks
    if not x:
        raise RuntimeError('exact positive control failed')
    checks += 1

def reject(x):
    global negative_checks
    if x:
        raise RuntimeError('negative mutation unexpectedly accepted')
    negative_checks += 1

def A(P, X):
    return P * (X + F(1, 2))**2 * (X - 1) / 12

def B(P, R, z):
    return -z*z/2 + R*z**3/(6*P*P)

def G(P, R, X, z):
    return (A(P, X) + (X*X-F(1, 4))*z/2 - z*z/2
            + X*z*z/P + (R+4)*z**3/(6*P*P))

def gradient(P, R, X, z):
    return (P*(X*X-F(1,4))/4 + X*z + z*z/P,
            (X*X-F(1,4))/2 - z + 2*X*z/P + (R+4)*z*z/(2*P*P))

def hessian(P, R, X, z):
    return (P*X/2+z, X+2*z/P, -1+2*X/P+(R+4)*z/(P*P))

for P in map(F, [4, 9, 16, 25, 100]):
    root = isqrt(P.numerator)
    check(root*root == P)
    q = 4*P*root
    for m in map(F, [1, 2, 3]):
        Rm, Rp = q-m, q+m
        check(Rm > 0)
        check(q-Rm == m and q-Rp == -m)
        for R in [Rm, Rp]:
            chi, beta = (R+4)/(P*P), 2/P
            check(chi-beta*beta == R/(P*P) > 0)
            zstar = 2*P*P/R
            height = -2*P**4/(3*R*R)
            LS = -P/24
            values = []
            for U in [F(-1,2), F(1,2)]:
                for z in [F(0), zstar]:
                    X = U-2*z/P
                    check(gradient(P,R,X,z) == (0,0))
                    hxx,hxz,hzz = hessian(P,R,X,z)
                    check(abs(hxx*hzz-hxz*hxz) == P/4)
                    expected = (0 if U<0 else LS)+(0 if z==0 else height)
                    check(G(P,R,X,z) == expected)
                    values.append(expected)
            check(len(set(values)) == 4)
            check((height>LS) == (R>q))
            reject((height>LS) == (R<q))
            for X in map(F, [-2,-1,0,1,2]):
                for z in [F(-3,2),F(-1,2),F(0),F(2,3),F(3)]:
                    U=X+2*z/P
                    check(G(P,R,X,z) == A(P,U)+B(P,R,z))
                    check(G(P,Rp,X,z)-G(P,Rm,X,z) == m*z**3/(3*P*P))
            for s in [F(-1),F(-1,2),F(0),F(1,2),F(1)]:
                check(B(P,R,zstar+s) == height+s*s/2+R*s**3/(6*P*P))
            # D=0 is R x²+4P x-P²-(R+4)/4=0. Its
            # quadratic discriminant is 4(R+4)(P²+R/4).
            disc = (4*P)**2-4*R*(-P*P-(R+4)/4)
            check(disc == 4*(R+4)*(P*P+R/4))
            for X in [F(-1,2),F(1,2)]:
                check(hessian(P,Rm,X,0) == hessian(P,Rp,X,0))
        amin = (-P/24)-(-2*P**4/(3*Rm*Rm))
        aplus = (-2*P**4/(3*Rp*Rp))-(-P/24)
        x = m/q
        check(amin == (P/24)*((1-x)**(-2)-1))
        check(aplus == (P/24)*(1-(1+x)**(-2)))
        check(amin>0 and aplus>0)
        for k in [F(1,3),F(1),F(2)]:
            gamma, rawB, lt = 1/F(root), 1/(12*k*P), F(1,24)
            rawC = [(R+4)*gamma**3/(576*k*k) for R in [Rm,Rp]]
            check(12*k*rawB/gamma**2 == 1)
            check(gamma**2/(24*lt) == 1/P)
            for R,C in zip([Rm,Rp],rawC):
                check(576*k*k*C/gamma**3 == R+4)
                check(k*rawB/lt == 2/P)
                check(k*k*C*gamma/(lt*lt) == (R+4)/(P*P))
                check((gamma/lt)*(2*P*P/R) == 12*q/R)
            check(rawC[1]-rawC[0] == m*gamma**3/(288*k*k))
            check(k*k*(rawC[1]-rawC[0])/6 == m*gamma**3/1728)

# Three meaningful normalization mutations, at a nonzero rational point.
P,R,X,z = F(9),F(107),F(2,3),F(3,2)
reject(G(P,R,X,z) == A(P,X+z/P)+B(P,R,z))
reject(G(P,R,X,z) == A(P,X+2*z/P)-z*z/2+R*z**3/(6*P))
reject(-2*P**4/(3*R**2) == -2*P**3/(3*R**2))

print(f'PASS: {checks} exact positive controls; {negative_checks} negative controls rejected.')
