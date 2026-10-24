import sympy as sp
r,b,k,t=sp.symbols('r b k t',positive=True)
# P's ordering O_r=(f(a),f_x(a),f(c),f_x(c), f_y1(a),f_y1(c),...)
for m in range(1,5):
    n=4+2*m
    T=sp.zeros(n,n)
    T[0,0]=sp.Rational(1,2); T[0,2]=sp.Rational(1,2)
    T[1,0]=-1/r; T[1,2]=1/r
    T[2,1]=-1/r; T[2,3]=1/r
    T[3,1]=6/r**2; T[3,3]=6/r**2; T[3,0]=12/r**3; T[3,2]=-12/r**3
    for j in range(m):
        i=4+2*j
        T[i,i]=sp.Rational(1,2); T[i,i+1]=sp.Rational(1,2)
        T[i+1,i]=-1/r; T[i+1,i+1]=1/r
    d=m+1
    print('m',m,'det',sp.simplify(T.det()), 'expected', 12/r**(d+3), sp.simplify(T.det()-12/r**(d+3))==0)
    O=sp.Matrix([b,0,b-k*r**3,0]+[0]*(2*m))
    print('  v_r', list(sp.simplify(T*O)))
# exact kernel for U3: (6/r^3) int_a^c (t-a)(c-t) g'''(t) dt  ?
a,c=-r/2,r/2
g=sp.Function('g')
# test on polynomials up to degree 7 and exp
for test in [t**3, t**4, t**5, t**6, sp.exp(t), sp.sin(2*t+1)]:
    gp=sp.diff(test,t)
    lhs=sp.Rational(6)/r**2*(gp.subs(t,a)+gp.subs(t,c)-2*(test.subs(t,c)-test.subs(t,a))/r)
    rhs=sp.Rational(6)/r**3*sp.integrate((t-a)*(c-t)*sp.diff(test,t,3),(t,a,c))
    print('U3 kernel identity', test, sp.simplify(lhs-rhs)==0)
print('kernel mass', sp.simplify(sp.Rational(6)/r**3*sp.integrate((t-a)*(c-t),(t,a,c))))
# series of U3 in terms of derivatives at 0
cs=sp.symbols('c0:8')
gg=sum(cs[i]*t**i/sp.factorial(i) for i in range(8))
gp=sp.diff(gg,t)
U=[ (gg.subs(t,a)+gg.subs(t,c))/2, (gg.subs(t,c)-gg.subs(t,a))/r, (gp.subs(t,c)-gp.subs(t,a))/r,
    6/r**2*(gp.subs(t,a)+gp.subs(t,c)-2*(gg.subs(t,c)-gg.subs(t,a))/r)]
for i,u in enumerate(U):
    print('U%d'%i, sp.expand(sp.simplify(u)))
