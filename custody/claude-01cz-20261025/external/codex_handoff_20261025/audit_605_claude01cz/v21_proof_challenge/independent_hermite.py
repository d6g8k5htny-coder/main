import sympy as sp, itertools
X,Y=sp.symbols('X Y'); R=sp.Rational
def check(sites,label):
    z=[sp.Matrix(s) for s in sites]; Zv=sp.Matrix([X,Y]); good=True
    for i in range(4):
        L=1
        for j in range(4):
            if j!=i:
                L*=(((Zv-z[j]).dot(z[i]-z[j]))/((z[i]-z[j]).dot(z[i]-z[j])))**2
        g=sum((2*(z[i]-z[j])/((z[i]-z[j]).dot(z[i]-z[j])) for j in range(4) if j!=i), sp.zeros(2,1))
        H0=(1-g.dot(Zv-z[i]))*L; Hk=[(Zv[k]-z[i][k])*L for k in range(2)]
        for j in range(4):
            sub={X:z[j][0],Y:z[j][1]}
            vals=[sp.simplify(H0.subs(sub)-(1 if i==j else 0)), sp.simplify(sp.diff(H0,X).subs(sub)), sp.simplify(sp.diff(H0,Y).subs(sub))]
            for k in range(2):
                vals+=[sp.simplify(Hk[k].subs(sub)), sp.simplify(sp.diff(Hk[k],X).subs(sub)-(1 if (i==j and k==0) else 0)), sp.simplify(sp.diff(Hk[k],Y).subs(sub)-(1 if (i==j and k==1) else 0))]
            good&=all(v==0 for v in vals)
        good&= sp.Poly(sp.expand(H0),X,Y).total_degree()<=7
    print(('PASS ' if good else 'FAIL ')+label)
check([(R(-1,2),0),(R(1,2),0),(R(1,7),R(9,8)),(R(-3,5),R(-5,4))],'Hermite (4)-(5), generic sites, degree <= 7')
check([(R(-1,2),0),(R(1,2),0),(R(3,2),0),(R(-2),0)],'Hermite (4)-(5), fully collinear sites')
