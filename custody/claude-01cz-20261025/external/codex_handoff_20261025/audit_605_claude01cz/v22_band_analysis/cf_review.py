import sympy as sp, itertools
X,Y=sp.symbols('X Y')
mons=[X**a*Y**b for d in range(7) for a in range(d+1) for b in [d-a]]
assert len(mons)==28
def rows(q,eta,z,w):
    out=[]
    for m in mons:
        f=lambda x,y: m.subs({X:x,Y:y})
        fx=sp.diff(m,X); fy=sp.diff(m,Y)
        f1=lambda x,y: fx.subs({X:x,Y:y}); f2=lambda x,y: fy.subs({X:x,Y:y})
        if q==0:
            J=lambda a,b: sp.diff(m,X,a,Y,b).subs({X:0,Y:0}) if a+b>0 else m.subs({X:0,Y:0})
            pin=[J(0,0),J(1,0),J(2,0),J(3,0),J(0,1),J(1,1)]
        else:
            u0=(f(-q/2,0)+f(q/2,0))/2; u1=(f(q/2,0)-f(-q/2,0))/q
            u2=(f1(q/2,0)-f1(-q/2,0))/q; u3=6*(f1(-q/2,0)+f1(q/2,0)-2*u1)/q**2
            a1=u1-q**2*u3/24; v0=(f2(-q/2,0)+f2(q/2,0))/2; v1=(f2(q/2,0)-f2(-q/2,0))/q
            pin=[u0,a1,u2,u3,v0,v1]
        zx,zy=z; wx,wy=w
        g=[f1(zx,zy),f2(zx,zy)]
        if eta==0:
            H=sp.hessian(m,(X,Y)).subs({X:zx,Y:zy}); D=list(H*sp.Matrix([wx,wy]))
        else:
            D=[(f1(zx+eta*wx,zy+eta*wy)-g[0])/eta,(f2(zx+eta*wx,zy+eta*wy)-g[1])/eta]
        out.append(pin+g+D)
    return sp.Matrix(out).T  # 10 x 28
R=sp.Rational
pts=[((1,R(1,2)),(R(3,5),R(4,5))),((R(-3,2),R(1,3)),(1,0)),((0,1),(0,1)),((2,0),(R(5,13),R(12,13))),((R(1,5),R(-7,5)),(R(-4,5),R(3,5)))]
for z,w in pts:
    for q,eta in [(0,0),(R(1,10),R(1,10)),(R(1,100),R(1,100)),(R(1,2),0)]:
        A=rows(q,eta,z,w); rk=A.rank()
        g=(A*A.T).det()
        print(z,w,'q=',q,'eta=',eta,'rank',rk,'gram>0',g>0)
# Hessian surjectivity identity
h1,h2,w1,w2=sp.symbols('h1 h2 w1 w2'); h=sp.Matrix([h1,h2]); w=sp.Matrix([w1,w2])
H=h*w.T+w*h.T-(h.dot(w))*w*w.T
print('H symmetric:',sp.simplify(H-H.T)==sp.zeros(2)); print('Hw-h with |w|=1:',sp.simplify((H*w-h).subs(w2**2,1-w1**2)))
