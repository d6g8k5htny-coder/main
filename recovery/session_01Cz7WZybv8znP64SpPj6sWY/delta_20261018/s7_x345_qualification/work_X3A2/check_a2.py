import sympy as sp
r,b,k,h,t = sp.symbols('r b k h t', positive=True)
# (3.1)-(3.2): build T_r for m transverse dirs, O_r order as in P l.73-74
def T(m):
    n = 4+2*m
    M = sp.zeros(n,n)
    # columns: 0 f(a),1 fx(a),2 f(c),3 fx(c), then (fy_j(a), fy_j(c))
    M[0,0]=sp.Rational(1,2); M[0,2]=sp.Rational(1,2)
    M[1,0]=-1/r; M[1,2]=1/r
    M[2,1]=-1/r; M[2,3]=1/r
    M[3,1]=6/r**2; M[3,3]=6/r**2; M[3,2]=-12/r**3; M[3,0]=12/r**3
    for j in range(m):
        M[4+2*j,4+2*j]=sp.Rational(1,2); M[4+2*j,5+2*j]=sp.Rational(1,2)
        M[5+2*j,4+2*j]=-1/r; M[5+2*j,5+2*j]=1/r
    return M
for m in range(1,5):
    d=m+1
    D=sp.simplify(T(m).det())
    print('m',m,'det T_r =',D,' |det|==12 r^-(d+3):',sp.simplify(sp.Abs(D)-12*r**(-(d+3)))==0)
# axial block with X3 column order (f(a),f(c),fx(a),fx(c))
Ax = sp.Matrix([[sp.Rational(1,2),sp.Rational(1,2),0,0],[-1/r,1/r,0,0],[12/r**3,-12/r**3,6/r**2,6/r**2],[0,0,-1/r,1/r]])
# rows U0,U1,U3,U2? reorder to U0,U1,U2,U3
Ax = sp.Matrix([[sp.Rational(1,2),sp.Rational(1,2),0,0],[-1/r,1/r,0,0],[0,0,-1/r,1/r],[12/r**3,-12/r**3,6/r**2,6/r**2]])
print('value block det', sp.simplify(Ax[:2,:2].det()), ' deriv block det', sp.simplify(Ax[2:,2:].det()), ' axial det', sp.simplify(Ax.det()))
# (3.3)
O = sp.Matrix([b,0,b-k*r**3,0,0,0,0,0])
print('v_r =', list(sp.simplify(T(2)*O)))
v0 = sp.Matrix([b,0,0,12*k,0,0,0,0])
print('v_r - v_0 =', list(sp.simplify(T(2)*O - v0)))
# (3.4) Taylor of U3 along the axis, a=-h,c=h, h=r/2
f = sp.Function('f')
x=sp.symbols('x')
cs = sp.symbols('c0:9')
poly = sum(cs[i]*x**i/sp.factorial(i) for i in range(9))
fa=poly.subs(x,-r/2); fc=poly.subs(x,r/2)
fxa=sp.diff(poly,x).subs(x,-r/2); fxc=sp.diff(poly,x).subs(x,r/2)
U0=(fa+fc)/2; U1=(fc-fa)/r; U2=(fxc-fxa)/r; U3=6/r**2*(fxa+fxc-2*(fc-fa)/r)
inner = sp.expand(fxa+fxc-2*(fc-fa)/r)
print('inner =', sp.series(inner, r, 0, 6))
for name,U in [('U0',U0),('U1',U1),('U2',U2),('U3',U3)]:
    print(name, sp.expand(U))
