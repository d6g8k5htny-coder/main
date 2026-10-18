import sympy as sp
x1,x2,x3,b,k = sp.symbols('x1 x2 x3 b k', real=True); X=(x1,x2,x3)
K = sp.exp(-(x1**2+x2**2+x3**2)/2)
def cov(a,bb):
    e=K
    for i,n in enumerate(a): e=sp.diff(e,X[i],n)
    for i,n in enumerate(bb): e=sp.diff(e,X[i],n)
    return sp.simplify((-1)**sum(bb)*e.subs({x1:0,x2:0,x3:0}))
pins = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1)]   # P's U0*
A    = [(0,2,0),(0,1,1),(0,0,2)]                                          # transverse Hessian A_0 entries
S11 = sp.Matrix(8,8,lambda i,j: cov(pins[i],pins[j]))
S21 = sp.Matrix(3,8,lambda i,j: cov(A[i],pins[j]))
S22 = sp.Matrix(3,3,lambda i,j: cov(A[i],A[j]))
v0 = sp.Matrix([b,0,0,12*k,0,0,0,0])
mean = sp.simplify(S21*S11.inv()*v0); C = sp.simplify(S22 - S21*S11.inv()*S21.T)
print('P z_0 law: A_0 | U0*=v0  mean (yy,yz,zz) =', list(mean), ' cov =', C.tolist())
# after b-integration P conditions only on V_u=(fxx,fxy,fxz)=0
V = [(2,0,0),(1,1,0),(1,0,1)]
SV = sp.Matrix(3,3,lambda i,j: cov(V[i],V[j])); SAV = sp.Matrix(3,3,lambda i,j: cov(A[i],V[j]))
print('A | V_u=0 cov =', sp.simplify(S22 - SAV*SV.inv()*SAV.T).tolist(), ' det Cov(V_u)=', SV.det(), ' tau^2=Var(fxxx|G=0)=', sp.simplify(cov((3,0,0),(3,0,0)) - cov((3,0,0),(1,0,0))**2/cov((1,0,0),(1,0,0))))
