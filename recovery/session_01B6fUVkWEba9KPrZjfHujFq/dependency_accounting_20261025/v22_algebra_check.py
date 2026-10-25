import sympy as sp
a,b,c,t,s,eta,r,e=sp.symbols('a b c t s eta r e',nonzero=True)
# rows L1,L2,L3 as coefficient vectors on (J21,J12,J03), formulas (4)
L1=[c*t, t**2/2, 0]
L2=[a*t+b*c, b*t, 0]
L3=[a*c-b*c**2/(2*t), a*t, b*t/2]
M=sp.Matrix([L1,L2,L3])
d=sp.simplify(M.det()); target=b*t**3*(b*c-a*t)/4
print("det M =",sp.factor(d)," eq(5):",sp.simplify(d-target)==0)
# strata
M_b0=M.subs(b,0); print("rank at b=0:",M_b0.rank())
M_rad=M.subs(b,a*t/c); print("rank at bc=at:",sp.simplify(M_rad).rank(), " det:",sp.simplify(M_rad.det()))
# L2 = (2*lam/|z|) L1 on radial edge: a=lam*c/|z|, so 2a/c
print("L2-(2a/c)L1 on radial edge:",[sp.simplify(x) for x in (sp.Matrix([L2]).subs(b,a*t/c)-(2*a/c)*sp.Matrix([L1]))])
# generic rank 3 numeric
print("generic rank:",M.subs({a:1,b:2,c:3,t:5}).rank())
# determinant bookkeeping (6)
det6=12*r**-5*(e/s)**-2*(1/t)*s**-7
print("eq(6) rewrite:",sp.simplify(det6-12*r**-5*e**-2*(1/t)*s**-5)==0)
# scaling exponent 1+2+2+2
print("exponent sum:",1+2+2+2)
# ten-row limit rank = 7 independent first jets + rank of L-block
print("limit rank generic:",7+3,"b=0:",7+M_b0.rank(),"radial:",7+sp.simplify(M_rad).rank())
