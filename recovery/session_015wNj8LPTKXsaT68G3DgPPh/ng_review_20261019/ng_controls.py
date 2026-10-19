import sympy as sp
r=sp.symbols('r',positive=True)
# (14): |d U_r / d O_MS| = 12 r^-(d+3), for d=1,2,3 (O_MS = (h_M, grad_M, h_S, grad_S); grad = (g_u, g_v1..))
for d in (1,2,3):
    hM,hS=sp.symbols('hM hS'); gM=sp.symbols(f'gM0:{d}'); gS=sp.symbols(f'gS0:{d}')
    O=[hM,*gM,hS,*gS]
    U=[(hM+hS)/2,(hS-hM)/r,(gS[0]-gM[0])/r,(6/r**2)*(gM[0]+gS[0]-2*(hS-hM)/r)]
    for i in range(1,d): U+=[(gM[i]+gS[i])/2,(gS[i]-gM[i])/r]
    Jm=sp.Matrix([[sp.diff(u,o) for o in O] for u in U])
    print(f'C1 d={d}: |det dU/dO| =',sp.simplify(sp.Abs(Jm.det())),' expected',12*r**(-(d+3)))
    # targets v_r from heights b, b-k r^3 and zero gradients
    b,k=sp.symbols('b k')
    sub={hM:b,hS:b-k*r**3,**{g:0 for g in gM+gS}}
    print('   v_r =',[sp.simplify(u.subs(sub)) for u in U])
# (24): midpoint map derivative det, d=3 blocks
a=sp.eye(3); print('C2 det [[I,-I/2],[I,I/2]] =',sp.Matrix(sp.BlockMatrix([[a,-a/2],[a,a/2]])).det())
# (17): block determinant identity det H = r(alpha det A - r beta^T adj(A) beta)
al,b1,b2=sp.symbols('alpha b1 b2'); A=sp.Matrix(2,2,sp.symbols('A11 A12 A21 A22')); A[1,0]=A[0,1]
beta=sp.Matrix([b1,b2]); Hm=sp.Matrix(sp.BlockMatrix([[sp.Matrix([[r*al]]),r*beta.T],[r*beta,A]]))
print('C3 det H - r(alpha detA - r beta^T adj(A) beta) =',sp.simplify(Hm.det()-r*(al*A.det()-r*(beta.T*A.adjugate()*beta)[0])))
# (26)->(27)->(28) constants and the near substitution r^-2 dr = (1/3) ell^-1/3 k^-2/3 dk
ell,kk=sp.symbols('ell k',positive=True); rr=(ell/kk)**sp.Rational(1,3)
print('C4 r^-2 |dr/dk| =',sp.simplify(rr**-2*sp.Abs(sp.diff(rr,kk))),'; 12/3 =',sp.Rational(12,3),'; 4*36 =',4*36,
      '; 144*12^(-7/3) == 12^(-1/3):',sp.simplify(144*sp.Integer(12)**sp.Rational(-7,3)-sp.Integer(12)**sp.Rational(-1,3))==0)
# §11 cumulants: X~N(0,1): E[(X^4-3)^3] and E[(X^2-1)^3]
X=sp.symbols('X'); m=lambda p: sp.factorial2(p-1) if p%2==0 else 0
E=lambda poly: sum(c*m(int(e[0])) for e,c in sp.Poly(sp.expand(poly),X).terms())
print('C5 E[(X^4-3)^3] =',E((X**4-3)**3),'(proof: 9504);  E[(X^2-1)^3] =',E((X**2-1)**3),'(proof: 8)')
# §8 integrability: int_R int_0^inf k^-2/3 (1+|b|+k)^-M dk db finite iff M>4/3 (power count at large k)
M=sp.symbols('M',positive=True)
print('C6 large-k exponent after b-integration: -2/3+1-M =',sp.simplify(-sp.Rational(2,3)+1-M),'-> integrable iff M>4/3')
# sign-cone refusal example: A~N(m,1), m=1: E[A^2 1{A>0}] vs E[A^2 1{A<0}]
z=sp.symbols('z'); dens=sp.exp(-(z-1)**2/2)/sp.sqrt(2*sp.pi)
print('C7 N(1,1): E[A^2;A>0] =',sp.N(sp.integrate(z**2*dens,(z,0,sp.oo))),' E[A^2;A<0] =',sp.N(sp.integrate(z**2*dens,(z,-sp.oo,0))))
