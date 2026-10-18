import sympy as sp
x,y,r,c,k,Q,t,n=sp.symbols('x y r c k Q t n',positive=True)
print('exposure (x/y)*y-x =',sp.simplify((x/y)*y-x))
print('normalization (r/k)/(c/k)-r/c =',sp.simplify((r/k)/(c/k)-r/c))
# tail square: actual polynomial expansions
X=sp.symbols('X')
p1=sp.Poly(X**2-4*Q*(X-Q),X).all_coeffs(); p2=sp.Poly((X-2*Q)**2,X).all_coeffs()
print('x^2-4Q(x-Q) coeffs',p1,' (x-2Q)^2 coeffs',p2, 'equal:',[sp.simplify(a-b) for a,b in zip(p1,p2)])
print('4Q*Q-(2Q)^2 =',sp.expand(4*Q*Q-(2*Q)**2))
# (x-Q)_+ <= x^2/(4Q) on x>=Q
print('x^2/(4Q)-(x-Q) - (x-2Q)^2/(4Q) =',sp.simplify(x**2/(4*Q)-(x-Q)-(x-2*Q)**2/(4*Q)))
# hoeffding
f=-t*n*r+n*t**2*c**2/8
ts=4*r/c**2
print('f(t*)+2nr^2/c^2 =',sp.simplify(f.subs(t,ts)+2*n*r**2/c**2))
print("f'(t*) =",sp.simplify(sp.diff(f,t).subs(t,ts)), " f''=",sp.diff(f,t,2))
# would a wrong tilt be caught by the instance check? n=7,r=3,c=5
for tt in [2*r/c**2, 8*r/c**2, 4*r/c]:
    val=f.subs(t,tt).subs({n:7,r:3,c:5}); tgt=(-2*n*r**2/c**2).subs({n:7,r:3,c:5})
    print('tilt',tt,'instance check passes?', sp.nsimplify(val)==tgt)
# bad event & whole interval
m=sp.symbols('m',positive=True)
print('beta*rare - m =', sp.simplify((1/m)*m**2-m))
T,D=sp.symbols('T D',positive=True)
print('T+D<=T+D trivially; T+2D-(T+D)=',sp.simplify(T+2*D-(T+D)))
