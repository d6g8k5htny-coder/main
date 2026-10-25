import sympy as sp
s,r,e,ph,th,k,b=sp.symbols('s r e phi theta k b',positive=True)
X,Y=sp.symbols('X Y')
J={(a,c):sp.Symbol(f'J{a}{c}') for a in range(5) for c in range(5) if 2<=a+c<=4}
# confluent pin jet limit: J00=b, J10=J01=0, J20=J11=0, J30=12k
J[(2,0)]=0; J[(1,1)]=0; J[(3,0)]=12*k
F=b+sum(J[(a,c)]*X**a*Y**c/(sp.factorial(a)*sp.factorial(c)) for (a,c) in J)
Fx,Fy=sp.diff(F,X),sp.diff(F,Y)
p,t=sp.cos(th),sp.sin(th)          # witness direction, theta away from axis
xs={X:s*p,Y:s*t}
# solve grad F(x)=0 for J02 and J12 (two unknowns, leading orders)
sol=sp.solve([Fx.subs(xs),Fy.subs(xs)],[J[(0,2)],J[(1,2)]],dict=True)[0]
J02=sp.series(sol[J[(0,2)]],s,0,3).removeO()
print('J02 forced  =',sp.simplify(J02),'  (order s)')
H=sp.hessian(F,(X,Y)).subs(sol).subs(xs)
Hs=[sp.simplify(sp.series(h,s,0,2).removeO()) for h in H]
print('H(x) leading:',Hs,' -> all O(s):',all(sp.limit(h/s,s,0)!=sp.oo for h in Hs))
# pin Hessians at +-r/2 u with the forced J02
Hp=sp.hessian(F,(X,Y)).subs(sol)
for sg in (1,-1):
    d=sp.simplify(Hp.subs({X:sg*r/2,Y:0}).det())
    print('det H(pin %+d) leading:'%sg,sp.factor(sp.series(sp.series(d,r,0,2).removeO(),s,0,2).removeO()))
# height offset at the witness
dF=sp.simplify(sp.series((F-b).subs(sol).subs(xs),s,0,4).removeO())
print('F(x)-b leading:',sp.factor(dF),' (order s^3)')
print('---- collision limit: impose H(x)w=0 at leading order')
ph=sp.symbols('phi'); w=sp.Matrix([sp.cos(ph),sp.sin(ph)])
Hl=sp.Matrix(2,2,Hs)/s
c=sp.solve(list(Hl*w),[J[(2,1)],J[(0,3)]],dict=True)
print('solutions:',len(c))
for cc in c:
    lead=sp.simplify(sp.limit(dF.subs(cc)/s**3,s,0))
    print('F(x)-b = s^3 *',sp.factor(sp.simplify(lead)))
