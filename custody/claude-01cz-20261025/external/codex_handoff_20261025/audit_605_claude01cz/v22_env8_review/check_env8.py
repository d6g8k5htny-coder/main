# Independent symbolic check (reviewer side) of Env8 candidate steps 2-3. Diagnostic only.
import sympy as sp
q,eta,s,k,b=sp.symbols('q eta s k b',positive=True)
c,t,a,d=sp.symbols('c t a d',real=True); u,v=sp.symbols('u v')
Cs={(i,j):sp.Symbol(f'C{i}{j}') for i in range(4) for j in range(4) if i+j<=3}; keys=list(Cs)
P=sum(Cs[m]*u**m[0]*v**m[1] for m in keys); Pu,Pv=sp.diff(P,u),sp.diff(P,v)
r=q*s; A=q/2; h=P.subs(v,0); gv=Pv.subs(v,0)
rows=[h.subs(u,0)-(b-k*r**3/2), sp.diff(h,u).subs(u,0)+sp.Rational(3,2)*k*r**2*s, sp.diff(h,u,2).subs(u,0), sp.diff(h,u,3).subs(u,0)-12*k*s**3,
      (gv.subs(u,A)+gv.subs(u,-A))/2, (gv.subs(u,A)-gv.subs(u,-A))/(2*A),
      Pu.subs({u:c,v:t}), Pv.subs({u:c,v:t}),
      sp.expand((Pu.subs({u:c+eta*a,v:t+eta*d})-Pu.subs({u:c,v:t}))/eta), sp.expand((Pv.subs({u:c+eta*a,v:t+eta*d})-Pv.subs({u:c,v:t}))/eta)]
sol=sp.solve(rows,[Cs[m] for m in keys],dict=True)[0]
Pn=sp.expand(P.subs(sol))
print('C20,C11,C30 =',sp.simplify(sol[Cs[(2,0)]]),sp.simplify(sol[Cs[(1,1)]]),sp.simplify(sol[Cs[(3,0)]]))
# pin six observations equivalence: rows 1-6 vanish on cubic R iff R,grad R vanish at (+-A,0)
R=P; obs=[R.subs({u:A,v:0}),R.subs({u:-A,v:0}),Pu.subs({u:A,v:0}),Pu.subs({u:-A,v:0}),Pv.subs({u:A,v:0}),Pv.subs({u:-A,v:0})]
M1=sp.Matrix([[sp.diff(x,Cs[m]) for m in keys] for x in rows[:6]]); M2=sp.Matrix([[sp.diff(x,Cs[m]) for m in keys] for x in obs])
print('pin rows span = six pin observations (rank 6 each, rank 6 jointly):',M1.rank(),M2.rank(),M1.col_join(M2).rank())
H=sp.hessian(Pn,(u,v))
for sg in (1,-1):
    Hp=H.subs({u:sg*A,v:0})
    print(f'pin {sg:+d}: Puu/(k s^3 q) =',sp.simplify(Hp[0,0]/(k*s**3*q)),'; Puv/(k s^3 q) =',sp.simplify(Hp[0,1]/(k*s**3*q)),'; Pvv/(k s^3) finite at q=eta=0:',sp.limit(sp.limit(sp.simplify(Hp[1,1]/(k*s**3)),eta,0),q,0))
Hx=H.subs({u:c,v:t}); print('witness Hessian/(k s^3) at q=eta=0:',[sp.simplify(sp.limit(sp.limit(Hx[i,j]/(k*s**3),eta,0),q,0)) for i,j in ((0,0),(0,1),(1,1))])
# Hermite and Rolle orders on general quartic remainder with the pin conditions
al=sp.symbols('al',positive=True); co=sp.symbols('p0:15')
mon=[(i,j) for i in range(5) for j in range(5) if i+j<=4]
Rg=sum(co[n]*u**m[0]*v**m[1] for n,m in enumerate(mon))
cond=[Rg.subs({u:al,v:0}),Rg.subs({u:-al,v:0}),sp.diff(Rg,u).subs({u:al,v:0}),sp.diff(Rg,u).subs({u:-al,v:0}),sp.diff(Rg,v).subs({u:al,v:0}),sp.diff(Rg,v).subs({u:-al,v:0})]
ss=sp.solve(cond,[co[mon.index(m)] for m in [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1)]],dict=True)[0]
Rs=sp.expand(Rg.subs(ss))
ruu=sp.factor(sp.diff(Rs,u,2).subs({u:al,v:0})); ruv=sp.factor(sp.diff(Rs,u,v).subs({u:al,v:0}))
print('R_uu(al,0) =',ruu,' ; R_uuuu =',sp.diff(Rs,u,4)); print('R_uv(al,0) =',ruv,' ; R_uuv =',sp.factor(sp.diff(Rs,u,u,v)))
