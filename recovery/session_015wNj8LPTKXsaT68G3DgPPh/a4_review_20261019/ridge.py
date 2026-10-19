# Check (4): third, fourth and fifth derivatives of the transverse maximizing ridge
# g(s) = max_t F(s,t) near (0,0), for a general quintic Taylor polynomial with
# F_s=F_t=F_st=0, F_ss=-lam, F_tt=a at the origin.
import sympy as sp
s,t=sp.symbols('s t')
lam,a=sp.symbols('lam a')
# raw derivative coefficients F_{s^i t^j}
c={}
F=-lam*s**2/2+a*t**2/2
for n in range(3,6):
    for i in range(n+1):
        j=n-i
        c[(i,j)]=sp.Symbol(f'F{i}{j}')
        F+=c[(i,j)]*s**i*t**j/(sp.factorial(i)*sp.factorial(j))
# solve F_t(s,t(s))=0 as a power series t(s)=sum_k t_k s^k, k=1..4
tk=sp.symbols('t1:5')
T=sum(tk[k-1]*s**k for k in range(1,5))
eq=sp.expand(sp.diff(F,t).subs(t,T))
sol={}
for k in range(1,5):
    ck=sp.expand(eq.coeff(s,k)).subs(sol)
    sk=sp.solve(ck,tk[k-1])[0]
    sol[tk[k-1]]=sp.simplify(sk)
g=sp.expand(F.subs(t,T.subs(sol)))
g3=sp.simplify(sp.factorial(3)*g.coeff(s,3))
g4=sp.simplify(sp.factorial(4)*g.coeff(s,4))
g5=sp.simplify(sp.factorial(5)*g.coeff(s,5))
gam=c[(2,1)]; eta=c[(3,1)]; B=c[(1,2)]
print('t1..t4 =',[sol[x] for x in tk])
print('g3 - F30 =',sp.simplify(g3-c[(3,0)]))
print('g4 - (F40-3gam^2/a) =',sp.simplify(g4-(c[(4,0)]-3*gam**2/a)))
print('g5 - (F50-10 eta gam/a+15 gam^2 B/a^2) =',sp.simplify(g5-(c[(5,0)]-10*eta*gam/a+15*gam**2*B/a**2)))
print('g5 =',sp.factor(g5))
