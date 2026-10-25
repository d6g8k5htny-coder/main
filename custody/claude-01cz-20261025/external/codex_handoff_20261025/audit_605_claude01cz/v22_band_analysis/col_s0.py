import sympy as sp, sys
src=open(sys.argv[1]).read().split('# Jacobian')[0].replace("    print('a=%+d'%a,out)","    res.append((a,[(nm,sp.simplify(sp.limit(sp.expand(sp.cancel(sp.expand(row).subs(r,q*s)))*s,s,0))) for nm,row,lim in [('A',A,0),('B',Bv,0),('C',C,0),('D',D,0)]]))")
res=[]; exec(src.replace("    out=[]\n","    out=[]\n    continue_flag=0\n"))
for a,l in res: print('a=%+d  limit_{s->0} s*row (q,eta free):'%a, l)
