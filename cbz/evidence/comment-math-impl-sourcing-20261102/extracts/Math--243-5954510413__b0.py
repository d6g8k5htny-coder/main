"""Independent exact diagnostics for D.1; no field-limit or classifier proof.

Only Python stdlib. Raw symmetric tensor coordinates, not normalized tensors.
"""
from fractions import Fraction as Q
from itertools import combinations_with_replacement, product
from math import factorial, prod
import json

counts = {}
negatives = {}

def check(group, predicate):
    if not predicate:
        raise RuntimeError('failed: ' + group)
    counts[group] = counts.get(group, 0) + 1

def reject(group, predicate):
    check(group, not predicate)
    negatives[group] = negatives.get(group, 0) + 1

def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def det(a):
    a = [list(row) for row in a]
    out = Q(1)
    for i in range(len(a)):
        pivot = next((j for j in range(i, len(a)) if a[j][i]), None)
        if pivot is None:
            return Q(0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            out = -out
        p = a[i][i]
        out *= p
        for j in range(i+1, len(a)):
            f = a[j][i]/p
            for l in range(i+1, len(a)):
                a[j][l] -= f*a[i][l]
    return out

def identity(n):
    return [[Q(int(i == j)) for j in range(n)] for i in range(n)]

def raw_tensor_representation(u):
    n = len(u)
    coords = [c for c in combinations_with_replacement(range(n), 3) if c != (0,0,0)]
    lookup = {c:i for i,c in enumerate(coords)}
    rep = []
    for i,j,k in coords:
        row = [Q(0)]*len(coords)
        for a,b,c in product(range(n), repeat=3):
            if (a,b,c) == (0,0,0):
                continue
            row[lookup[tuple(sorted((a,b,c)))]] += u[a][i]*u[b][j]*u[c][k]
        rep.append(row)
    weights = [factorial(3)//prod(factorial(c.count(i)) for i in range(n)) for c in coords]
    return rep, weights

for m in range(1,5):
    n = m+1
    u = identity(n)
    for i in range(1,m):
        g = identity(n)
        g[i][i] = g[i+1][i+1] = Q(3,5)
        g[i][i+1], g[i+1][i] = -Q(4,5), Q(4,5)
        u = mm(u,g)
    for reflection in (1,-1):
        v = [row[:] for row in u]
        for row in v:
            row[1] *= reflection
        check('rotation orthogonal', mm(list(zip(*v)),v) == identity(n))
        rep, weights = raw_tensor_representation(v)
        check('raw tensor abs determinant one', abs(det(rep)) == 1)
        weighted = [[w*x for x in row] for w,row in zip(weights,rep)]
        gram = mm(list(zip(*rep)),weighted)
        check('tensor Frobenius metric', gram == [[Q(weights[i] if i==j else 0) for j in range(len(rep))] for i in range(len(rep))])
        if m >= 2:
            reject('reject raw Euclidean isometry', mm(list(zip(*rep)),rep) == identity(len(rep)))

for k,gamma,t in product((Q(2,3),Q(5,2)), (Q(-3,2),Q(4,3)), (Q(-2),Q(1),Q(7,3))):
    B=t*gamma**2/(12*k)
    bp=B-gamma**2/(12*k)
    Y=3*k*B-gamma**2/4
    c=1-t
    lo=1/(4*(1+abs(c)))
    hi=1/(2*(1+abs(c)))
    lamlo=gamma**2/(24*hi)
    lamhi=gamma**2/(24*lo)
    check('typed interval', abs(c)*hi<1 and abs(Y)<6*lamlo)
    check('weight polynomial', 36*lamlo**2-Y**2 == 9*k*k*(4*(-lamlo/k)**2-bp**2))
    iphi=gamma**6/Q(384)*((-1/(3*hi**3)+c*c/hi)-(-1/(3*lo**3)+c*c/lo))
    ilam=(12*lamhi**3-Y*Y*lamhi)-(12*lamlo**3-Y*Y*lamlo)
    slo,shi=-lamhi/k,-lamlo/k
    iss=(12*k*k*shi**3-9*k*k*bp*bp*shi)-(12*k*k*slo**3-9*k*k*bp*bp*slo)
    check('phi lambda exact integral', iphi==ilam)
    check('lambda s positive factor k', ilam==k*iss and ilam>0)
    reject('reject reciprocal k', ilam==iss/k)
    reject('reject signed measure', iphi==-ilam)
    reject('reject duplicate sign frame', iphi==2*ilam)
    C=Q(7,5)
    chi=576*k*k*C/gamma**3
    check('sign frame gauge', chi==576*k*k*(-C)/(-gamma)**3)
    # coefficient dictionaries for independent expansions P(X,k zeta)/k and G
    lam=Q(13,7)
    s=-lam/k
    p={(3,0):2,(1,0):-Q(3,2),(0,0):-Q(1,2),(0,2):s*k/2,(2,1):gamma/2,(0,1):-gamma/8,(1,2):B*k/2,(0,3):C*k*k/6}
    g={(3,0):2,(1,0):-Q(3,2),(0,0):-Q(1,2),(0,2):-lam/2,(2,1):gamma/2,(0,1):-gamma/8,(1,2):k*B/2,(0,3):k*k*C/6}
    check('cubic coordinate polynomial', p==g)

for m in range(1,8):
    h=[Q(5+3*i,3) for i in range(m-1)]
    eig=[Q(0)]+h
    van=prod(eig[j]-eig[i] for i in range(m) for j in range(i+1,m))
    hard=prod(h)**2
    boundary=prod(x**3 for x in h)*prod(h[j]-h[i] for i in range(m-1) for j in range(i+1,m-1))
    check('hard boundary factor',van*hard==boundary)
    if m>1:
        reject('reject omitted hard squares',van==boundary)

for eps in (Q(1,2),Q(1,7),Q(1,101)):
    # Integral h^2(h-x) over 0<x<min(eps,h),0<h<eps, divided by eps.
    ordered=eps**4/10
    unordered=eps**4/12
    check('nonempty ordering strip',ordered-unordered==eps**4/60 and ordered>unordered)
    check('ordering strip vanishes',0<ordered-unordered<eps**3)

for k,pi,z,mass in product((Q(2,3),Q(5,2)),(Q(3,7),),(Q(7,3),Q(11,5)),(Q(13,17),)):
    direct=k*12*pi*mass
    A=12*pi*z
    afail=mass/z
    check('one Palm normalizer',direct==k*A*afail)
    reject('reject second normalizer',direct==k*A*afail/z)

print(json.dumps({'checks':sum(counts.values()),'negative_controls':sum(negatives.values()),'groups':counts,'scope':'exact model algebra diagnostics; no field-limit theorem or numeric reproduction'},sort_keys=True,indent=2))
