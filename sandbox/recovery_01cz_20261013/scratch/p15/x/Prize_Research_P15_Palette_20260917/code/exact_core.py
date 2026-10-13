"""Exact finite companions. No theorem is inferred from a finite test count."""
from __future__ import annotations
from fractions import Fraction as F
from itertools import combinations, product
from math import comb
from typing import Iterable, Sequence

class GuardFailure(RuntimeError):
    pass

def need(condition: bool, label: str) -> None:
    if not condition:
        raise GuardFailure(label)

def rational(x: object) -> F:
    need(isinstance(x, (str,int,F)) and not isinstance(x,bool), 'rational-input-required')
    return F(x)

def mask(vertices: Iterable[int]) -> int:
    result=0
    for v in vertices:
        need(isinstance(v,int) and v>=0, 'vertex-index')
        result |= 1<<v
    return result

def minimal(edges: Iterable[int]) -> tuple[int,...]:
    ordered=sorted(set(edges),key=lambda e:(e.bit_count(),e))
    result=[]
    for e in ordered:
        if not any(e&f==f for f in result): result.append(e)
    return tuple(result)

def good_table(n: int, edges: Sequence[int]) -> list[bool]:
    need(n>=0 and all(0<e<(1<<n) for e in edges),'nonempty-bounded-witnesses')
    return [not any(u&e==e for e in edges) for u in range(1<<n)]

def part_table(good: Sequence[bool]) -> list[int|None]:
    """Exact minimum number of good parts; None means no finite partition."""
    N=len(good)
    need(N>0 and N&(N-1)==0 and good[0], 'downset-table-size')
    result: list[int|None]=[None]*N
    result[0]=0
    for u in range(1,N):
        if good[u]: result[u]=1; continue
        first=u&-u; sub=u; best=None
        while sub:
            if sub&first and good[sub] and result[u^sub] is not None:
                v=1+result[u^sub]
                if best is None or v<best:best=v
            sub=(sub-1)&u
        result[u]=best
    return result

def hard(value: int|None,k: int) -> bool:
    return value is None or value>k

def selected_probability(p: Sequence[F], selected: int) -> F:
    value=F(1)
    for i,x in enumerate(p):
        need(0<=x<=1,'probability-range')
        value*=x if selected>>i&1 else 1-x
    return value

def prob(p: Sequence[F], event: Sequence[bool]) -> F:
    need(len(event)==1<<len(p),'event-size')
    return sum((selected_probability(p,u) for u,yes in enumerate(event) if yes),F(0))

def price(weights: Sequence[F], e: int) -> F:
    value=F(1)
    for i,w in enumerate(weights):
        if e>>i&1:value*=w
    return value

def cost(weights: Sequence[F], gens: Iterable[int]) -> F:
    return sum((price(weights,g) for g in set(gens)),F(0))

def covered(u: int, gens: Iterable[int]) -> bool:
    return any(u&g==g for g in gens)

def localize(edges: Sequence[int], vertices: Sequence[int]) -> tuple[int,...]:
    B=mask(vertices)
    return minimal(sum(1<<j for j,v in enumerate(vertices) if e>>v&1)
                   for e in edges if e&B==e)

def lift(g: int, vertices: Sequence[int]) -> int:
    return sum(1<<v for j,v in enumerate(vertices) if g>>j&1)

def intersection(sets: Sequence[set[int]]) -> set[int]:
    need(bool(sets),'nonempty-intersection-list')
    r=set(sets[0])
    for s in sets[1:]:r.intersection_update(s)
    return r

def validate_palettes(n: int,edges: Sequence[int],blocks: Sequence[Sequence[int]],
                      palettes: Sequence[Sequence[int]],demands: Sequence[int],K: int) -> None:
    need(K>=1 and len(blocks)==len(palettes)==len(demands),'palette-dimensions')
    flat=[v for B in blocks for v in B]
    need(len(flat)==n and sorted(flat)==list(range(n)) and all(blocks),'true-disjoint-partition')
    P=[]
    for B,p,k in zip(blocks,palettes,demands):
        need(k>=1 and len(p)==len(set(p)) and len(p)>=k,'local-palette-demand')
        need(all(isinstance(x,int) and 0<=x<K for x in p),'palette-label-range')
        P.append(set(p))
    masks=[mask(B) for B in blocks]
    for e in edges:
        touched=[i for i,B in enumerate(masks) if e&B]
        need(bool(touched),'witness-ground-set')
        if len(touched)>1:
            need(not intersection([P[i] for i in touched]),'cross-palette-common-color')

def complete_graph_lift(blocks: Sequence[Sequence[int]],local_edges: Sequence[Sequence[int]],
                        macro_edges: Sequence[tuple[int,int]]) -> tuple[int,...]:
    result=[lift(e,B) for B,es in zip(blocks,local_edges) for e in es]
    for i,j in macro_edges:
        result.extend((1<<u)|(1<<v) for u in blocks[i] for v in blocks[j])
    return minimal(result)

def scalar_certificate(n: int,edges: Sequence[int],a: Sequence[F],kappa: F) -> None:
    need(len(a)==n and kappa>=1 and all(0<=x<1 for x in a),'scalar-domain')
    good=good_table(n,edges)
    for u,isgood in enumerate(good):
        w=sum((a[i] for i in range(n) if u>>i&1),F(0))
        need(w>=1 or isgood,'scalar-safe-direction')
        need(not isgood or w<kappa,'scalar-strict-upper')

def critical_certificate(n: int,edges: Sequence[int],w: Sequence[F],
                         bad_weights: dict[int,F],good_weights: dict[int,F],value: F) -> None:
    good=good_table(n,edges)
    need(len(w)==n and all(x>=0 for x in w),'critical-primal-nonnegative')
    need(all(price_sum(w,e)>=1 for e in edges),'critical-primal-witness')
    maximum=max(price_sum(w,u) for u,yes in enumerate(good) if yes)
    need(maximum==value,'critical-primal-objective')
    need(all(x>=0 and not good[e] for e,x in bad_weights.items()),'dual-bad-support')
    need(all(x>=0 and good[u] for u,x in good_weights.items()),'dual-good-support')
    need(sum(good_weights.values(),F(0))==1,'dual-good-distribution')
    for i in range(n):
        left=sum((x for e,x in bad_weights.items() if e>>i&1),F(0))
        right=sum((x for u,x in good_weights.items() if u>>i&1),F(0))
        need(left<=right,'dual-coordinate-marginal')
    need(sum(bad_weights.values(),F(0))==value,'dual-objective-match')

def price_sum(w: Sequence[F], u: int) -> F:
    return sum((v for i,v in enumerate(w) if u>>i&1),F(0))

def strict_translate(n: int,edges: Sequence[int],w: Sequence[F]) -> tuple[list[F],F,int]:
    good=good_table(n,edges)
    need(all(x>=0 for x in w) and all(price_sum(w,e)>=1 for e in edges),'feasible-critical-upper')
    beta=max(price_sum(w,u) for u,yes in enumerate(good) if yes)
    if beta<1:
        a=list(w);kappa=F(1);K=408
    else:
        K=(408*beta).numerator//(408*beta).denominator+1
        kappa=F(K,408)
        theta=(kappa-beta)/(n+1)
        a=[(1-theta)*min(F(1),x)+theta/2 for x in w]
    scalar_certificate(n,edges,a,kappa)
    return a,kappa,K

def matmul(A: list[list[F]],B: list[list[F]]) -> list[list[F]]:
    return [[sum((A[i][k]*B[k][j] for k in range(2)),F(0)) for j in range(2)] for i in range(2)]

def matpow(A: list[list[F]], n: int) -> list[list[F]]:
    R=[[F(1),F(0)],[F(0),F(1)]]
    while n:
        if n&1:R=matmul(R,A)
        A=matmul(A,A);n//=2
    return R

def binomial(n: int,k: int,p: F) -> F:
    return F(comb(n,k))*p**k*(1-p)**(n-k)

def cycle_count_good(counts: Sequence[int],s: int) -> bool:
    return all(c<2*s for c in counts) and all(not(counts[i]>=s and counts[(i+1)%len(counts)]>=s) for i in range(len(counts)))

def cycle_probability_by_counts(s: int,m: int,N: int,p: F) -> F:
    probs=[binomial(N,k,p) for k in range(N+1)]
    answer=F(0)
    for c in product(range(N+1),repeat=2*m):
        if cycle_count_good(c,s):
            v=F(1)
            for k in c:v*=probs[k]
            answer+=v
    return answer

def cycle_probability_by_matrix(s: int,m: int,N: int,p: F) -> F:
    low=sum((binomial(N,k,p) for k in range(min(s,N+1))),F(0))
    high=sum((binomial(N,k,p) for k in range(s,min(2*s,N+1))),F(0))
    R=matpow([[low,high],[low,F(0)]],2*m)
    return R[0][0]+R[1][1]
