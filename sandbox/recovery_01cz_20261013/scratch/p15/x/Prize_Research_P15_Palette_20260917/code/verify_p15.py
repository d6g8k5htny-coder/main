#!/usr/bin/env python3
"""Finite exact companions to P15. No independent review or all-size proof by testing."""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, random, sys
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
from exact_core import *
from certify_palette_plan import validate

COUNTS={}
def ck(cond,label,group):
    need(cond,label);COUNTS[group]=COUNTS.get(group,0)+1

def subsets(n):return range(1<<n)
def project(u,B):return sum(1<<i for i,v in enumerate(B) if u>>v&1)
def obstruct_gens(good,k):
    parts=part_table(good)
    return minimal(u for u,p in enumerate(parts) if hard(p,k))

def palette_example():
    blocks=[[0,1],[2,3],[4,5]]
    edges=[mask(B) for B in blocks]+[mask(e) for e in product(*blocks)]
    return blocks,tuple(edges),[[0,1],[1,2],[0,2]]

def plan_example():
    blocks,edges,pals=palette_example()
    raw=[[v for v in range(6) if e>>v&1] for e in edges]
    bind=hashlib.sha256(json.dumps(sorted([sorted(e) for e in raw]),separators=(',',':')).encode()).hexdigest()
    return {'n':6,'K':3,'minimal_witnesses':raw,'source_witness_sha256':bind,'blocks':[{'vertices':B,'palette':pal,'local_certificate':{'kind':'proper_coloring','colors':[0,1]}} for B,pal in zip(blocks,pals)]}

def test_palette():
    blocks,edges,P=palette_example()
    validate_palettes(6,edges,blocks,P,[2]*3,3)
    g=good_table(6,edges);pt=part_table(g)
    ck(pt[-1]==3,'overlapping-palette-chromatic-exact','palette')
    ck(all(set(P[i])&set(P[j]) for i,j in combinations(range(3),2)) and not intersection([set(p) for p in P]),'pairwise-vs-total-intersection','palette')
    # Exhaustive original palettes and amalgamation on the three singleton blocks.
    allowed=[{i for i in range(3) if m>>i&1} for m in range(1,8)]
    for pals in product(allowed,repeat=3):
        ps=not intersection(list(pals))
        automatic=all(len(set(col))>=2 for col in product(*[sorted(x) for x in pals]))
        ck(ps==automatic,'uniform-amalgamation-iff','palette')
    rng=random.Random(151507)
    cases=0
    for n in range(3,9):
        for rep in range(24):
            b=2+(rep%min(3,n-1));blocks=[[] for _ in range(b)]
            for i in range(n):blocks[i%b].append(i)
            K=2+(rep%4)
            pals=[]
            for B in blocks:
                p=sorted(rng.sample(range(K),rng.randrange(1,K+1)));pals.append(p)
            masks=[mask(B) for B in blocks]
            es=[]
            for e in range(1,1<<n):
                if e.bit_count()>4 or rng.random()>.11:continue
                touched=[i for i,B in enumerate(masks) if e&B]
                if len(touched)==1 or not intersection([set(pals[i]) for i in touched]):es.append(e)
            es=minimal(es);demands=[len(p) for p in pals]
            validate_palettes(n,es,blocks,pals,demands,K)
            goods=[good_table(len(B),localize(es,B)) for B in blocks]
            localgens=[obstruct_gens(gt,k) for gt,k in zip(goods,demands)]
            G=minimal(lift(h,B) for hs,B in zip(localgens,blocks) for h in hs)
            gp=part_table(good_table(n,es))
            for u in range(1<<n):ck(not hard(gp[u],K) or covered(u,G),'palette-cover-complete','palette')
            p=[F(rng.randrange(0,5),4) for _ in range(n)]
            globalprob=prob(p,good_table(n,es));joint=F(1)
            for B,gt in zip(blocks,goods):joint*=prob([p[i] for i in B],gt)
            ck(globalprob<=joint,'actual-local-probability-direction','palette')
            z=[F(rng.randrange(0,6),5) for _ in range(n)]
            ck(cost(z,G)<=sum((cost([z[i] for i in B],hs) for B,hs in zip(blocks,localgens)),F(0)),'union-cover-cost','palette')
            cases+=1
    return {'random_exact_instances':cases,'uniform_palette_assignments':343}

def test_critical():
    examples=0
    for m in range(2,5):
        n=2*m;edges=[(1<<i)|(1<<j) for i in range(m) for j in range(m,2*m)]
        w=[F(1,2)]*n;bad={(1<<i)|(1<<(m+i)):F(1,2) for i in range(m)}
        good={mask(range(m)):F(1,2),mask(range(m,n)):F(1,2)}
        critical_certificate(n,edges,w,bad,good,F(m,2));ck(True,'bipartite-exact-dual','critical')
        a,kappa,K=strict_translate(n,edges,w)
        ck(K==max(408,204*m+1),'strict-integer-boundary','critical');examples+=1
    for n in range(2,8):
        for r in range(2,n+1):
            edges=[mask(e) for e in combinations(range(n),r)];val=F(r-1,r)
            bad={e:val/F(len(edges)) for e in edges}
            gs=[mask(e) for e in combinations(range(n),r-1)]
            nu={u:F(1,len(gs)) for u in gs}
            critical_certificate(n,edges,[F(1,r)]*n,bad,nu,val)
            ck(True,'uniform-critical-dual','critical')
            a,kappa,K=strict_translate(n,edges,[F(1,r)]*n)
            ck(kappa==1 and K==408,'subunit-critical','critical');examples+=1
    # alpha=1: balanced four-coordinate winning/losing pair certificate.
    es=[mask(e) for e in [(0,2),(0,3),(1,2),(1,3)]]
    nu={mask([0,1]):F(1,2),mask([2,3]):F(1,2)}
    mu={mask([0,2]):F(1,2),mask([1,3]):F(1,2)}
    critical_certificate(4,es,[F(1,2)]*4,mu,nu,F(1));ck(True,'alpha-one-endpoint','critical')
    rng=random.Random(1515)
    for _ in range(140):
        n=rng.randrange(3,9)
        es=minimal(mask(rng.sample(range(n),rng.randrange(2,min(n,4)+1))) for j in range(rng.randrange(1,10)))
        proposal=[F(rng.randrange(1,10),5) for i in range(n)]
        gamma=min(price_sum(proposal,e) for e in es);w=[x/gamma for x in proposal]
        a,kappa,K=strict_translate(n,es,w)
        ck(all(x<1 for x in a) and K==math.ceil(408*kappa),'critical-directed-translation','critical')
        for u,yes in enumerate(good_table(n,es)):
            ck((price_sum(a,u)>=1 or yes) and (not yes or price_sum(a,u)<kappa),'all-subsets-strict-sandwich','critical')
    return {'exact_matching_duals':examples+1,'rational_upper_translations':140}

def macro_obstruction(n,edges,k):return obstruct_gens(good_table(n,edges),k)

def test_macro():
    rng=random.Random(15256);cases=0
    for b in (2,3,4):
        for _ in range(20):
            sizes=[rng.randrange(1,3) for _ in range(b)];blocks=[];n=0
            for s in sizes:blocks.append(list(range(n,n+s)));n+=s
            local=[]
            for B in blocks:
                local.append(tuple([3] if len(B)==2 and rng.random()<.6 else []))
            mac=[(i,j) for i,j in combinations(range(b),2) if rng.random()<.55]
            edges=complete_graph_lift(blocks,local,mac)
            gt=good_table(n,edges);parts=part_table(gt)
            p=[F(rng.randrange(0,5),4) for i in range(n)]
            qs=[1-math.prod(1-p[v] for v in B) for B in blocks]
            mg=good_table(b,[mask(e) for e in mac]);QG=1-prob(qs,mg);Q=1-prob(p,gt)
            ck(QG<=Q,'macro-probability-domination','macro')
            for K0,L in product((1,2,3),(1,2)):
                GG=macro_obstruction(b,[mask(e) for e in mac],K0)
                gens=[]
                for e in GG:
                    selectedblocks=[i for i in range(b) if e>>i&1]
                    gens.extend(mask(vs) for vs in product(*[blocks[i] for i in selectedblocks]))
                for B,es in zip(blocks,local):gens.extend(lift(g,B) for g in obstruct_gens(good_table(len(B),es),L))
                gens=minimal(gens)
                for u,v in enumerate(parts):ck(not hard(v,K0*L) or covered(u,gens),'macro-product-cover','macro')
            # Occupancy-price factorization including zero-cost original singletons.
            z=[F(rng.randrange(0,5),4) for i in range(n)]
            occgens=[([1<<v for v in B] if sum(z[v] for v in B)<=1 else [0]) for B in blocks]
            cc=[cost(z,gs) for gs in occgens]
            for i,j in mac:
                descriptions=sum((price(z,a|b) for a in occgens[i] for b in occgens[j]),F(0))
                ck(descriptions==cc[i]*cc[j],'cartesian-price-identity','macro')
                for u in range(1<<n):
                    if u&mask(blocks[i]) and u&mask(blocks[j]):
                        ck(covered(u,[a|b for a in occgens[i] for b in occgens[j]]),'occupancy-lift-complete','macro')
            cases+=1
    # Two color refinement checks all assignments, preserving true edge certificates.
    for n in range(2,9):
        es=[(i,j) for i,j in combinations(range(n),2) if (i+j)%3]
        z=[F(i+1,n+2) for i in range(n)]
        old=sum((z[i]*z[j] for i,j in es),F(0));tot=F(0)
        for colors in product((0,1),repeat=n):
            retained=[(i,j) for i,j in es if colors[i]==colors[j]]
            tot+=sum((z[i]*z[j] for i,j in retained),F(0))
        ck(tot/F(2**n)==old/2,'binary-edge-refinement-exact','macro')
    ck(F(1,2)+F(85,192)==F(181,192)<1,'macro-cost-slack','macro')
    ck(256*408==104448,'macro-palette','macro')
    return {'complete_lift_instances':cases,'cover_demands_per_instance':6,'exhaustive_refinement_instances':7}

def cycle_edges(s,m,N):
    blocks=[list(range(i*N,(i+1)*N)) for i in range(2*m)];es=[]
    for B in blocks:es.extend(mask(e) for e in combinations(B,2*s))
    for i in range(2*m):
        es.extend(mask(a+b) for a in combinations(blocks[i],s) for b in combinations(blocks[(i+1)%(2*m)],s))
    return blocks,minimal(es)

def test_cycle():
    probs=0
    for s,m,N in [(2,2,4),(2,2,5),(2,3,4),(3,2,6)]:
        M=m*(3*s-2);best=0
        for c in product(range(2*s),repeat=2*m):
            if cycle_count_good(c,s):best=max(best,sum(c))
        ck(best==M,'cycle-extremal-cardinality','cycle')
        alpha=F(M,2*s)
        ck(alpha*F(2*s,2*m*N)==F(3*s-2,2*N),'cycle-dual-marginal','cycle')
        for p in (F(0),F(1,7),F(1,3),F(1)):
            x=cycle_probability_by_counts(s,m,N,p);y=cycle_probability_by_matrix(s,m,N,p)
            ck(x==y,'cycle-transfer-vs-count-enumeration','cycle');probs+=1
    # Direct enumeration in ORIGINAL vertex coordinates, rather than block count vectors.
    s,m,N=2,2,4;blocks,es=cycle_edges(s,m,N);n=2*m*N
    counts_by_size=[0]*(n+1)
    for u in range(1<<n):
        original=not any(u&e==e for e in es)
        via_counts=cycle_count_good([(u&mask(B)).bit_count() for B in blocks],s)
        ck(original==via_counts,'cycle-original-vs-block-event','cycle')
        if original:counts_by_size[u.bit_count()]+=1
    for p in (F(1,5),F(2,5),F(0),F(1)):
        exact=sum((F(c)*p**k*(1-p)**(n-k) for k,c in enumerate(counts_by_size)),F(0))
        ck(exact==cycle_probability_by_matrix(s,m,N,p),'cycle-original-probability','cycle')
    for s in range(2,81):
        b=2**(s+1)
        bound=F(b,100**(2*s))*(F(1,math.factorial(2*s))+F(1,math.factorial(s)**2))
        ck(bound<F(1,4),'cycle-failure-upper','cycle')
        if s>=14:ck(b//100>10*(2*s-1),'cycle-empty-mass-fails','cycle')
    for t in range(2,151):ck(F(t-1,t)**t<F(1,2),'integer-empty-block-decay','cycle')
    s=14;m=2**s;b=2*m;r=2*s;N=816*(r-1)+1
    ck((b*N,r,N)==(721977344,28,22033),'implicit-instance-size','cycle')
    ck(F(m*(3*s-2),r)==F(163840,7),'implicit-instance-critical-value','cycle')
    ck(math.ceil(F(N,r-1))==817,'nonempty-816-obstruction','cycle')
    return {'count_vector_probability_tests':probs,'original_coordinate_outcomes':1<<16,'symbolic_rank_rows':79,'implicit_large_vertices_NOT_ENUMERATED':b*N}

FAULTS=['shared_blocks','cross_common_color','local_demand','witness_omission','strict_endpoint','invalid_dual','unsafe_clipping','drop_zero_generator','occupancy_is_probability','incomplete_macro','local_macro_independence','reverse_probability','empty_generator','float_certificate']
def run_fault(name):
    if name=='shared_blocks':
        p=plan_example();p['blocks'][1]['vertices']=[0,3];validate(p)
    elif name=='cross_common_color':
        p=plan_example();p['blocks'][2]['palette']=[0,1];validate(p)
    elif name=='local_demand':
        p=plan_example();p['blocks'][0]['palette']=[0];validate(p)
    elif name=='witness_omission':
        p=plan_example();p['minimal_witnesses'].pop();validate(p)
    elif name=='strict_endpoint':
        es=[mask((i,j)) for i in (0,1) for j in (2,3)]
        scalar_certificate(4,es,[F(1,2)]*4,F(1))
    elif name=='invalid_dual':
        critical_certificate(2,[3],[F(1,2)]*2,{3:F(1)},{1:F(1,2),2:F(1,2)},F(1,2))
    elif name=='unsafe_clipping':scalar_certificate(2,[3],[F(9,10),F(0)],F(2))
    elif name=='drop_zero_generator':
        need(covered(3,[]) or cost([F(0),F(1)], [3])!=0,'zero-cost-generator-required')
    elif name=='occupancy_is_probability':
        q=1-F(1,2)**2;c=min(F(1),F(1,2)+F(1,2));need(c<=q,'occupancy-cost-is-not-probability')
    elif name=='incomplete_macro':
        p=[F(1,100),F(1),F(1,100),F(1)];Q=1-prob(p,good_table(4,[mask([0,2])]))
        need(F(1)<=Q,'incomplete-macro-probability')
    elif name=='local_macro_independence':
        p=[F(1,2)]*3
        local=good_table(3,[3]);macro=good_table(3,[5,6]);both=[a and b for a,b in zip(local,macro)]
        need(prob(p,both)==prob(p,local)*prob(p,macro),'dependent-local-macro-events')
    elif name=='reverse_probability':
        need(prob([F(1,2)]*2,good_table(2,[3]))>=1,'global-local-probability-reversed')
    elif name=='empty_generator':need(cost([F(1,2)], [0])==0,'empty-generator-cost-one')
    elif name=='float_certificate':rational(0.5)
    else:raise ValueError('unknown test')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fault',choices=FAULTS+['survivor']);args=ap.parse_args()
    if args.fault:
        try:
            if args.fault=='survivor':validate(plan_example())
            else:run_fault(args.fault)
        except GuardFailure as e:
            print(json.dumps({'status':'REJECT','variant':args.fault,'reason':str(e)},sort_keys=True));return 1
        print(json.dumps({'status':'UNDETECTED','variant':args.fault},sort_keys=True));return 0
    instances={}
    instances['palette']=test_palette();instances['critical']=test_critical();instances['macro']=test_macro();instances['cycle']=test_cycle()
    plan=plan_example();receipt=validate(plan);ck(receipt['status']=='PASS','cli-valid-plan','cli')
    for file in Path(__file__).parent.glob('*.py'):
        ck(not any(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(file.read_text()))),'no-removable-asserts','implementation')
    print(json.dumps({'status':'PASS','checks':sum(COUNTS.values()),'groups':COUNTS,'instances':instances,'external_review_credit':0,'claim_grade':'finite exact computational companions; all-size arguments are written separately'},sort_keys=True));return 0
if __name__=='__main__':sys.exit(main())
