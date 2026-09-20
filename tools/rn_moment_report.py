#!/usr/bin/env python3
"""Reproduce a finite synthetic reuse experiment and an unreviewed RN candidate.

No field certification, canonical utility promotion, independence credit or
scientific status change is established. Costs are a declared abstract model.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.operations.rn_applicability import Context, Law, RN5_ID, RN5_SHA, applicable, apply_moment_bound
from research.interval import Interval
from research.rn.gaussian_moments import Cost, MomentEngine, coefficient_cap, evaluate, second_moment_trace

PLAN = ROOT / 'research/rn/candidates/reuse_plan_20260920.json'
SOURCE = ROOT / 'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md'


def fixture(case):
    m=tuple(F((case+3*i)%7-3) for i in range(3))
    b=tuple(F((2*case+i)%5-2,5) for i in range(3))
    lower=tuple(tuple(F((case+2*i+j)%5-2,10)+(1 if i==j else 0) if j<=i else F(0) for j in range(3)) for i in range(3))
    s=tuple(tuple(sum(lower[i][k]*lower[j][k] for k in range(3)) for j in range(3)) for i in range(3))
    ctx=Context(RN5_ID,RN5_SHA,2,('xx','yy','xy'),'synthetic rational Hessian units',f'preselected synthetic law {case}',(F(-1),F(1)))
    return Law(ctx,m,b,s,ctx.conditioned_law,ctx.conditioned_law)


def numerical_identity(law):
    """Exclude labels and source IDs; only mathematical inputs define overlap."""
    value=(law.intercept,law.slope,law.covariance,law.context.domain,law.context.order)
    def exact(x):
        if isinstance(x,(list,tuple)):return [exact(v) for v in x]
        return str(F(x)) if type(x) in (int,F) else x
    return hashlib.sha256(json.dumps(exact(value),separators=(',',':')).encode()).hexdigest()


def numerical_overlaps(development,evaluation):
    known={numerical_identity(fixture(case)):case for case in development}
    return [{'evaluation_case':case,'development_case':known[identity],'numerical_sha256':identity}
            for case in evaluation if (identity:=numerical_identity(fixture(case))) in known]


def compare(law, marks, budget):
    costs={arm:Cost(budget) for arm in ('recompute','reuse')}
    values={arm:[] for arm in costs}
    for arm,c in costs.items():
        # Source hash cost, parsing, case generation and reporting are excluded
        # equally. The lookup/comparison is charged, never mistaken for hashing.
        c.charge()
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==RN5_SHA
        engine=None
        if arm=='reuse':
            applicable(law,law.context,cost=c)
            engine=MomentEngine(law.intercept,law.slope,law.covariance,cost=c)
        for degree in (2,4):
            if arm=='reuse':
                p=engine.determinant(degree)
                c.charge()  # retained polynomial insertion
            for t in marks:
                applicable(law,law.context,cost=c)
                if arm=='recompute':
                    mean=tuple(c.add(a,c.mul(b,t)) for a,b in zip(law.intercept,law.slope))
                    p=MomentEngine(mean,(0,0,0),law.covariance,cost=c).determinant(degree)
                else:
                    c.charge()  # retained polynomial lookup
                values[arm].append(evaluate(p,t,c))
    if values['recompute']!=values['reuse']:
        raise ValueError('reuse changes exact values')
    # Independent second-moment check outside both arms; charged equally below.
    for t,got in zip(marks,values['reuse'][:len(marks)]):
        mean=tuple(a+b*t for a,b in zip(law.intercept,law.slope))
        if got!=second_moment_trace(mean,law.covariance):
            raise ValueError('trace cross-check fails')
    for c in costs.values():
        c.charge(len(values['reuse']))  # exact result comparisons, same both arms
    return {'law_sha256':law.fingerprint(),'costs':{k:v.used for k,v in costs.items()},
            'budget_per_arm':budget,'equal_values':True,'values':[str(x) for x in values['reuse']]}


def report():
    plan=json.loads(PLAN.read_text())
    assert plan['development_case_ids']==list(range(4))
    assert plan['held_out_case_ids']==list(range(100,116))
    assert set(plan['development_case_ids']).isdisjoint(plan['held_out_case_ids'])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==RN5_SHA
    cases=[{'case':case,**compare(fixture(case),[F(x) for x in plan['query_marks']],plan['budget_per_arm_per_case'])}
           for case in plan['held_out_case_ids']]
    overlaps=numerical_overlaps(plan['development_case_ids'],plan['held_out_case_ids'])
    # Separate explicit development witness; never presented as held-out.
    law=fixture(0)
    v=F(1,1000)
    law=Law(law.context,(-1,-1,0),(1,-1,0),tuple(tuple(v if i==j else F(0) for j in range(3)) for i in range(3)),law.mean_law,law.covariance_law)
    bound=apply_moment_bound(law,law.context,degree=4,source_bytes=SOURCE.read_bytes())
    p=bound['polynomial']
    return {'schema_version':1,'status':'UNREVIEWED_ALGEBRA_CANDIDATE','source_sha256':RN5_SHA,
            'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            'cost_model':'Counted rational add/multiply/divide calls, memo/cache lookup/insertion, covariance symmetry/PSD checks at instrumented points, applicability comparisons, source digest comparison, equal-result comparisons. Integer loop/index arithmetic, rational sign changes, canonicalization, bit complexity, SHA hashing, input construction, output serialization, and the external trace-oracle arithmetic are excluded equally. No wall-time or net research cost claim.',
            'plan_clarification':'The pre-implementation plan omitted division/PSD checks from its short unit description. They are conservatively charged in both arms; this instrumented model was fixed before generating the evaluation cases. No case or result was changed to favor reuse. The plan called these held-out, but the period-35 generator duplicates four development inputs; that split label is rejected and the measured sample is retained as a preselected synthetic evaluation.',
            'evaluation_split':'PRESELECTED_SYNTHETIC_WITH_DISCLOSED_DEVELOPMENT_OVERLAP',
            'held_out_claim_valid':False,'numerical_development_overlaps':overlaps,
            'evaluation_cases':cases,
            'totals':{arm:sum(r['costs'][arm] for r in cases) for arm in ('recompute','reuse')},
            'development_bound':{'law_sha256':law.fingerprint(),'covariance_variance':'1/1000',
                 'domain':['-1','1'],'moment_degree':4,'polynomial':[str(x) for x in p],
                 'rn5_coefficient_cap':str(coefficient_cap(p,Interval(-1,1))),
                 'bernstein_combined_cap':str(bound['upper']),
                 'strictly_improved':bound['upper']<coefficient_cap(p,Interval(-1,1))},
            'field_certified':False,'independence_credit':0,'canonical_utility':'UNMEASURED',
            'original_prize_closed':False,
            'does_not_establish':'No identified RN field covariance, interval covariance family, spatial cover, weighted Palm transfer, novelty, general usefulness, independent review, theorem promotion or status change.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write',type=Path)
    group.add_argument('--check',type=Path)
    args=parser.parse_args()
    result=report()
    if args.check:
        if json.loads(args.check.read_text())!=result:
            raise SystemExit('RN candidate mismatch')
    else:
        with args.write.open('x') as f:
            json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print('RN candidate: 16 synthetic cases (4 development overlaps), exact equality; totals '+str(result['totals'])+'; no held-out or status claim')


if __name__=='__main__':main()
