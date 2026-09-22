#!/usr/bin/env python3
"""Bounded H3 search -> exact scalar certification -> traceable candidate.

No network calls or scientific-register writes. Local trusted cache only;
--cold or --verify-result recomputes instead of trusting cached arithmetic.
Code/runtime/source/context identities are part of the key. This is NOT a
hermetic Nix build, authenticated cache, full-law falsifier, or proof assistant.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import zipfile
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from h3_scalar_certificate import prove, digest, canonical, need
from h3_scalar_arithmetic import I, BITS
import h3_source_admission as admission

ARCHIVE_SHA='73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21'
MANIFEST_SHA='81770e26ec654d3b462f06cebaa20d0dd282cfd4df10ac593c7887ea0d292b40'
FORMULA_SPEC=[
 ('parameters','exact epsilon,d,R,target',[]),
 ('kernel','normalized image-sum Hermite moments with full geometric tails',[]),
 ('energy','shifted Gram blocks + L2 transport -> uniform Q', ['kernel','parameters']),
 ('axial','1-2*sf(epsilon/(R*sqrt(m8)/12)-sqrt(Q))',['kernel','energy','parameters']),
 ('midpoint','Gaussian positive-part M1,M2 with mean/variance intervals',['kernel','parameters']),
 ('difference','1-2*sf((2*d-m2*R^3/6)/(R*sqrt((m4-m2^2)*m2)))',['kernel','parameters']),
 ('determinant','positive a^2*M2-a*R*(m4*m2/4)*M1; a=1-epsilon',['kernel','parameters','midpoint']),
 ('coefficient','pA*pD*positive_bracket',['axial','difference','determinant']),
]


def strict_json(text):
    def pairs(items):
        out={}
        for key,value in items:
            need(key not in out,'duplicate JSON key')
            out[key]=value
        return out
    def reject(value):raise ValueError('nonfinite JSON token')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=reject)


def file_hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


source_check = admission.source_check

def recipe(source,context,params):
    admission.validate_context(context)
    need(context['source_archive_sha256']==source.get('archive_sha256'),'context/source identity mismatch')
    return {'schema':'local-h3-recipe-v2','source':source,
            'code':{name:file_hash(HERE/name) for name in ('h3_scalar_arithmetic.py','h3_scalar_certificate.py','h3_solver.py','h3_source_admission.py')},
            'runtime':{'python':sys.version,'implementation':platform.python_implementation(),
                       'platform':platform.platform(),'bits':BITS},
            'context_sha256':digest(context),'parameters':params,
            'hermetic':False,'context_authentication':'caller-supplied deny-only filter; source admission is separate'}


def formula_lineage(result,source):
    v=result['values']; values={
      'parameters':{'parameters':result['parameters'],'conclusion':result['conclusion']},'kernel':{'moments':v['moments'],'tail':v['tail']},
      'energy':v['energy'],'axial':{'argument':v['axial_argument'],'p':v['p_axial']},
      'midpoint':{k:v[k] for k in ('midpoint_mean','midpoint_variance','positive_first_moment','positive_second_moment')},
      'difference':{'argument':v['difference_argument'],'p':v['p_difference']},
      'determinant':{'beta':v['beta'],'positive_bracket':v['positive_bracket']},
      'coefficient':v['derived_lower_coefficient']}
    nodes={}
    for nid,expr,deps in FORMULA_SPEC:
        payload={'id':nid,'expression':expr,'dependencies':{d:nodes[d]['hash'] for d in deps},
                 'value':values[nid],'source_archive_sha256':source['archive_sha256']}
        nodes[nid]={'hash':digest(payload),'payload':payload}
    return {'schema':'h3-formula-lineage-v1','root':nodes['coefficient']['hash'],'nodes':nodes,
            'spreadsheet_cells_used':[],'lineage_completeness':'declared formula-level dependencies; not a general Python or Sheets tracer',
            'scope':'Declared mathematical formula DAG, not whole-workspace or Google Sheets formula lineage',
            'hash_meaning':'content/dependency integrity only; not a mathematical proof'}


def affected_nodes(changed):
    known={x[0] for x in FORMULA_SPEC}
    need(set(changed)<=known,'unknown formula node')
    out=set(changed)
    for nid,_,deps in FORMULA_SPEC:
        if out.intersection(deps):out.add(nid)
    return sorted(out)


def classify_enclosure(lo,hi,lower,upper,quantity_kind):
    """For a certified enclosure of THE TARGET only, never a lower estimator."""
    lo,hi,lower,upper=map(F,(lo,hi,lower,upper))
    need(lo<=hi and lower<=upper,'malformed enclosure or target')
    if quantity_kind!='TARGET_QUANTITY_ENCLOSURE':return 'WRONG_QUANTITY_FOR_REFUTATION'
    if hi<lower or lo>upper:return 'CONTRADICTION_REQUIRES_SCOPE_AND_INDEPENDENT_REVIEW'
    if lower<=lo and hi<=upper:return 'COMPATIBLE_ON_THIS_ENCLOSED_DOMAIN_ONLY'
    return 'INCONCLUSIVE_REFINE_ENCLOSURE'


def search():
    """Finite double-precision proposal search; never decides truth."""
    import math
    R=.05; Q=1266466/160083; s=R*math.sqrt(105)/12
    best=(-1,None); evaluated=0; rejected=0
    for ei in range(100,301):
        eps=ei/1000; pa=1-math.erfc((eps/s-math.sqrt(Q))/math.sqrt(2))
        for di in range(50,201):
            evaluated+=1; d=di/1000
            pd=1-math.erfc((2*d-R**3/6)/(R*math.sqrt(2))/math.sqrt(2))
            if pa<=0 or pd<=0: rejected+=1;continue
            mu=1.2-R**3/12-d; sd=math.sqrt(2*(1-R*R/4)); t=mu/sd
            ph=math.exp(-t*t/2)/math.sqrt(2*math.pi); c=.5*(1+math.erf(t/math.sqrt(2)))
            M1=sd*ph+mu*c; M2=(sd*sd+mu*mu)*c+mu*sd*ph
            b=pa*pd*((1-eps)**2*M2-(1-eps)*R*.75*M1)
            if b>best[0]:best=(b,(ei,di))
    return {'status':'NONCERTIFYING_PARAMETER_SEARCH','evaluated':evaluated,'rejected_nonpositive_probability':rejected,
            'best_diagnostic_estimate':best[0],'epsilon':str(F(best[1][0],1000)),
            'gap_half':str(F(best[1][1],1000)),
            'limitation':'Optimizes a Gaussian-limit sufficient formula, not actual Z or a rigorous global optimum. Re-certify exact selected parameters.'}


def cached(recipe_obj,cache,compute,cold=False):
    key=digest(recipe_obj)
    if cache is None:return compute(),'DISABLED',key
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    path=cache/(key+'.json')
    if path.exists():
        old=strict_json(path.read_text())
        need(old.get('recipe')==recipe_obj and old.get('payload_sha256')==digest(old.get('payload')),'cache corruption/key mismatch')
        if not cold:return old['payload'],'LOCAL_TRUSTED_HIT_NOT_NEW_EVIDENCE',key
        fresh=compute()
        need(fresh==old['payload'],'cold recomputation disagrees with cache')
        return fresh,'COLD_RECOMPUTE_MATCH',key
    result=compute(); envelope={'recipe':recipe_obj,'payload':result,'payload_sha256':digest(result)}
    try:
        with path.open('x') as f:f.write(json.dumps(envelope,sort_keys=True,indent=2)+'\n')
    except FileExistsError:
        other=strict_json(path.read_text());need(other==envelope,'conflicting cache publication')
    return result,'MISS_COMPUTED',key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--search',action='store_true')
    p.add_argument('--verify-result',type=Path)
    p.add_argument('--expected-certificate-sha')
    p.add_argument('--epsilon',type=F)
    p.add_argument('--gap-half',type=F)
    p.add_argument('--target',type=F,default=F(1747,1000))
    args=p.parse_args()
    start=time.perf_counter();cpu=time.process_time()
    context=admission.strict_json(admission.read_bounded(args.context,admission.MAX_METADATA))
    source=source_check(args.source_archive,context)
    diagnostic=search() if args.search else None
    epsilon=args.epsilon if args.epsilon is not None else F(diagnostic['epsilon']) if diagnostic else F(103,500)
    gap_half=args.gap_half if args.gap_half is not None else F(diagnostic['gap_half']) if diagnostic else F(9,100)
    params={'epsilon':str(epsilon),'gap_half':str(gap_half),'target':str(args.target),'radius':'1/20'}
    rec=recipe(source,context,params)
    result,cache_state,key=cached(rec,args.cache,
        lambda:prove(epsilon=epsilon,gap_half=gap_half,target=args.target),args.cold or args.verify_result is not None or args.expected_certificate_sha is not None)
    if args.expected_certificate_sha:
        need(digest(result)==args.expected_certificate_sha,'exact candidate digest mismatch')
    if args.verify_result:
        expected=strict_json(args.verify_result.read_text())
        need(expected.get('certificate',expected)==result,'candidate differs from exact recomputation')
    final_context=admission.strict_json(admission.read_bounded(args.context,admission.MAX_METADATA))
    need(source_check(args.source_archive,final_context)==source and recipe(source,final_context,params)==rec,
         'source/context/code changed during run')
    output={'certificate':result,'formula_lineage':formula_lineage(result,source),'recipe':rec,'recipe_sha256':key,
            'cache':cache_state,'parameter_search':diagnostic,
            'recorded_at':datetime.now(timezone.utc).isoformat(),
            'timing':{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                      'energy_joules':None,'energy_measurement':'NOT_MEASURED'},
            'arb_backend':'NOT_EXECUTED; no second rigorous backend claimed',
            'source_admissibility':'pinned authored provenance and current local exclusions/upstream eligibility rechecked; caller context is deny-only; no live permission or canonical acceptance service',
            'automatic_promotion':False,'scheduled_execution':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:f.write(json.dumps(output,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'result':'CANDIDATE_WRITTEN','cache':cache_state,'coefficient':result['conclusion']['coefficient'],
                     'formula_bound':result['values']['derived_outward_decimal'],'seconds':output['timing']['wall_seconds']}))


if __name__=='__main__':
    try:main()
    except (ValueError,TypeError,KeyError,OSError,zipfile.BadZipFile) as e:
        print('PILOT REFUSAL: '+str(e),file=sys.stderr);sys.exit(2)
