"""Descriptive guarded profile fit; no continuum or statistical certificate."""
import datetime
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

PROTOCOL_SHA = '4932ec5e35c89b282f98301f6cd0cc85b0c65ea0e23528bb9e9dde2a5b0be8aa'

def power_integral(a, b, alpha):
    if not all(math.isfinite(x) for x in (a,b,alpha)) or not 0<a<b:
        raise ValueError('finite positive ordered bin required')
    p=1-alpha
    return math.log(b/a) if p==0 else a**p*math.expm1(p*math.log(b/a))/p

def fit(edges, counts, exposure):
    if (not math.isfinite(exposure) or exposure<=0 or len(edges)<4
            or len(counts)!=len(edges)-1
            or any(not math.isfinite(x) or x<0 for x in counts)
            or any(not math.isfinite(x) or x<=0 for x in edges)
            or any(a>=b for a,b in zip(edges,edges[1:]))):
        raise ValueError('invalid edges, counts or exposure')
    widths=[b-a for a,b in zip(edges,edges[1:])]
    z=[exposure*w for w in widths]
    weights=[1/(c+1) for c in counts]
    szz=math.fsum(w*v*v for w,v in zip(weights,z))
    szc=math.fsum(w*v*c for w,v,c in zip(weights,z,counts))
    def profile(alpha):
        if alpha==0:
            A=0.0; B=szc/szz; relative_det=0.0
            x=z
        else:
            x=[exposure*power_integral(a,b,alpha) for a,b in zip(edges,edges[1:])]
            sxx=math.fsum(w*v*v for w,v in zip(weights,x))
            sxz=math.fsum(w*v*u for w,v,u in zip(weights,x,z))
            sxc=math.fsum(w*v*c for w,v,c in zip(weights,x,counts))
            det=sxx*szz-sxz*sxz; relative_det=det/(sxx*szz)
            if relative_det<=1e-14:
                return {'alpha':alpha,'objective':math.inf,'rejected_ill_conditioned':True}
            A=(sxc*szz-szc*sxz)/det
            B=(sxx*szc-sxz*sxc)/det
        predicted=[A*v+B*u for v,u in zip(x,z)]
        return {'alpha':alpha,'A':A,'B':B,'predicted_counts':predicted,
                'objective':math.fsum(w*(p-c)**2 for w,p,c in zip(weights,predicted,counts)),
                'relative_normal_matrix_determinant':relative_det,
                'constant_branch':alpha==0}
    coarse=[profile(-1+i*.01) for i in range(301)]
    candidates=list(coarse)
    ratio=(math.sqrt(5)-1)/2
    for i in range(1,300):
        if coarse[i]['objective']>min(coarse[i-1]['objective'],coarse[i+1]['objective']):
            continue
        lo=coarse[i-1]['alpha']; hi=coarse[i+1]['alpha']
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo)
        fa=profile(a);fb=profile(b)
        for _ in range(80):
            if fa['objective']<=fb['objective']:
                hi=b;b=a;fb=fa;a=hi-ratio*(hi-lo);fa=profile(a)
            else:
                lo=a;a=b;fa=fb;b=lo+ratio*(hi-lo);fb=profile(b)
        candidates.extend((fa,fb,profile((lo+hi)/2)))
    best=min(candidates,key=lambda d:(d['objective'],abs(d['alpha'])))
    constant=profile(0)
    if constant['objective']<=best['objective']+1e-20*max(1,math.fsum(w*c*c for w,c in zip(weights,counts))):
        best=constant
    endpoint_values=[best['A']*ell**(-best['alpha'])+best['B'] for ell in (edges[0],edges[-1])]
    best.update(boundary_minimum=abs(best['alpha']+1)<1e-5 or abs(best['alpha']-2)<1e-5,
                A_positive=best['A']>0,density_nonnegative=min(endpoint_values)>=0,
                endpoint_densities=endpoint_values,
                rejected_coarse_count=sum(not math.isfinite(x['objective']) for x in coarse),
                search_scope='Guarded coarse profile with refined coarse local minima, not a certified global optimizer; near-zero log-limit and parameter identifiability unresolved.')
    return best

def main():
    root=Path(__file__).resolve().parent
    protocol_bytes=(root/'PROTOCOL.json').read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest()==PROTOCOL_SHA
    protocol=json.loads(protocol_bytes)
    inputs=[]
    for binding in protocol['input_bindings']:
        raw=Path(binding['path']).read_bytes()
        assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
        assert hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()==binding['git_blob']
        inputs.append(json.loads(raw))
    config,data,run=inputs
    assert data['config']==config
    for key in ['dimension','side','grids','methods','seeds']:
        assert config[key]==protocol[key]
    assert config['bin_edges']==protocol['bins']
    expected=set(itertools.product(protocol['seeds'],protocol['grids'],protocol['methods']))
    indexed={}
    edge=protocol['bins']
    total_lifetimes=0
    for record in data['records']:
        key=(record['seed'],record['grid'],record['method'])
        assert key in expected and key not in indexed
        lifetimes=[b-d for b,d in record['intervals']]
        assert all(math.isfinite(x) and x>0 for x in lifetimes)
        counts=[sum(a<=x<b for x in lifetimes) for a,b in zip(edge,edge[1:])]
        assert counts==record['counts']
        indexed[key]=counts;total_lifetimes+=len(lifetimes)
    assert set(indexed)==expected and len(indexed)==96
    rows=[];started=time.monotonic()
    for grid,method in itertools.product(protocol['grids'],protocol['methods']):
        by_field=[indexed[(seed,grid,method)] for seed in protocol['seeds']]
        pooled=[sum(xs[j] for xs in by_field) for j in range(len(edge)-1)]
        row={'grid':grid,'method':method,'pooled_counts':pooled,'fit':fit(edge,pooled,8*protocol['side']**2),'delete_one':[]}
        for i,seed in enumerate(protocol['seeds']):
            remaining=[pooled[j]-by_field[i][j] for j in range(len(pooled))]
            row['delete_one'].append({'omitted_seed':seed,'fit':fit(edge,remaining,7*protocol['side']**2)})
        alphas=[x['fit']['alpha'] for x in row['delete_one']]
        row['delete_one_alpha_range']=[min(alphas),max(alphas)]
        rows.append(row)
    gates=protocol['diagnostic_gates'];failed=[]
    for row in rows:
        f=row['fit'];name=str(row['grid'])+'/'+row['method']
        for flag in ['A_positive','density_nonnegative']:
            if not f[flag]:failed.append(name+':'+flag)
        if f['boundary_minimum']:failed.append(name+':profile_boundary')
        if row['delete_one_alpha_range'][1]-row['delete_one_alpha_range'][0]>gates['max_finest_delete_one_alpha_range_width'] and row['grid']==max(protocol['grids']):
            failed.append(name+':delete_one_instability')
    adjacent=[]
    for method in protocol['methods']:
        subset=[r for r in rows if r['method']==method]
        for a,b in zip(subset,subset[1:]):
            delta=abs(a['fit']['alpha']-b['fit']['alpha']);adjacent.append({'method':method,'from':a['grid'],'to':b['grid'],'alpha_difference':delta})
            if delta>gates['max_adjacent_grid_alpha_difference']:
                failed.append(method+':grid_'+str(a['grid'])+'_'+str(b['grid']))
    finest=[r['fit']['alpha'] for r in rows if r['grid']==max(protocol['grids'])]
    spread=max(finest)-min(finest)
    if spread>gates['max_finest_method_alpha_spread']:failed.append('finest_method_spread')
    output={'task_id':protocol['task_id'],'protocol_sha256':PROTOCOL_SHA,
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'actual_completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-started,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'input_validation':{'unique_rows':len(indexed),'positive_lifetimes_rebinned':total_lifetimes,'all_half_open_histograms_match':True},
            'rows':rows,'adjacent_grid_differences':adjacent,'finest_method_spread':spread,
            'failed_diagnostic_predicates':failed,'diagnostic_verdict':'FAIL' if failed else 'PASS',
            'scientific_effect':'NONE','scope':protocol['scope'],
            'interpretation':'The analytic O(1) remainder need not be constant B on these bins. Guarded exploratory curve fits do not test a certified continuum exponent or coefficient; near-zero log-limit, global optimization and identifiability are not certified. Delete-one ranges are not confidence intervals. Historical grid failures and held-out/continuum obligations remain.'}
    raw=(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with (root/'RESULTS.json').open('xb') as stream:stream.write(raw)
    print(json.dumps({'status':'EXECUTED','diagnostic_verdict':output['diagnostic_verdict'],'failed':failed,'result_sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':output['elapsed_seconds']}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

