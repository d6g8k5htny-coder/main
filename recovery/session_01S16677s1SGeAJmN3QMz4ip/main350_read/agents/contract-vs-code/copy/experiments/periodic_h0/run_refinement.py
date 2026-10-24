"""Execute the frozen coupled resolution diagnostic; no continuum verdict."""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import gudhi
import experiment as ex
import refinement as ref
import run_pilot as pilot

METHODS=('cubical','plus','minus')

def validate(config):
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    require(config['schema_version']==1 and config['dimension']==2 and config['side']==24 and config['cutoff']==config['bank_max_cutoff']==24,'Only cutoff24 planar side24 design implemented')
    s=config['seeds'];g=config['grids']
    require(isinstance(s,list) and len(s)>=2 and all(type(x)is int and x>=0 for x in s) and len(set(s))==len(s),'Invalid seeds')
    require(isinstance(g,list) and len(g)>=2 and all(type(x)is int and x>48 for x in g) and g==sorted(set(g)) and all(b%a==0 for a,b in zip(g,g[1:])),'Nested grids above Nyquist required')
    require(config['methods']==list(METHODS),'All three comparison methods required')
    ex.bin_counts([],config['bin_edges'])

def diagram(record):
    return np.asarray([[-b,-d] for b,d in record['intervals']],dtype=float).reshape(-1,2)

def comparisons(records,config):
    lookup={(r['seed'],r['grid'],r['method']):r for r in records};out=[]
    pairs=[(a,m,b,m,'grid') for m in METHODS for a,b in zip(config['grids'][:-1],config['grids'][1:])]
    pairs += [(g,'cubical',g,m,'filtration') for g in config['grids'] for m in ('plus','minus')]
    for a,m,b,n,kind in pairs:
        per=[]
        for seed in config['seeds']:
            p,q=lookup[seed,a,m],lookup[seed,b,n]
            per.append({'seed':seed,'count_delta':[y-x for x,y in zip(p['counts'],q['counts'])],
                        'finite_bottleneck':float(gudhi.bottleneck_distance(diagram(p),diagram(q),e=0))})
        delta=np.asarray([r['count_delta'] for r in per],dtype=float)/576
        out.append({'kind':kind,'from':[a,m],'to':[b,n],'per_field':per,
                    'mean_mass_difference':np.mean(delta,axis=0).tolist(),
                    'se_mass_difference':(np.std(delta,axis=0,ddof=1)/math.sqrt(len(per))).tolist()})
    return out

def summaries(records,config):
    return [{'grid':g,'method':m,'replicates':len(config['seeds']),
             'finite_positive_total':sum(len(r['intervals']) for r in records if r['grid']==g and r['method']==m),
             'bins':ex.summarize_counts([r['counts'] for r in records if r['grid']==g and r['method']==m],config['bin_edges'])}
            for g in config['grids'] for m in METHODS]

def execute(config):
    validate(config);records=[];diagnostics=[]
    for seed in config['seeds']:
        bank=ex.mode_bank(seed,24);h=ref.hessian_diagnostic(bank,24)
        diagnostics.append({'seed':seed,'hessian_majorant_float':h,
                            'interpolation_epsilon_float':[ref.interpolation_diagnostic(h,g) for g in config['grids']]})
        previous=None
        for grid in config['grids']:
            f=ex.field_grid(bank,grid,24)
            if previous is not None and not np.allclose(f[::grid//previous[0],::grid//previous[0]],previous[1],rtol=0,atol=1e-12):raise ValueError('Nested Fourier coupling failed')
            previous=(grid,f)
            for method in METHODS:
                bars=ex.gudhi_h0(f) if method=='cubical' else ref.triangulated_h0(f,method)
                records.append({'seed':seed,'grid':grid,'method':method,'counts':ex.bin_counts([b-d for b,d in bars['intervals']],config['bin_edges']),**bars})
        print(json.dumps({'completed_seed':seed}),flush=True)
    return {'schema_version':1,'config':config,'records':records,'summary':summaries(records,config),
            'comparisons':comparisons(records,config),'diagnostics':diagnostics,
            'scope':'Exploratory finite Fourier and sampled-filtration comparison. Diagnostic bounds are not outward-rounded certificates. No continuum tail, asymptotic window or scientific acceptance.'}

def equivalent(expected,actual):
    """Tolerate roundoff only in recomputed derived floats, never count identities."""
    if isinstance(expected,dict):
        return isinstance(actual,dict) and expected.keys()==actual.keys() and all(equivalent(expected[k],actual[k]) for k in expected)
    if isinstance(expected,list):
        return isinstance(actual,list) and len(expected)==len(actual) and all(equivalent(x,y) for x,y in zip(expected,actual))
    if isinstance(expected,(float,np.floating)):
        return (isinstance(actual,(float,np.floating)) or type(actual) is int) and math.isfinite(actual) and math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-15)
    return type(expected) is type(actual) and expected==actual

def verify(data):
    config=data['config'];validate(config);seen=set()
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    expected={(s,g,m) for s in config['seeds'] for g in config['grids'] for m in METHODS}
    for r in data['records']:
        key=r['seed'],r['grid'],r['method'];require(key in expected and key not in seen,'Unknown/duplicate record');seen.add(key)
        require(len(r['essential'])==1 and np.isfinite(r['essential']).all(),'Essential count')
        require(type(r['zero_count'])is int and r['zero_count']>=0,'Invalid zero count')
        require(all(math.isfinite(b) and math.isfinite(d) and b>d for b,d in r['intervals']),'Invalid intervals')
        require(len(r['intervals'])+r['zero_count']==r['grid']**2-1,'Vertex accounting')
        require(r['counts']==ex.bin_counts([b-d for b,d in r['intervals']],config['bin_edges']),'Count mismatch')
    require(seen==expected,'Incomplete matrix')
    require(equivalent(summaries(data['records'],config),data['summary']),'Summary mismatch')
    # Regenerate derived fields; tolerate tiny platform roundoff, not changed counts.
    require(equivalent(comparisons(data['records'],config),data['comparisons']),'Comparison mismatch')
    expected_diagnostics=[]
    for seed in config['seeds']:
        h=ref.hessian_diagnostic(ex.mode_bank(seed,24),24)
        expected_diagnostics.append({'seed':seed,'hessian_majorant_float':h,'interpolation_epsilon_float':[ref.interpolation_diagnostic(h,g) for g in config['grids']]})
    require(equivalent(expected_diagnostics,data['diagnostics']),'Diagnostic mismatch')
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('refinement_config.json'))
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--output',type=Path);group.add_argument('--verify',type=Path)
    args=p.parse_args()
    if args.verify:verify(json.loads(args.verify.read_text()));print('Internal refinement consistency verified');return
    if args.output.exists():p.error('Output must be a new directory')
    raw=args.config.read_bytes();config=json.loads(raw);sources={}
    for role,path in {'runner':Path(__file__),'model':Path(ex.__file__),'refinement':Path(ref.__file__),'pilot_utilities':Path(pilot.__file__),'controls':Path(__file__).with_name('controls.py'),'dependencies':Path(__file__).with_name('requirements.txt'),'config':args.config}.items():
        content=raw if role=='config' else path.read_bytes()
        sources[role]={'filename':path.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    start=time.perf_counter();data=execute(config);verify(data);args.output.mkdir(parents=True)
    pilot.save(args.output/'observations.json',data)
    pilot.save(args.output/'RUN.json',{'source_receipt_schema':2,'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'python':sys.version,'numpy':np.__version__,'gudhi':gudhi.__version__,'platform':platform.platform(),'sources':sources,
        'observations_sha256':hashlib.sha256((args.output/'observations.json').read_bytes()).hexdigest(),'scientific_effect':'NONE'})
    print(json.dumps({'records':len(data['records']),'elapsed_seconds':time.perf_counter()-start,'output':str(args.output)}))

if __name__=='__main__':main()
