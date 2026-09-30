"""Execute the predeclared exploratory pilot; retain every realization and bin."""
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
from controls import run_controls

def save(path,data):
    # Large observation arrays stay compact; the human report carries readable tables.
    path.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')

def validate_config(config):
    def require(ok,message):
        if not ok:raise ValueError(message)
    integer=lambda x:type(x) is int and x>=0
    require(config['dimension']==2 and config['side']==24 and config['denominator_cutoff']==64,
            'This pilot implements only d=2,L=24,denominator64')
    seeds=config['seeds'];grids=config['grids'];cutoffs=config['cutoffs']
    require(isinstance(seeds,list) and len(seeds)>=2 and all(integer(s) for s in seeds) and len(set(seeds))==len(seeds),
            'Distinct nonnegative integer replicate seeds required')
    for seq in (grids,cutoffs):
        require(isinstance(seq,list) and len(seq)>0 and all(integer(s) and s>0 for s in seq) and seq==sorted(set(seq)),
                'Increasing distinct positive integer grids/cutoffs required')
    require(max(cutoffs)<=64 and min(grids)>2*max(cutoffs),'Aliased or unsupported cutoff')
    start=config['covariance_seeds_start'];count=config['covariance_replicates']
    require(integer(start) and integer(count) and count>=2,'At least two covariance replicates required')
    require(not any(start<=s<start+count for s in seeds),'Pilot and calibration seed sets must be disjoint')
    n=config['covariance_grid'];c=config['covariance_cutoff']
    require(integer(n) and integer(c) and 1<=c<=64 and n>2*c,'Invalid covariance grid/cutoff')
    lags=config['covariance_index_lags']
    require(isinstance(lags,list) and len(lags)>0 and all(isinstance(t,list) and len(t)==2 and all(integer(x) and x<n for x in t) for t in lags),
            'Covariance lags must be in-grid integer pairs')
    require(lags[0]==[0,0],'First calibration point must be the origin')
    ex.bin_counts([],config['bin_edges'])
    return True

def paired_differences(records,config):
    cutoffs,grids,seeds=config['cutoffs'],config['grids'],config['seeds']
    result=[]
    comparisons=[((c,a),(c,b),'grid') for c in cutoffs for a,b in zip(grids[:-1],grids[1:])]
    comparisons += [((a,g),(b,g),'cutoff') for g in grids for a,b in zip(cutoffs[:-1],cutoffs[1:])]
    lookup={(r['seed'],r['cutoff'],r['grid']):np.array(r['counts'])/24**2 for r in records}
    for (c1,g1),(c2,g2),kind in comparisons:
        delta=np.array([lookup[s,c2,g2]-lookup[s,c1,g1] for s in seeds])
        result.append({'kind':kind,'from':[c1,g1],'to':[c2,g2],
                       'mean_mass_difference':np.mean(delta,axis=0).tolist(),
                       'se_mass_difference':(np.std(delta,axis=0,ddof=1)/math.sqrt(len(seeds))).tolist()})
    return result

def verify_observations(data):
    """Recompute report fields from retained observations; does not rerun persistence."""
    config=data['config'];validate_config(config)
    def require(ok,message):
        if not ok:raise ValueError(message)
    def equivalent(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return isinstance(b,(float,int)) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)
        return a==b
    expected={(s,c,g) for s in config['seeds'] for c in config['cutoffs'] for g in config['grids']}
    seen=set()
    for r in data['records']:
        key=r['seed'],r['cutoff'],r['grid']
        require(key in expected and key not in seen,'Missing/duplicate/unknown realization');seen.add(key)
        require(len(r['essential'])==1 and np.isfinite(r['essential']).all(),'Essential H0 mismatch')
        require(type(r['zero_count']) is int and r['zero_count']>=0,'Invalid zero-persistence count')
        lengths=[]
        for b,d in r['intervals']:
            require(math.isfinite(b) and math.isfinite(d) and b>d,'Invalid finite positive interval')
            lengths.append(b-d)
        require(r['counts']==ex.bin_counts(lengths,config['bin_edges']),'Observation/count mismatch')
        require(len(lengths)+r['zero_count']==r['grid']**2-1,'Vertex/finite/essential accounting mismatch')
    require(seen==expected,'Incomplete realization matrix')
    summaries=[]
    for c in config['cutoffs']:
        for g in config['grids']:
            rows=[r for r in data['records'] if r['cutoff']==c and r['grid']==g]
            summaries.append({'cutoff':c,'grid':g,'replicates':len(rows),
                'finite_positive_total':sum(len(r['intervals']) for r in rows),
                'bins':ex.summarize_counts([r['counts'] for r in rows],config['bin_edges'])})
    require(equivalent(summaries,data['summary']),'Summary does not match retained counts')
    require(equivalent(paired_differences(data['records'],config),data['paired_differences']),'Paired differences mismatch')
    empirical=data['covariance_empirical']
    require('point_samples' in empirical,'Missing calibration point samples')
    samples=np.asarray(empirical['point_samples'],dtype=float)
    require(samples.shape==(config['covariance_replicates'],len(config['covariance_index_lags'])) and np.isfinite(samples).all(),'Invalid calibration sample array')
    require(empirical['replicates']==config['covariance_replicates'],'Calibration replicate count mismatch')
    products=samples*samples[:,[0]]
    require(equivalent(np.mean(products,axis=0).tolist(),empirical['mean_products']),'Calibration product mean mismatch')
    require(equivalent((np.std(products,axis=0,ddof=1)/math.sqrt(len(samples))).tolist(),empirical['se_products']),'Calibration product SE mismatch')
    require(equivalent(np.mean(samples,axis=0).tolist(),empirical['mean_values']),'Calibration value mean mismatch')
    target=[ex.reference_covariance(config['covariance_cutoff'],(x*24/config['covariance_grid'],y*24/config['covariance_grid'])) for x,y in config['covariance_index_lags']]
    require(equivalent(target,empirical['finite_target']),'Calibration target mismatch')
    require(equivalent({str(c):ex.spectral_diagnostics(c) for c in config['cutoffs']},data['spectral_diagnostics']),'Spectral diagnostics mismatch')
    require(equivalent(run_controls(),data['controls']),'Control replay mismatch')
    return True

def execute(config):
    validate_config(config)
    seeds=config['seeds'];grids=config['grids'];cutoffs=config['cutoffs'];edges=config['bin_edges']
    controls=run_controls()
    if not controls['all_controls_pass']:raise ValueError('Negative controls failed')
    records=[]
    for seed in seeds:
        bank=ex.mode_bank(seed,max(cutoffs))
        for cutoff in cutoffs:
            for grid in grids:
                f=ex.field_grid(bank,grid,cutoff)
                bars=ex.gudhi_h0(f)
                if len(bars['essential'])!=1:raise ValueError('Expected one essential H0 interval')
                lengths=[b-d for b,d in bars['intervals']]
                records.append({'seed':seed,'grid':grid,'cutoff':cutoff,
                                'counts':ex.bin_counts(lengths,edges),**bars})
    summaries=[]
    for cutoff in cutoffs:
        for grid in grids:
            rows=[r for r in records if r['cutoff']==cutoff and r['grid']==grid]
            summaries.append({'cutoff':cutoff,'grid':grid,'replicates':len(rows),
                              'finite_positive_total':sum(len(r['intervals']) for r in rows),
                              'bins':ex.summarize_counts([r['counts'] for r in rows],edges)})
    # Paired differences reuse fields; they are not independent additional samples.
    paired=paired_differences(records,config)
    # Empirical covariance uses a separate set of independent fields.
    products=[];point_values=[]
    for seed in range(config['covariance_seeds_start'],config['covariance_seeds_start']+config['covariance_replicates']):
        f=ex.field_grid(ex.mode_bank(seed,config['covariance_cutoff']),config['covariance_grid'],config['covariance_cutoff'])
        values=np.array([f[x,y] for x,y in config['covariance_index_lags']])
        products.append(values*f[0,0]);point_values.append(values)
    products=np.array(products);point_values=np.array(point_values)
    target=np.array([ex.reference_covariance(config['covariance_cutoff'],(x*24/config['covariance_grid'],y*24/config['covariance_grid'])) for x,y in config['covariance_index_lags']])
    empirical={'replicates':len(products),'finite_target':target.tolist(),'point_samples':point_values.tolist(),
               'mean_products':np.mean(products,axis=0).tolist(),
               'se_products':(np.std(products,axis=0,ddof=1)/math.sqrt(len(products))).tolist(),
               'mean_values':np.mean(point_values,axis=0).tolist(),
               'scope':'Independent-field product means, known ensemble mean zero; marginal standard errors, no simultaneous coverage claim'}
    return {'schema_version':1,'config':config,'records':records,
            'summary':summaries,'paired_differences':paired,'controls':controls,
            'covariance_empirical':empirical,
            'spectral_diagnostics':{str(c):ex.spectral_diagnostics(c) for c in cutoffs},
            'interpretation':'Exploratory discrete-model output. No slope fit, confirmation window, continuum certificate or theorem verdict.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('pilot_config.json'))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path)
    mode.add_argument('--verify',type=Path,help='Recompute summaries/counts from saved observations; no new simulation')
    args=parser.parse_args()
    if args.verify:
        verify_observations(json.loads(args.verify.read_text()))
        print(json.dumps({'verified':str(args.verify),'scope':'Internal observation/report consistency only'}));return
    if args.output.exists():parser.error('Output must be a new directory; preserve previous runs')
    config=json.loads(args.config.read_text());start=time.perf_counter()
    data=execute(config)
    verify_observations(data)
    args.output.mkdir(parents=True)
    save(args.output/'observations.json',data)
    sources={}
    for p in [Path(__file__),Path(ex.__file__),Path(__file__).with_name('controls.py'),args.config,Path(__file__).with_name('requirements.txt')]:
        raw=p.read_bytes();sources[p.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    save(args.output/'RUN.json',{'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'python':sys.version,'numpy':np.__version__,'gudhi':gudhi.__version__,'platform':platform.platform(),
        'float':'IEEE754 float64 / complex128','rng':'NumPy PCG64; seed per independent realization; shared bank across grids/cutoffs',
        'sources':sources,'observations_sha256':hashlib.sha256((args.output/'observations.json').read_bytes()).hexdigest(),
        'scientific_effect':'NONE','reproducibility':'Exact replay expected in the tested environment; cross-platform floating-point ties may differ'})
    print(json.dumps({'records':len(data['records']),'elapsed_seconds':time.perf_counter()-start,'output':str(args.output)}))

if __name__=='__main__':main()
