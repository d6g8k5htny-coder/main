"""Exact planning and finite-law controls for conditional count confidence."""
import argparse
import hashlib
import itertools
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc
import finite_count_loss_certificate as prior
import count_confidence as cc
from gaussian_tail import exp_neg_bounds

BASE=Path(__file__).resolve().parent
PARENT_DIR='results/finite_count_loss1'
PARENT_CERT=PARENT_DIR+'/CERTIFICATE.json'
PARENT_RUN=PARENT_DIR+'/RUN.json'
PARENT_SHA256='24aa80e76b45280f2562d040bc51136cb055ef3b9a5f54d156afef2a1f525d03'
PARENT_RUN_SHA256='dfc4a55c1ccf43f42d2db88e5b95d81e4aad1ed6337af8ad11084a71d81a499b'
SOURCE_PATHS=tuple(sorted(set(prior.SOURCE_PATHS)|{
    'count_confidence.py','count_confidence_certificate.py','COUNT_CONFIDENCE.md',
    PARENT_CERT,PARENT_RUN,PARENT_DIR+'/RUN.sha256',PARENT_DIR+'/RESULTS.md'}))
SCOPE=('Fixed-sample simultaneous confidence transfer for finite-cutoff expected '
       'counts under the stated IID word law and pointwise row bounds. Exact '
       'prospective plans and abstract finite-law controls, not observed Gaussian '
       'data, authenticated sampling, infinite-field inference or lifetime-law confirmation.')
identity=prior.identity


def load_parent():
    fc.require(identity(BASE/PARENT_CERT)['sha256']==PARENT_SHA256,'Frozen count-loss certificate changed')
    fc.require(identity(BASE/PARENT_RUN)['sha256']==PARENT_RUN_SHA256,'Frozen count-loss receipt changed')
    # The unchanged parent's semantic replay is a separate required stage.
    return prior.validate_directory(BASE/PARENT_DIR)


def toy_controls():
    exact=retained=discarded=adaptive_bad=0
    for bits in itertools.product((0,1),repeat=8):
        rows=[(16*b,16*b) for b in bits]
        lo,hi=cc.mean_interval(1,Q(0),rows,Q(4),8)
        exact+=not(lo<=8<=hi)
        partial=[None if b else (0,0) for b in bits]
        retained+=cc.mean_interval(1,Q(0),partial,Q(4),8)[1]<8
        successes=[(0,0) for b in bits if not b]
        if successes:
            discarded+=cc.mean_interval(1,Q(0),successes,Q(4),len(successes))[1]<8
        adaptive=[None if (sum(bits)+i)%3==0 else row for i,row in enumerate(rows)]
        alo,ahi=cc.mean_interval(1,Q(0),adaptive,Q(4),8)
        adaptive_bad+=not(alo<=lo<=hi<=ahi)
    family_bad=0
    for chosen in range(4):
        intervals=[cc.mean_interval(1,Q(0),[(16*int(j==chosen),)*2],Q(10),1) for j in range(4)]
        family_bad+=any(not(lo<=4<=hi) for lo,hi in intervals)
    dependent_bad=0
    for b in (0,1):
        lo,hi=cc.mean_interval(1,Q(0),[(16*b,16*b)]*8,Q(6),8)
        dependent_bad+=not(lo<=8<=hi)
    return {'kind':'abstract finite laws, not field observations','enumerated_sequences':256,
        'exact_two_sided_failure':str(Q(exact,256)),
        'correct_two_sided_risk':str(cc.risk_upper(1,8,1,Q(4))),
        'wrong_capless_risk':str(2*exp_neg_bounds(Q(256))[1]),
        'retained_unresolved_upper_failure':str(Q(retained,256)),
        'discarding_failure':str(Q(discarded,256)),
        'discarding_empty_sequences':1,'adaptive_widening_violations':adaptive_bad,
        'family_failure':str(Q(family_bad,4)),
        'wrong_single_bin_risk':str(cc.risk_upper(1,1,1,Q(10))),
        'correct_family_risk':str(cc.risk_upper(1,1,4,Q(10))),
        'dependent_row_failure':str(Q(dependent_bad,2)),
        'wrong_independent_risk':str(cc.risk_upper(1,8,1,Q(6)))}


def certify():
    parent=load_parent();plans=[]
    for row in parent['budgets']:
        k=row['cutoff'];cap=row['finite_bar_cap'];r=Q(cap,100)
        risk=cc.risk_upper(k,40000,8,r)
        plans.append({'cutoff':k,'finite_bar_cap':cap,'sample_count':40000,'bins':8,
            'radius':str(r),'risk_upper':str(risk),'risk_decimal_up':fc.decimal_up(risk,18),
            'sample_count_for_radius_one':4*cap*cap,
            'clipping_failure_upper':row['clipping_failure_upper'],
            'expected_count_loss_upper':row['expected_count_loss_upper']})
    example=cc.mean_interval(1,Q(1,1000),[(4,5),None,(2,3),(6,8)],Q(1,2),4)
    return {'schema_version':1,'scope':SCOPE,
        'target':'Finite Gaussian F64,K on the side24 torus with fixed S_alpha,64 normalization; K in {1,24,32}.',
        'parent_certificate_sha256':PARENT_SHA256,'parent_receipt_sha256':PARENT_RUN_SHA256,
        'input_law_is_hypothesis':True,'iid_input_law_certified':False,
        'prospective_n_bins_radius_required':True,'pointwise_row_evidence_required':True,
        'adaptive_grid_refinement_allowed':True,'all_draws_retained_required':True,
        'family_risk_formula':'min(1,2*m*exp(-2*n*(r/(16*K^2))^2))',
        'interval_formula':'[max(0,mean(L)-r-16*K^2*p), min(16*K^2,mean(U)+r+16*K^2*p)]',
        'plans':plans,'toy_controls':toy_controls(),
        'arithmetic_example':{'kind':'invented numeric rows only; no authenticated grid or sampler budget',
            'rows':[[4,5],None,[2,3],[6,8]],'sample_count':4,'radius':'1/2',
            'illustrative_probability':'1/1000','interval':[str(x) for x in example]},
        'new_gaussian_barcodes':0,'new_deterministic_barcodes':0,
        'observed_ensemble_means_computed':False,'infinite_field_expected_count_certified':False,
        'lifetime_law_certified':False,'formal_proof_certified':False,
        'next_obligation':'Actual inference needs a fixed plan, the declared IID source and all pointwise certified rows; infinite-field count moments and numerical lifetime remainder remain separate.'}


def verify(cert):
    fc.require(fc.canonical_bytes(cert)==fc.canonical_bytes(certify()),'Confidence certificate differs from exact replay')
    return True


def render_report(c):
    lines=['# What sampling error would cost','',
        'The finite-cutoff bar cap gives simultaneous mean-confidence intervals',
        'under a fixed plan and the declared independent uniform-word input law.',
        'For eight bins and 40,000 draws, a radius of 1% of the cap has the',
        'following exact upper bound on family failure probability. These are',
        '**prospective calculations; no ensemble has been sampled.**','',
        '| Cutoff | Cap | Count radius | Failure bound | Draws for radius one at the same bound |',
        '|---|---:|---:|---:|---:|']
    for r in c['plans']:
        lines.append('| '+' | '.join(str(r[k]) for k in ('cutoff','finite_bar_cap','radius','risk_decimal_up','sample_count_for_radius_one'))+' |')
    lines+=['','The intervals are',
        '`[max(0, mean(L)-r-C*p), min(C, mean(U)+r+C*p)]`, with `C=16*K^2`.',
        'Every draw remains in the denominator. An unresolved computation supplies',
        '`[0,C]`. Grid refinement may depend on the whole sample because concentration',
        'is applied to fixed latent polynomial counts and rows only bound those counts.',
        'Bins, radius and sample count must be fixed before inspecting the sample.','',
        'The large sample sizes at larger cutoffs expose the cost of this worst-case',
        'bound. They are not a recommendation to launch an ensemble. The gap between',
        'lower and upper grid means and the inherited clipping correction add uncertainty.','',
        'Exact abstract finite-law checks detect omitted cap scaling, omitted bin',
        'multiplicity, dependent rows and discarding unresolved outcomes. They are',
        'neither polynomial data nor evidence of an IID physical source.','',
        'The arithmetic checks do not authenticate input-law or row-certificate premises.',
        'No observed ensemble mean, infinite-field inference or lifetime-law confirmation',
        'is certified. The existing deterministic word fixture remains deterministic.','',
        '[Proof and assumptions](../../COUNT_CONFIDENCE.md) · [Certificate](CERTIFICATE.json) · [Receipt](RUN.json)','',
        '```sh',
        'python -B -S experiments/periodic_h0/count_confidence_certificate.py --verify experiments/periodic_h0/results/count_confidence1',
        '```','',SCOPE,'']
    return '\n'.join(lines)


def validate_directory(directory):
    directory=Path(directory);r=fc.load_json(directory/'RUN.json')
    fc.require(type(r) is dict and set(r)=={'schema_version','utc','python','platform','sources','outputs','scope'},'Invalid receipt fields')
    fc.require(type(r['schema_version']) is int and r['schema_version']==1 and r['scope']==SCOPE,'Invalid receipt version/scope')
    fc.require(type(r['utc']) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00',r['utc']) is not None,'Invalid UTC')
    stamp=datetime.fromisoformat(r['utc'])
    fc.require(stamp.tzinfo==timezone.utc and stamp.isoformat(timespec='microseconds')==r['utc'],'Invalid UTC')
    for key in ('python','platform'):
        fc.require(type(r[key]) is str and 0<len(r[key])<=4096 and '\x00' not in r[key],'Invalid environment')
    fc.require(type(r['sources']) is dict and set(r['sources'])==set(SOURCE_PATHS),'Invalid source roles')
    fc.require(type(r['outputs']) is dict and set(r['outputs'])=={'CERTIFICATE.json'},'Invalid output roles')
    for item in [*r['sources'].values(),*r['outputs'].values()]:
        fc.require(type(item) is dict and set(item)=={'bytes','sha256'},'Invalid identity')
        fc.require(type(item['bytes']) is int and item['bytes']>0,'Invalid byte count')
        fc.require(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Invalid hash')
    fc.require((directory/'RUN.sha256').read_bytes()==(identity(directory/'RUN.json')['sha256']+'\n').encode(),'Receipt binding mismatch')
    for name in SOURCE_PATHS:fc.require(identity(BASE/name)==r['sources'][name],'Source binding mismatch: '+name)
    fc.require(identity(directory/'CERTIFICATE.json')==r['outputs']['CERTIFICATE.json'],'Output binding mismatch')
    cert=fc.load_json(directory/'CERTIFICATE.json')
    fc.require((directory/'RESULTS.md').read_bytes()==render_report(cert).encode(),'Report mismatch')
    return cert


def verify_directory(directory):
    verify(validate_directory(directory));return True


def produce(directory):
    directory=Path(directory);fc.require(not directory.exists(),'Output must be new')
    cert=certify();directory.mkdir(parents=True)
    (directory/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(cert))
    r={'schema_version':1,'utc':datetime.now(timezone.utc).isoformat(timespec='microseconds'),
       'python':sys.version,'platform':platform.platform(),'scope':SCOPE,
       'sources':{name:identity(BASE/name) for name in SOURCE_PATHS},
       'outputs':{'CERTIFICATE.json':identity(directory/'CERTIFICATE.json')}}
    raw=fc.canonical_bytes(r);(directory/'RUN.json').write_bytes(raw)
    (directory/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    (directory/'RESULTS.md').write_text(render_report(cert))
    return cert


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--produce',type=Path);g.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:verify_directory(a.verify)
    else:produce(a.produce)
    print('Exact fixed-sample confidence replay passed; sampling and row-evidence premises remain explicit.')


if __name__=='__main__':main()
