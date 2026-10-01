"""Exact finite-cutoff count-loss budgets under the declared word-input law."""
import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc
import gaussian_coupling as gc
import gaussian_coupling_certificate as prior
import finite_count_loss as cl

BASE=Path(__file__).resolve().parent
PARENT_CERT='results/gaussian_coupling1/CERTIFICATE.json'
PARENT_RUN='results/gaussian_coupling1/RUN.json'
PARENT_SHA256='1ab5e225ec3e0a570910afc11026adf38ce39d01a4a8a16caa9ad487c95e1e32'
PARENT_RUN_SHA256='dbc5b27e6ea7e147133b48194889599a7fe1471bdb218c0513cb1ccb96a77d61'
SOURCE_PATHS=tuple(sorted(set(prior.SOURCE_PATHS)|{
    'finite_count_loss.py','finite_count_loss_certificate.py','FINITE_COUNT_LOSS.md',
    'APPROXIMATION.md',PARENT_CERT,PARENT_RUN,'results/gaussian_coupling1/RUN.sha256'}))
SCOPE=('Uniform planar finite-polynomial H0 bar-count bound and finite-cutoff '
       'expected-count clipping-loss budgets under the stated IID word-input model. '
       'No observed Gaussian ensemble, infinite-field expected count, historical '
       'coupling, lifetime-law confirmation or formal proof certified.')
identity=prior.identity


def load_parent():
    fc.require(identity(BASE/PARENT_CERT)['sha256']==PARENT_SHA256,'Frozen sampler certificate changed')
    fc.require(identity(BASE/PARENT_RUN)['sha256']==PARENT_RUN_SHA256,'Frozen sampler receipt changed')
    r=fc.load_json(BASE/PARENT_RUN)
    fc.require(set(r['sources'])==set(prior.SOURCE_PATHS),'Parent source roles changed')
    fc.require((BASE/'results/gaussian_coupling1/RUN.sha256').read_bytes()==
               (PARENT_RUN_SHA256+'\n').encode(),'Parent receipt digest changed')
    for name,expected in r['sources'].items():
        fc.require(identity(BASE/name)==expected,'Changed sampler dependency: '+name)
    fc.require(identity(BASE/PARENT_CERT)==r['outputs']['CERTIFICATE.json'],'Parent output changed')
    return fc.load_json(BASE/PARENT_CERT)


def certify():
    parent=load_parent()
    inherited={x['cutoff']:x for x in parent['budgets']}
    budgets=[]
    for k in (1,24,32):
        # K24/K32 consume exact frozen evaluated parent bytes. Its independent
        # replay remains a separate unchanged CI stage; K1 is evaluated here.
        source=gc.budget(k) if k==1 else inherited[k]
        cap=cl.bar_cap(k);loss=cap*Q(source['clipping_failure_upper'])
        budgets.append({'cutoff':k,'real_coordinates':(2*k+1)**2,
            'finite_bar_cap':cap,'coefficient_error_upper':source['coefficient_error_upper'],
            'coefficient_error_decimal_up':source['coefficient_error_decimal_up'],
            'clipping_failure_upper':source['clipping_failure_upper'],
            'clipping_failure_decimal_up':source['clipping_failure_decimal_up'],
            'expected_count_loss_upper':str(loss),
            'expected_count_loss_decimal_up':fc.decimal_up(loss,24),
            'budget_source':'fresh exact K1 evaluation' if k==1 else 'fixed parent certificate'})
    return {'schema_version':1,'scope':SCOPE,
        'target':'Finite Gaussian F64,K on the side24 torus, with the fixed S_alpha,64 normalization; not the infinite field F.',
        'parent_certificate_sha256':PARENT_SHA256,'parent_receipt_sha256':PARENT_RUN_SHA256,
        'input_law_is_hypothesis':True,'iid_input_law_certified':False,
        'budgets':budgets,'positive_finite_bar_cap':'16*K^2',
        'expectation_formula':'max(0,E[L]-16*K^2*p_clip) <= E[N_F64,K([a,b))] <= min(16*K^2,E[U]+16*K^2*p_clip)',
        'unresolved_grid_policy':'Keep every realization with L=0,U=16*K^2; do not discard or condition on successful certificates.',
        'new_gaussian_barcodes':0,'new_deterministic_barcodes':0,
        'expected_bin_counts_computed':False,'infinite_field_expected_count_certified':False,
        'lifetime_law_certified':False,'formal_proof_certified':False,
        'historical_coupling_error':None,
        'next_obligation':'A declared ensemble at matching cutoff, unconditional observable-mean bounds, and independent infinite-field exceptional-count moments remain necessary; no sampled mean is certified here.'}


def verify(cert):
    fc.require(fc.canonical_bytes(cert)==fc.canonical_bytes(certify()),'Count-loss certificate differs from exact replay')
    return True


def render_report(c):
    lines=['# Quantifying the cost of rare clipping events','',
        'Every real planar Fourier polynomial with frequencies in the square',
        '[-K,K]^2 has at most **16 K² positive finite H₀ bars**, including',
        'degenerate coefficient choices. The proof uses a conservative Bézout',
        'bound for Morse perturbations and persistence stability.','',
        'Under the previously declared independent uniform-word model, this',
        'converts the clipping failure probability into the following absolute',
        '**expected-count correction** for the finite Gaussian polynomial F64,K:', '',
        '| Cutoff K | Uniform bar cap | Coupling error at most | Clipping probability at most | Expected-count correction at most |',
        '|---|---:|---|---|---|']
    for r in c['budgets']:
        lines.append('| '+' | '.join([str(r['cutoff']),str(r['finite_bar_cap']),
            r['coefficient_error_decimal_up'],r['clipping_failure_decimal_up'],
            r['expected_count_loss_decimal_up']])+' |')
    lines+=['','For a positive lifetime bin [a,b), the exact inequalities are',
        '`max(0, E[L] − correction) ≤ E[N_F64,K([a,b))] ≤ min(16 K², E[U] + correction)`.',
        'For polynomial counts, L uses the contracted bin. U uses the expanded',
        'bin only when a > 2 rho; otherwise U = 16 K². For the bounded grid',
        'observables in the proof, the expanded upper count requires',
        'a > 2(epsilon + rho); otherwise U = 16 K². Unresolved grids are kept',
        'with [L,U]=[0,16 K²]. Their outcomes must not be discarded.','',
        '**No expectations E[L] or E[U] have been evaluated here.** This delivery',
        'generates no random field or barcode. Sample means need their own',
        'sampling-error argument. The previous deterministic cutoff1 example',
        'does not become a Gaussian observation.','',
        'This removes the missing exceptional-count factor for the finite-cutoff',
        'clipping comparison. The infinite smooth field has no such fixed-cutoff',
        'cap. Its exceptional-count moments, the spectral-tail step in expected',
        'counts, and the numerical lifetime remainder remain separate.','',
        '[Proof and exact input assumptions](../../FINITE_COUNT_LOSS.md) ·',
        '[Certificate](CERTIFICATE.json) · [Source receipt](RUN.json)','',
        '```sh','python -B -S experiments/periodic_h0/finite_count_loss_certificate.py --verify experiments/periodic_h0/results/finite_count_loss1','```','',SCOPE,'']
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
    print('Exact finite-cutoff count-loss replay passed; input-law and infinite-field obligations remain explicit.')


if __name__=='__main__':main()
