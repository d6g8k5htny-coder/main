"""Replay a prospective quantized sampler; preserve the historical coupling gap."""
import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime,timezone
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc
import gaussian_tail as gt
import gaussian_tail_certificate as prior
import gaussian_coupling as gc

BASE=Path(__file__).resolve().parent
TAIL_CERT='results/gaussian_tail1/CERTIFICATE.json'
TAIL_RUN='results/gaussian_tail1/RUN.json'
SOURCE_PATHS=tuple(sorted(set(prior.SOURCE_PATHS)|{
    'gaussian_coupling.py','gaussian_coupling_certificate.py','GAUSSIAN_COUPLING.md',
    TAIL_CERT,TAIL_RUN,'results/gaussian_tail1/RUN.sha256'}))
SCOPE=('Prospective side24 finite-bit inverse-normal coupling under explicit ideal '
       'IID uniform-word inputs. No RNG law, historical Gaussian sample, barcode '
       'experiment, lifetime law, or expected-count confirmation certified.')
identity=prior.identity


def certify():
    fc.require(identity(BASE/TAIL_CERT)['sha256']=='9284099183648476bbceb6ddc6cd54bb212713402b1b8c15a8684e237587c960','Frozen tail certificate changed')
    fc.require(identity(BASE/TAIL_RUN)['sha256']=='2bab16d8bb51c55f599d30ff119aba57bf31d24bcb5ce1192658e085a25a199c','Frozen tail receipt changed')
    prior.verify_directory(BASE/'results/gaussian_tail1')
    budgets=[]
    for k in (24,32):
        r=gc.budget(k);tail=gt.tail_bound(k,Q(8))
        error=Q(tail['tail_supremum_upper'])+Q(r['coefficient_error_upper'])+Q(tail['normalization_loss_upper'])*Q(r['polynomial_norm_upper'])
        failure=min(Q(1),Q(r['clipping_failure_upper'])+Q(tail['failure_probability_upper']))
        r.update(ideal_field_supremum_error_upper=str(error),
                 ideal_field_supremum_error_decimal_up=fc.decimal_up(error,24),
                 combined_failure_upper=str(failure),
                 combined_failure_decimal_up=fc.decimal_up(failure,24),
                 tail_bound=tail)
        budgets.append(r)
    words=[i*((1<<128)-1)//8 for i in range(9)]
    p=gc.polynomial(1,words)
    fixture={'kind':'deterministic arithmetic fixture; not a random field',
             'cutoff':1,'words':[str(x) for x in words],
             'dc':str(p['dc']),'real_basis_modes':[[x,y,str(a),str(b)] for x,y,a,b in p['modes']]}
    return {'schema_version':1,'scope':SCOPE,
            'model':'The ideal side24 Gaussian field coupled through independent U_i uniform on (0,1), J_i=floor(2^128 U_i); independent omitted Gaussian modes.',
            'input_law_is_hypothesis':True,'iid_input_law_certified':False,
            'budgets':budgets,'precision_contract':gc.precision_contract(),'deterministic_fixture':fixture,
            'historical_object':{'source_sha256':prior.FINITE_SHA256,'coupling_error':None,'infinite_field_diagram_bound':None},
            'historical_gaussian_draw_certified':False,'new_gaussian_barcodes':0,
            'lifetime_law_certified':False,
            'next_obligation':'Use an explicit word-input ensemble and certify its newly constructed polynomial and barcode; numerical remainder and exceptional-count moments remain separate.'}


def verify(cert):
    fc.require(fc.canonical_bytes(cert)==fc.canonical_bytes(certify()),'Coupling certificate differs from exact replay')
    return True


def render_report(c):
    lines=['# A constructive Gaussian coefficient coupling','',
           'A finite-word sampler now has an explicit uniform error budget **under',
           'the declared independent uniform-input model**. It constructs a new',
           'dyadic polynomial; it does not certify the law of any physical or',
           'pseudorandom bit source. The historical polynomial remains separate.','',
           '| Cutoff | Low-mode error at most | Clipping failure at most | Full-field error at most | Combined failure at most |',
           '|---|---|---|---|---|']
    for r in c['budgets']:
        lines.append('| '+ ' | '.join([str(r['cutoff']),r['coefficient_error_decimal_up'],r['clipping_failure_decimal_up'],r['ideal_field_supremum_error_decimal_up'],r['combined_failure_decimal_up']])+' |')
    lines+=['','The full-field column includes the threshold8 infinite spectral tail and',
            'normalization error. It compares the ideal smooth field with the new',
            'polynomial, before any grid or barcode computation. Add a verified',
            'finite-polynomial barcode error to use the persistence bound.','',
            'The only executed construction is a nine-word deterministic arithmetic',
            'fixture at cutoff1. No new Gaussian barcode or held-out confirmation',
            'sample was generated. The table is a uniform sampler theorem for',
            'cutoffs24 and32, not an observation of those ensembles.','',
            '[Proof and input assumptions](../../GAUSSIAN_COUPLING.md) ·',
            '[Exact certificate](CERTIFICATE.json) · [Source receipt](RUN.json)','',
            '```sh','python -B -S experiments/periodic_h0/gaussian_coupling_certificate.py --verify experiments/periodic_h0/results/gaussian_coupling1','```','',SCOPE,'']
    return '\n'.join(lines)


def verify_directory(directory):
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
    cert=fc.load_json(directory/'CERTIFICATE.json');verify(cert)
    fc.require((directory/'RESULTS.md').read_bytes()==render_report(cert).encode(),'Report mismatch')
    return True


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


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--produce',type=Path);g.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:verify_directory(a.verify)
    else:produce(a.produce)
    print('Prospective quantized coupling replay passed; IID input law is a premise, historical coupling remains open.')


if __name__=='__main__':main()
