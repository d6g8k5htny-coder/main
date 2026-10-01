"""Replay a Gaussian tail bound; keep the finite-sample coupling premise missing."""
import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc
import gaussian_tail as gt
import exact_barcode_certificate as bc

BASE=Path(__file__).resolve().parent
FINITE_CERT='results/exact_h0_1/CERTIFICATE.json'
FINITE_RUN='results/exact_h0_1/RUN.json'
FINITE_SHA256='845dc940e1781e570d4a6614a00889f33b239274c00e367dc2af1e3b86906fb8'
FINITE_RUN_SHA256='e51fba013218e746c6845283312cd9ac1e688a8b960e9a2c577a097be5b54c45'
SOURCE_PATHS=tuple(sorted(set(bc.SOURCE_PATHS)|{
    'gaussian_tail.py','gaussian_tail_certificate.py','GAUSSIAN_TAIL.md',
    'experiment.py',FINITE_CERT,FINITE_RUN,'results/exact_h0_1/RUN.sha256'}))
SCOPE=('Planar side24 ideal Gaussian omitted-mode probability bound and conditional '
       'finite-polynomial coupling contract. Historical Gaussian sampling, '
       'coefficient coupling, lifetime-law and held-out confirmation remain uncertified.')
identity=bc.identity


def certify():
    fc.require(identity(BASE/FINITE_CERT)['sha256']==FINITE_SHA256,
               'Not the frozen C39 exact barcode certificate')
    fc.require(identity(BASE/FINITE_RUN)['sha256']==FINITE_RUN_SHA256,
               'Not the frozen C39 exact barcode receipt')
    finite=bc.validate_directory(BASE/'results/exact_h0_1')
    data=fc.load_json(BASE/'results/certificate8/COEFFICIENTS.json');fc.validate(data)
    row=next(r for r in data['records'] if r['seed']==finite['seed'])
    norm=abs(Q(row['dc'][0]))+sum((2*fc.sqrt_bounds(Q(a)**2+Q(b)**2)[1]
                                  for _,_,a,b in row['modes']),Q(0))
    return {'schema_version':1,'scope':SCOPE,
            'model':'normalized periodized Bargmann-Fock on (R/24Z)^2 with ideal independent standard Gaussian coefficients',
            'arithmetic':'Fraction; Machin pi128; alternating exponential with dyadic256 outward squaring',
            'tail_bounds':[gt.tail_bound(k,Q(t)) for k in (24,32) for t in (6,8)],
            'finite_object':{'source_sha256':FINITE_SHA256,'seed_label':finite['seed'],
                'cutoff':data['cutoff'],'norm_upper':str(norm),
                'norm_decimal_up':fc.decimal_up(norm,24),
                'finite_polynomial_diagram_bound':finite['finite_polynomial_diagram_bound'],
                'coupling_error':None,'infinite_field_diagram_bound':None,
                'reason':'No certified relation between the stored dyadic coefficients and ideal Gaussian low modes.'},
            'conditional_contract':'epsilon_total <= epsilon_finite + tau + rho + normalization_loss * norm(P)',
            'historical_gaussian_draw_certified':False,'lifetime_law_certified':False,
            'new_random_fields_sampled':0}


def verify(cert):
    fc.require(fc.canonical_bytes(cert)==fc.canonical_bytes(certify()),
               'Gaussian tail certificate differs from exact replay')
    return True


def render_report(cert):
    lines=['# A quantified omitted-mode bound', '',
           'For the **ideal normalized Gaussian field on the side-24 square torus**,',
           'the following uniform tail bounds hold outside the stated probability budget.',
           'Every mode outside the cutoff is included, including modes beyond64.', '',
           '| Square cutoff K | Rayleigh threshold t | Uniform tail at most | Failure probability at most |',
           '|---|---|---|---|']
    for r in cert['tail_bounds']:
        lines.append(f"| {r['cutoff']} | {r['threshold']} | {r['tail_supremum_decimal_up']} | {r['failure_probability_decimal_up']} |")
    lines += ['', 'The table concerns the ideal field and its coupled ideal low modes.',
              'K32 is a possible future truncation, not the stored K24 polynomial.',
              'The infinite versus mode64 normalization loss is below **3 × 10⁻⁶⁴**.', '',
              'To connect a certified finite polynomial P to this field, additionally prove',
              '`||F64,K - P||∞ <= rho` for the same low-mode Gaussian coefficients.',
              'Then the existing finite barcode bound composes with the tail as', '',
              '`epsilon_total <= epsilon_finite + tau + rho + normalization_loss * ||P||∞`.', '',
              'The retained seed34000 polynomial has a certified norm upper bound of',
              cert['finite_object']['norm_decimal_up']+'. Its coefficient-coupling error',
              '**rho is still unknown**, so no full-field error or Gaussian probability is',
              'attached to its barcode. A missing premise remains null, not zero.', '',
              '[Proof and exact conventions](../../GAUSSIAN_TAIL.md) ·',
              '[Exact certificate](CERTIFICATE.json) · [Source receipt](RUN.json)', '',
              '```sh',
              'python -B -S experiments/periodic_h0/gaussian_tail_certificate.py --verify experiments/periodic_h0/results/gaussian_tail1',
              '```', '', SCOPE, '']
    return '\n'.join(lines)


def verify_directory(directory):
    directory=Path(directory);r=fc.load_json(directory/'RUN.json')
    fc.require(type(r) is dict and set(r)=={'schema_version','utc','python','platform','sources','outputs','scope'},
               'Invalid receipt fields')
    fc.require(type(r['schema_version']) is int and r['schema_version']==1 and r['scope']==SCOPE,
               'Invalid receipt version/scope')
    fc.require(type(r['utc']) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00',r['utc']) is not None,
               'Invalid receipt UTC')
    stamp=datetime.fromisoformat(r['utc'])
    fc.require(stamp.tzinfo==timezone.utc and stamp.isoformat(timespec='microseconds')==r['utc'],'Invalid receipt UTC')
    for key in ('python','platform'):
        fc.require(type(r[key]) is str and 0<len(r[key])<=4096 and '\x00' not in r[key], 'Invalid environment metadata')
    fc.require(type(r['sources']) is dict and set(r['sources'])==set(SOURCE_PATHS),'Invalid source roles')
    fc.require(type(r['outputs']) is dict and set(r['outputs'])=={'CERTIFICATE.json'},'Invalid output roles')
    for item in [*r['sources'].values(),*r['outputs'].values()]:
        fc.require(type(item) is dict and set(item)=={'bytes','sha256'},'Invalid identity schema')
        fc.require(type(item['bytes']) is int and item['bytes']>0,'Invalid byte count')
        fc.require(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Invalid hash')
    fc.require((directory/'RUN.sha256').read_bytes()==(identity(directory/'RUN.json')['sha256']+'\n').encode(), 'Receipt binding mismatch')
    for name in SOURCE_PATHS:
        fc.require(identity(BASE/name)==r['sources'][name], 'Source binding mismatch: '+name)
    fc.require(identity(directory/'CERTIFICATE.json')==r['outputs']['CERTIFICATE.json'],'Output binding mismatch')
    cert=fc.load_json(directory/'CERTIFICATE.json')
    verify(cert)
    fc.require((directory/'RESULTS.md').read_bytes()==render_report(cert).encode(),'Report mismatch')
    return True


def produce(directory):
    directory=Path(directory);fc.require(not directory.exists(),'Output must be a new directory')
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
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify',type=Path);g.add_argument('--produce',type=Path)
    a=p.parse_args()
    if a.verify:
        verify_directory(a.verify)
        print('Exact ideal Gaussian tail replay passed; frozen-sample coupling remains open.')
    else:
        produce(a.produce)
        print('Ideal tail certificate produced; no Gaussian promotion of historical fields.')


if __name__=='__main__':main()
