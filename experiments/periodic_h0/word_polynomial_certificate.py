"""Replay explicit finite words through a smooth polynomial and exact H0 bounds."""
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
import word_polynomial as wp

BASE=Path(__file__).resolve().parent
PARENT_CERT='results/gaussian_coupling1/CERTIFICATE.json'
PARENT_RUN='results/gaussian_coupling1/RUN.json'
PARENT_SHA256='1ab5e225ec3e0a570910afc11026adf38ce39d01a4a8a16caa9ad487c95e1e32'
PARENT_RUN_SHA256='dbc5b27e6ea7e147133b48194889599a7fe1471bdb218c0513cb1ccb96a77d61'
SOURCE_PATHS=tuple(sorted(set(prior.SOURCE_PATHS)|{
    'word_polynomial.py','word_polynomial_certificate.py','WORD_POLYNOMIAL.md',
    'nodal_core.py','NODAL_CERTIFICATE.md','hessian_grid_core.py','HESSIAN_GRID.md',
    'exact_h0.py','EXACT_H0.md','APPROXIMATION.md',PARENT_CERT,PARENT_RUN,
    'results/gaussian_coupling1/RUN.sha256'}))
SCOPE=('One deterministic cutoff 1 side 24 finite-word polynomial: exact coefficient '
       'conversion, certified grid/Hessian error and finite H0 lifetime-bin counts. '
       'No IID source law, Gaussian observation, infinite-field diagram bound, '
       'historical coupling or lifetime-law confirmation certified.')
identity=prior.identity


def load_parent():
    """Consume fixed reviewed bytes; unchanged parent replay remains in master CI."""
    fc.require(identity(BASE/PARENT_CERT)['sha256']==PARENT_SHA256,'Frozen sampler certificate changed')
    fc.require(identity(BASE/PARENT_RUN)['sha256']==PARENT_RUN_SHA256,'Frozen sampler receipt changed')
    r=fc.load_json(BASE/PARENT_RUN)
    # The exact receipt hash fixes its schema and every role; also bind all bytes
    # it consumes. We do not reinterpret or promote its probabilistic premises.
    fc.require(set(r['sources'])==set(prior.SOURCE_PATHS),'Parent source roles changed')
    fc.require((BASE/'results/gaussian_coupling1/RUN.sha256').read_bytes()==
               (PARENT_RUN_SHA256+'\n').encode(),'Parent receipt digest changed')
    for name,expected in r['sources'].items():
        fc.require(identity(BASE/name)==expected,'Changed sampler dependency: '+name)
    fc.require(identity(BASE/PARENT_CERT)==r['outputs']['CERTIFICATE.json'],'Parent output changed')


def certify():
    load_parent()
    words=[j*((1<<128)-1)//17 for j in (1,13,4,16,7,11,2,15,9)]
    edges=[Q(2**j,1000) for j in range(9)]
    inp={'kind':'deterministic arithmetic fixture; no RNG used','record_id':0,
         'cutoff':1,'side':24,'grid':128,'hessian_grid':32,
         'words':[str(x) for x in words],
         'word_order':'DC then cosine,sine for each ordered half-square mode',
         'bin_edges':[str(x) for x in edges]}
    p=gc.polynomial(1,words)
    real={'dc':str(p['dc']),'modes':[[x,y,str(a),str(b)] for x,y,a,b in p['modes']]}
    result=wp.analyze(p,1,128,32,edges)
    return {'schema_version':1,'scope':SCOPE,'input':inp,'input_sha256':fc.digest(inp),
            'real_polynomial':real,'real_polynomial_sha256':fc.digest(real),
            'parent_certificate_sha256':PARENT_SHA256,'parent_receipt_sha256':PARENT_RUN_SHA256,
            'result':result,'iid_input_law_certified':False,'new_gaussian_barcodes':0,
            'new_deterministic_barcodes':1,'historical_objects_unchanged':True,
            'next_obligation':'A declared ensemble at a matching cutoff, with source-law premises and exceptional-count control, is still needed for statistical or infinite-field conclusions.'}


def verify(cert):
    fc.require(fc.canonical_bytes(cert)==fc.canonical_bytes(certify()),'Word-polynomial certificate differs from exact replay')
    return True


def render_report(c):
    r=c['result']
    lines=['# From explicit words to a certified smooth-field barcode','',
           'This deterministic example connects the new finite-word sampler to',
           'exact Fourier-grid evaluation and periodic H₀ persistence. It uses',
           'nine declared words, cutoff 1, a 128×128 sample grid and a 32×32',
           'Hessian grid. No random source is used.','',
           f"The exact grid has **{len(r['barcode']['intervals'])} positive finite bars** and one essential class.",
           'A separate breadth-first connectivity calculation verifies the barcode.',
           'Its endpoints are exact for the computed dyadic samples.','',
           f"The diagram bound against the original smooth finite polynomial is at most **{r['finite_polynomial_diagram_bound_decimal_up']}**.",
           'The bounds below account for both nodal arithmetic and spatial interpolation.','',
           '| Lifetime bin | Exact grid count | Smooth polynomial lower count | Smooth polynomial upper count |',
           '|---|---:|---:|---:|']
    for row in r['bins']:
        upper=row['target_upper_count']
        lines.append(f"| [{row['lower']}, {row['upper']}) | {row['sample_count']} | {row['target_lower_count']} | {upper if upper is not None else 'not bounded'} |")
    lines+=['','Thus the smooth finite polynomial has **exactly one finite bar in',
            '[1/125,2/125)** and none in the other seven displayed bins. This does',
            'not exclude bars below 1/1000 or above 32/125. The two smaller positive',
            'grid bars are below twice the diagram error and are not certified',
            'smooth-field detections. Counts here are for this one polynomial,',
            'not expected intensities or evidence for a Gaussian lifetime law.','',
            'The real Fourier coefficients are converted without rounding. An',
            'exact positive power of two lifts them into the existing fixed-point',
            'evaluator; sample units and all error bounds are divided back by',
            'that same factor. The 96-bit twiddle accuracy is unchanged. The',
            '`seed:0` field in the compatibility record is only record ID 0.','',
            '[Derivation and scope](../../WORD_POLYNOMIAL.md) ·',
            '[Exact inputs, endpoints and bounds](CERTIFICATE.json) ·',
            '[Source receipt](RUN.json)','',
            '```sh','python -B -S experiments/periodic_h0/word_polynomial_certificate.py --verify experiments/periodic_h0/results/word_polynomial1','```','',SCOPE,'']
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
    print('Exact word-to-polynomial H0 replay passed; deterministic finite-polynomial scope only.')


if __name__=='__main__':main()
