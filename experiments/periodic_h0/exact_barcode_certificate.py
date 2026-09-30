"""Exact H0 endpoints and finite-field bin bounds for the frozen C38 grid."""
import argparse
import hashlib
import platform
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc
import nodal_core as nc
import hessian_grid_certificate as upstream
import exact_h0 as h0

BASE = Path(__file__).resolve().parent
UPSTREAM_CERT = 'results/hessian1/CERTIFICATE.json'
UPSTREAM_RUN = 'results/hessian1/RUN.json'
UPSTREAM_SHA256 = 'c46cfc061465b951968f675270b649cd6b61c1dc4989635a1c3c7a3f7a564519'
UPSTREAM_RUN_SHA256 = '3bea8119b5fc3b8aadf887b439d3032f7d88377771c338db44e320dfc60979e1'
SOURCE_PATHS = tuple(sorted(set(upstream.SOURCE_PATHS) | {
    'exact_h0.py', 'EXACT_H0.md', 'exact_barcode_certificate.py',
    UPSTREAM_CERT, UPSTREAM_RUN, 'results/hessian1/RUN.sha256'}))
SCOPE = ('Exact periodic vertex-cubical H0 barcode of the C38 dyadic grid and '
         'deterministic bin bounds for its one frozen finite polynomial. '
         'No historical FFT, ideal Gaussian ensemble, infinite-field, lifetime-law '
         'or held-out confirmation certificate.')


def identity(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_upstream(base=BASE):
    """Consume unchanged reviewed C38 bytes; do not re-author its certificate."""
    base = Path(base)
    fc.require(identity(base/UPSTREAM_CERT)['sha256'] == UPSTREAM_SHA256,
               'Not the frozen C38 certificate')
    fc.require(identity(base/UPSTREAM_RUN)['sha256'] == UPSTREAM_RUN_SHA256,
               'Not the frozen C38 receipt')
    receipt = fc.load_json(base/UPSTREAM_RUN)
    upstream.validate_receipt(receipt)
    fc.require((base/'results/hessian1/RUN.sha256').read_bytes() ==
               (UPSTREAM_RUN_SHA256+'\n').encode(), 'C38 receipt binding mismatch')
    for name, expected in receipt['sources'].items():
        fc.require(identity(base/name) == expected, 'Changed C38 dependency: '+name)
    fc.require(identity(base/UPSTREAM_CERT) == receipt['outputs']['CERTIFICATE.json'],
               'C38 output binding mismatch')
    return fc.load_json(base/UPSTREAM_CERT)


def analyze(values, n, scale, epsilon, edges):
    barcode = h0.compute(values, n)
    checked = h0.verify_by_connectivity(values, n, barcode)
    fc.require(checked is not False, 'Independent connectivity verification failed')
    bins = h0.bin_transfer(barcode, scale, epsilon, edges)
    # Endpoint integers are strings to preserve them in JSON consumers whose
    # number type cannot exactly represent 96-bit fixed-point samples.
    encoded = dict(barcode)
    encoded['intervals'] = [[str(b), str(d)] for b, d in barcode['intervals']]
    encoded['essential'] = [str(b) for b in barcode['essential']]
    return {'barcode': encoded,
            'bins': [{k: str(v) if type(v) is Q else v for k,v in row.items()} for row in bins]}


def certify():
    old = load_upstream()
    data = fc.load_json(BASE/upstream.COEFFICIENT_PATH)
    fc.validate(data)
    fc.require(fc.digest(data) == old['coefficient_object_sha256'], 'C38 coefficient object mismatch')
    record = next(r for r in data['records'] if r['seed'] == old['seed'])
    n = old['samples']['grid']
    centers, stages, error = nc.evaluate(record, n)
    fc.require(upstream.real_center_digest(centers) == old['samples']['real_integer_sha256'],
               'C38 real sample digest mismatch')
    fc.require(stages == old['samples']['stages'] and str(error) == old['samples']['nodal_error'],
               'C38 nodal replay mismatch')
    values = [v for v, _ in centers]
    del centers
    epsilon = Q(old['abstract_diagram_bound'])
    result = analyze(values, n, nc.SCALE, epsilon, [Q(2**j,1000) for j in range(9)])
    return dict(result, schema_version=1, seed=old['seed'], grid=n,
                fixed_point_scale=str(nc.SCALE), sample_count=len(values),
                sample_sha256=old['samples']['real_integer_sha256'],
                upstream_sha256=UPSTREAM_SHA256, computed_barcode_error='0',
                finite_polynomial_diagram_bound=str(epsilon),
                finite_polynomial_diagram_bound_decimal_up=fc.decimal_up(epsilon,24),
                independent_connectivity_check=True, historical_fft_certified=False,
                gaussian_law_certified=False, infinite_field_certified=False,
                lifetime_law_certified=False, scope=SCOPE)


def render_report(cert):
    lines = ['# Exact barcode and certified finite-field counts', '',
             f"One frozen finite polynomial (seed label {cert['seed']}), on the C38 **{cert['grid']}² exact dyadic grid**.", '',
             f"The integer-only sweep gives **{len(cert['barcode']['intervals'])} positive finite H₀ bars** and one essential class.",
             'A separate plateau and breadth-first connectivity check verifies the birth and death witnesses.',
             'Zero-lifetime pairs are omitted; the longest finite interval is retained.', '',
             f"The barcode endpoints have **zero arithmetic error** relative to these exact samples. The diagram bound",
             f"against the frozen smooth finite polynomial is at most **{cert['finite_polynomial_diagram_bound_decimal_up']}**",
             '(the unchanged C38 interpolation/stability bound).', '',
             '| Lifetime bin | Exact grid count | Smooth finite-field lower bound | Smooth finite-field upper bound |',
             '|---|---:|---:|---:|']
    for row in cert['bins']:
        upper = row['target_upper_count']
        lines.append(f"| [{row['lower']}, {row['upper']}) | {row['sample_count']} | {row['target_lower_count']} | {upper if upper is not None else 'not bounded by this argument'} |")
    lines += ['', 'Bounds use contracted and expanded bins with exact rational endpoints.',
              'They are counts for one deterministic smooth finite field, not expected intensities,',
              'a Gaussian-law fit, a tail enclosure, a remainder estimate or a held-out test.', '',
              '[Algorithm and proof](../../EXACT_H0.md) · [Exact endpoints and counts](CERTIFICATE.json)',
              '· [Source and execution receipt](RUN.json) · [C38 spatial bound](../hessian1/RESULTS.md)', '',
              '```sh', 'python -B -S experiments/periodic_h0/exact_barcode_certificate.py --verify experiments/periodic_h0/results/exact_h0_1',
              '```', '', SCOPE, '']
    return '\n'.join(lines)


def validate_directory(directory):
    """Check custody only; verify_directory additionally reruns the computation."""
    directory = Path(directory)
    receipt = fc.load_json(directory/'RUN.json')
    fc.require(type(receipt) is dict and set(receipt) ==
               {'schema_version','utc','python','platform','sources','outputs','scope'}, 'Invalid receipt fields')
    fc.require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1
               and receipt['scope'] == SCOPE, 'Invalid receipt version or scope')
    fc.require(type(receipt['utc']) is str and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00',receipt['utc']) is not None,
        'Invalid receipt timestamp')
    stamp = datetime.fromisoformat(receipt['utc'])
    fc.require(stamp.tzinfo == timezone.utc and stamp.isoformat(timespec='microseconds') == receipt['utc'],
               'Invalid receipt timestamp')
    for key in ('python','platform'):
        fc.require(type(receipt[key]) is str and 0 < len(receipt[key]) <= 4096
                   and '\x00' not in receipt[key], 'Invalid environment metadata')
    fc.require(type(receipt['sources']) is dict and set(receipt['sources']) == set(SOURCE_PATHS),
               'Invalid source roles')
    fc.require(type(receipt['outputs']) is dict and set(receipt['outputs']) == {'CERTIFICATE.json'},
               'Invalid output roles')
    for item in [*receipt['sources'].values(),*receipt['outputs'].values()]:
        fc.require(type(item) is dict and set(item) == {'bytes','sha256'}, 'Invalid identity fields')
        fc.require(type(item['bytes']) is int and item['bytes'] > 0, 'Invalid byte count')
        fc.require(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,
                   'Invalid SHA256')
    fc.require((directory/'RUN.sha256').read_bytes() ==
               (identity(directory/'RUN.json')['sha256']+'\n').encode(), 'Receipt binding mismatch')
    for name in SOURCE_PATHS:
        fc.require(identity(BASE/name) == receipt['sources'][name], 'Source binding mismatch: '+name)
    fc.require(identity(directory/'CERTIFICATE.json') == receipt['outputs']['CERTIFICATE.json'],
               'Output binding mismatch')
    cert = fc.load_json(directory/'CERTIFICATE.json')
    fc.require((directory/'RESULTS.md').read_bytes() == render_report(cert).encode(), 'Report binding mismatch')
    return cert


def verify_directory(directory):
    cert = validate_directory(directory)
    fc.require(fc.canonical_bytes(cert) == fc.canonical_bytes(certify()), 'Exact barcode replay mismatch')
    return True


def produce(directory):
    directory = Path(directory)
    fc.require(not directory.exists(), 'Output must be a new directory')
    cert = certify()
    directory.mkdir(parents=True)
    (directory/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(cert))
    receipt = {'schema_version':1, 'utc':datetime.now(timezone.utc).isoformat(timespec='microseconds'),
               'python':sys.version,'platform':platform.platform(),'scope':SCOPE,
               'sources':{name:identity(BASE/name) for name in SOURCE_PATHS},
               'outputs':{'CERTIFICATE.json':identity(directory/'CERTIFICATE.json')}}
    raw = fc.canonical_bytes(receipt)
    (directory/'RUN.json').write_bytes(raw)
    (directory/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    (directory/'RESULTS.md').write_text(render_report(cert))
    return cert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--produce',type=Path); group.add_argument('--verify',type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_directory(args.verify)
        print('Exact dyadic H0 replay and connectivity verification passed; Gaussian-law gates remain open.')
    else:
        result = produce(args.produce)
        print(len(result['barcode']['intervals']), 'positive finite bars; finite-field scope only.')


if __name__ == '__main__': main()
