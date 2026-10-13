"""Compact exact replay of a sharper finite-field spatial and nodal bound."""
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
import hessian_grid_core as hc

BASE = Path(__file__).resolve().parent
COEFFICIENT_PATH = 'results/certificate8/COEFFICIENTS.json'
SOURCE_PATHS = ('hessian_grid_core.py', 'hessian_grid_certificate.py',
                'HESSIAN_GRID.md', 'nodal_core.py', 'finite_certificate.py',
                'NODAL_CERTIFICATE.md', 'FINITE_CERTIFICATE.md',
                'APPROXIMATION.md', COEFFICIENT_PATH)
SCOPE = ('Exact finite rounded polynomial and the abstract filtrations of '
         'deterministically regenerated dyadic center samples only. No historical '
         'FFT, computed barcode, Gaussian-law, infinite-field or lifetime-law certificate.')


def real_center_digest(centers):
    """Hash canonical JSON of decimal integer strings without a second array."""
    digest = hashlib.sha256(); digest.update(b'[')
    for j, (real, _) in enumerate(centers):
        if j: digest.update(b',')
        digest.update(('"'+str(real)+'"').encode('ascii'))
    digest.update(b']\n')
    return digest.hexdigest()


def bin_conditions(epsilon):
    fc.require(type(epsilon) is Q and epsilon >= 0, 'Nonnegative rational error required')
    edges = [Q(2**j, 1000) for j in range(9)]
    return [{'lower': str(a), 'upper': str(b), 'clean_upper': a > 2*epsilon,
             'nonempty_contraction': b-a > 4*epsilon}
            for a, b in zip(edges, edges[1:])]


def certify(data, seed, derivative_grid, sample_grid):
    fc.validate(data)
    fc.require(type(seed) is int and seed in [r['seed'] for r in data['records']], 'Unknown seed')
    for n in (derivative_grid, sample_grid):
        fc.require(type(n) is int and n in data['grids'] and 2 <= n <= 1024
                   and n & (n-1) == 0, 'Unsupported or unlisted grid')
    record = next(r for r in data['records'] if r['seed'] == seed)
    derivatives = hc.bound_record(record, data['side'], derivative_grid)
    centers, stages, error = nc.evaluate(record, sample_grid)
    samples = {'definition': 'real integer centers / fixed_point_scale; no float conversion',
               'grid': sample_grid, 'count': len(centers),
               'layout': 'row-major-x-then-y', 'fixed_point_scale': str(nc.SCALE),
               'real_integer_sha256': real_center_digest(centers),
               'nodal_error': str(error), 'stages': stages}
    spatial = Q(data['side']**2, sample_grid**2)*Q(derivatives['spatial_coefficient'])
    old = next(r for r in fc.certify(data)['records'] if r['seed'] == seed)
    old_spatial = Q(next(g['spatial_bound'] for g in old['grids'] if g['grid'] == sample_grid))
    epsilon = error+spatial
    return {'schema_version': 1, 'seed': seed, 'derivative_grid': derivative_grid,
            'coefficient_object_sha256': fc.digest(data), 'derivatives': derivatives,
            'samples': samples, 'spatial_bound': str(spatial),
            'previous_triangle_spatial_bound': str(old_spatial),
            'abstract_diagram_bound': str(epsilon),
            'spatial_bound_decimal_up': fc.decimal_up(spatial, 24),
            'nodal_error_decimal_up': fc.decimal_up(error, 30),
            'abstract_diagram_bound_decimal_up': fc.decimal_up(epsilon, 24),
            'bin_conditions': bin_conditions(epsilon), 'historical_fft_certified': False,
            'gaussian_law_certified': False, 'continuum_field_certified': False,
            'persistence_implementation_certified': False, 'computed_barcode_error': None,
            'scope': SCOPE}


def verify(data, certificate):
    fc.require(type(certificate) is dict and type(certificate.get('samples')) is dict,
               'Invalid grid certificate')
    expected = certify(data, certificate.get('seed'), certificate.get('derivative_grid'),
                       certificate['samples'].get('grid'))
    fc.require(fc.canonical_bytes(certificate) == fc.canonical_bytes(expected),
               'Hessian/grid certificate differs from exact replay')
    return True


def file_identity(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def render_report(cert):
    rows = ['# A smaller certified spatial error', '',
            f"Frozen finite field seed **{cert['seed']}**; Hessian grid **{cert['derivative_grid']}²**;",
            f"deterministically defined dyadic sample grid **{cert['samples']['grid']}²**.", '',
            '| Quantity | Certified upward endpoint |', '|---|---:|',
            f"| Global Hessian operator norm | {fc.decimal_up(Q(cert['derivatives']['H']), 18)} |",
            f"| Earlier coefficient-triangle spatial bound | {fc.decimal_up(Q(cert['previous_triangle_spatial_bound']), 24)} |",
            f"| New spatial bound B | {cert['spatial_bound_decimal_up']} |",
            f"| Nodal error eta | {cert['nodal_error_decimal_up']} |",
            f"| Abstract filtration bound eta+B | {cert['abstract_diagram_bound_decimal_up']} |", '',
            'Every sample is the exact real integer evaluator center divided by 2^96.',
            'Replay regenerates all samples and checks their ordered digest; the large',
            'array is not stored. These are new mathematical samples, not the historical',
            'NumPy FFT arrays, and no persistence software or computed barcode is certified.', '',
            'The following conditions concern the proved abstract-filtration bin sandwich.',
            'They do not provide counts, confidence intervals or an asymptotic window.', '',
            '| Lifetime bin | Clean upper condition a > 2 epsilon | Nonempty contracted bin |',
            '|---|---|---|']
    for row in cert['bin_conditions']:
        rows.append(f"| [{row['lower']}, {row['upper']}) | {'yes' if row['clean_upper'] else 'no'} | {'yes' if row['nonempty_contraction'] else 'no'} |")
    rows += ['', SCOPE, '', '[Derivation](../../HESSIAN_GRID.md) ·',
             '[Exact certificate](CERTIFICATE.json) · [Execution/source receipt](RUN.json)', '',
             '```sh',
             'python -B -S experiments/periodic_h0/hessian_grid_certificate.py --verify experiments/periodic_h0/results/hessian1',
             '```', '']
    return '\n'.join(rows)


def validate_receipt(receipt):
    fc.require(type(receipt) is dict and set(receipt) ==
               {'schema_version','utc','python','platform','sources','outputs','scope'}, 'Unexpected receipt schema')
    fc.require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1
               and receipt['scope'] == SCOPE, 'Unexpected receipt version/scope')
    utc = receipt['utc']
    fc.require(type(utc) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00', utc) is not None, 'Invalid receipt UTC')
    try: stamp = datetime.fromisoformat(utc)
    except ValueError as exc: raise ValueError('Invalid receipt UTC') from exc
    fc.require(stamp.tzinfo == timezone.utc and stamp.isoformat(timespec='microseconds') == utc, 'Invalid receipt UTC')
    for key in ('python', 'platform'):
        fc.require(type(receipt[key]) is str and 0 < len(receipt[key]) <= 4096
                   and '\x00' not in receipt[key], 'Invalid observed environment')
    fc.require(type(receipt['sources']) is dict and set(receipt['sources']) == set(SOURCE_PATHS), 'Unexpected source roles')
    fc.require(type(receipt['outputs']) is dict and set(receipt['outputs']) == {'CERTIFICATE.json'}, 'Unexpected output roles')
    for item in [*receipt['sources'].values(), *receipt['outputs'].values()]:
        fc.require(type(item) is dict and set(item) == {'bytes','sha256'}, 'Invalid identity schema')
        fc.require(type(item['bytes']) is int and item['bytes'] > 0, 'Invalid integer byte count')
        fc.require(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None, 'Invalid identity hash')


def verify_directory(directory):
    directory = Path(directory); receipt = fc.load_json(directory/'RUN.json')
    validate_receipt(receipt)
    fc.require((directory/'RUN.sha256').read_bytes() ==
               (file_identity(directory/'RUN.json')['sha256']+'\n').encode('ascii'), 'Receipt byte binding mismatch')
    for name in SOURCE_PATHS:
        fc.require(receipt['sources'][name] == file_identity(BASE/name), 'Source identity mismatch: '+name)
    fc.require(receipt['outputs']['CERTIFICATE.json'] == file_identity(directory/'CERTIFICATE.json'), 'Output identity mismatch')
    cert = fc.load_json(directory/'CERTIFICATE.json')
    verify(fc.load_json(BASE/COEFFICIENT_PATH), cert)
    fc.require((directory/'RESULTS.md').read_bytes() == render_report(cert).encode(), 'Report differs from exact certificate')
    return True


def produce(directory, *, seed=34000, derivative_grid=256, sample_grid=1024):
    directory = Path(directory)
    fc.require(not directory.exists(), 'Output must be a new directory')
    cert = certify(fc.load_json(BASE/COEFFICIENT_PATH), seed, derivative_grid, sample_grid)
    directory.mkdir(parents=True)
    (directory/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(cert))
    receipt = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
               'python': sys.version, 'platform': platform.platform(), 'scope': SCOPE,
               'sources': {name:file_identity(BASE/name) for name in SOURCE_PATHS},
               'outputs': {'CERTIFICATE.json':file_identity(directory/'CERTIFICATE.json')}}
    raw = fc.canonical_bytes(receipt); (directory/'RUN.json').write_bytes(raw)
    (directory/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    (directory/'RESULTS.md').write_bytes(render_report(cert).encode())
    return cert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify',type=Path);group.add_argument('--produce',type=Path)
    args=parser.parse_args()
    if args.verify:
        verify_directory(args.verify)
        print('Exact finite Hessian/grid replay passed; computed barcode and infinite-field gates remain open.')
    else:
        print(produce(args.produce)['abstract_diagram_bound_decimal_up'])


if __name__=='__main__':main()
