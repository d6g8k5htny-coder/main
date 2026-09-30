"""Exact replay of newly recorded finite-polynomial nodal error."""
import argparse
import hashlib
import math
import re
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc
import nodal_core as core

BASE = Path(__file__).resolve().parent
COEFFICIENT_PATH = 'results/certificate8/COEFFICIENTS.json'
SOURCE_PATHS = ('nodal_core.py', 'nodal_certificate.py', 'run_nodal_certificate.py',
                'finite_certificate.py', 'APPROXIMATION.md', 'requirements.txt',
                COEFFICIENT_PATH)
SCOPE = {
    'execution': 'New NumPy ifft2(norm=forward) evaluation of exactly representable stored dyadic coefficients; real part recorded in row-major x-then-y order.',
    'historical_link': 'A new sample snapshot, not a receipt for pilot32 or refinement8 executions.',
    'scientific_effect': 'Finite rounded-polynomial nodal error and abstract filtration stability bound only. No Gaussian-law, infinite-field, historical FFT, computed barcode or lifetime-law certificate.'}


def check_selection(data, seed, grid):
    fc.validate(data)
    fc.require(type(seed) is int and seed in [r['seed'] for r in data['records']], 'Unknown seed label')
    fc.require(type(grid) is int and grid in data['grids'] and 2 <= grid <= 1024
               and grid & (grid-1) == 0, 'Unsupported or unlisted grid')
    return next(r for r in data['records'] if r['seed'] == seed)


def parse_samples(data, samples):
    fc.require(type(samples) is dict and set(samples) == {'schema_version', 'coefficient_object_sha256', 'seed', 'grid', 'layout', 'values'}, 'Unexpected samples schema')
    fc.require(type(samples['schema_version']) is int and samples['schema_version'] == 1, 'Unknown sample version')
    fc.require(samples['coefficient_object_sha256'] == fc.digest(data), 'Wrong coefficient object')
    record = check_selection(data, samples['seed'], samples['grid'])
    fc.require(samples['layout'] == 'row-major-x-then-y', 'Unexpected sample layout')
    values = samples['values']
    fc.require(type(values) is list and len(values) == samples['grid']**2, 'Incomplete sample array')
    exact = []
    for value in values:
        fc.require(type(value) is str and len(value) <= 32, 'Invalid float64 sample')
        try:
            f = float.fromhex(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError('Invalid float64 sample') from exc
        fc.require(math.isfinite(f) and f.hex() == value, 'Nonfinite or noncanonical float64 sample')
        exact.append(Q.from_float(f))
    return record, exact


def certify(data, samples):
    record, exact = parse_samples(data, samples)
    n = samples['grid']
    centers, stages, enclosure = core.evaluate(record, n)
    difference = max(abs(v-Q(z[0], core.SCALE)) for v, z in zip(exact, centers))
    eta = difference + enclosure
    derivative = fc.certify(data)
    row = next(r for r in derivative['records'] if r['seed'] == samples['seed'])
    spatial = Q(next(g['spatial_bound'] for g in row['grids'] if g['grid'] == n))
    return {'schema_version': 1, 'coefficient_object_sha256': fc.digest(data),
            'sample_object_sha256': fc.digest(samples), 'seed': samples['seed'], 'grid': n,
            'fixed_point_scale': str(core.SCALE),
            'center_object_sha256': fc.digest([[str(a), str(b)] for a, b in centers]),
            'stages': stages, 'center_error_bound': str(enclosure),
            'max_sample_center_difference': str(difference), 'nodal_error': str(eta),
            'spatial_bound': str(spatial), 'abstract_diagram_bound': str(eta+spatial),
            'nodal_error_decimal_up': fc.decimal_up(eta, 24),
            'abstract_diagram_bound_decimal_up': fc.decimal_up(eta+spatial, 18),
            'historical_fft_certified': False, 'continuum_field_certified': False,
            'gaussian_law_certified': False, 'persistence_implementation_certified': False,
            'computed_barcode_error': None, 'scope': SCOPE['scientific_effect']}


def verify(data, samples, certificate):
    fc.require(fc.canonical_bytes(certify(data, samples)) == fc.canonical_bytes(certificate), 'Nodal certificate differs from exact replay')
    return True


def file_identity(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def render_report(cert):
    return '\n'.join([
        '# A certified nodal error for one finite field', '',
        f"Seed label **{cert['seed']}**, grid **{cert['grid']}×{cert['grid']}**.", '',
        'This new snapshot records every real float64 FFT sample and compares it',
        'with an independent integer evaluator of the frozen exact finite polynomial.',
        'The replay uses only Python integers, rational arithmetic and exact parsing',
        'of the stored float64 hexadecimal values.', '',
        '| Bound | Certified upper endpoint |', '|---|---:|',
        f"| Maximum nodal error eta | {cert['nodal_error_decimal_up']} |",
        f"| Spatial interpolation budget B | {fc.decimal_up(Q(cert['spatial_bound']), 18)} |",
        f"| Abstract filtration diagram bound eta+B | {cert['abstract_diagram_bound_decimal_up']} |", '',
        'Decimal endpoints are rounded upward; exact rational values and all FFT',
        'stage error bounds are in [CERTIFICATE.json](CERTIFICATE.json). The last',
        'row applies the declared deterministic interpolation/stability argument',
        'to the mathematical filtration of these supplied samples. No persistence',
        'software or rounded barcode endpoint is certified by this calculation.', '',
        'This does not certify historical pilot/refinement samples, the infinite',
        'Gaussian field, its sampling law, a spectral tail or a lifetime asymptotic.',
        'The large spatial budget remains the limiting input on this coarse grid.', '',
        '[Derivation](../../NODAL_CERTIFICATE.md) · [Samples](SAMPLES.json) ·',
        '[Execution and source identity](RUN.json)', '',
        '```sh',
        'python -B -S experiments/periodic_h0/nodal_certificate.py --verify experiments/periodic_h0/results/nodal1',
        '```', ''])


def validate_receipt(receipt):
    expected = {'schema_version', 'utc', 'python', 'numpy', 'platform', 'sources', 'outputs'} | set(SCOPE)
    fc.require(type(receipt) is dict and set(receipt) == expected, 'Unexpected nodal receipt schema')
    fc.require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1, 'Unknown nodal receipt version')
    for k, v in SCOPE.items(): fc.require(receipt[k] == v, 'Unexpected receipt scope: '+k)
    utc = receipt['utc']
    fc.require(type(utc) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00', utc) is not None, 'Invalid receipt UTC')
    try: parsed = datetime.fromisoformat(utc)
    except ValueError as exc: raise ValueError('Invalid receipt UTC') from exc
    fc.require(parsed.tzinfo == timezone.utc and parsed.isoformat(timespec='microseconds') == utc, 'Invalid receipt UTC')
    for k in ('python', 'numpy', 'platform'):
        fc.require(type(receipt[k]) is str and 0 < len(receipt[k]) <= 4096 and '\x00' not in receipt[k], 'Invalid observed environment')
    fc.require(type(receipt['sources']) is dict and set(receipt['sources']) == set(SOURCE_PATHS), 'Unexpected receipt sources')
    fc.require(type(receipt['outputs']) is dict and set(receipt['outputs']) == {'SAMPLES.json', 'CERTIFICATE.json'}, 'Unexpected receipt outputs')


def verify_directory(directory):
    directory = Path(directory)
    receipt = fc.load_json(directory/'RUN.json')
    validate_receipt(receipt)
    fc.require((directory/'RUN.sha256').read_bytes() == (file_identity(directory/'RUN.json')['sha256']+'\n').encode('ascii'), 'Receipt byte identity mismatch')
    for filename in SOURCE_PATHS:
        fc.require(receipt['sources'][filename] == file_identity(BASE/filename), 'Source byte identity mismatch: '+filename)
    for filename in ('SAMPLES.json', 'CERTIFICATE.json'):
        fc.require(receipt['outputs'][filename] == file_identity(directory/filename), 'Output byte identity mismatch: '+filename)
    data = fc.load_json(BASE/COEFFICIENT_PATH)
    samples = fc.load_json(directory/'SAMPLES.json'); cert = fc.load_json(directory/'CERTIFICATE.json')
    verify(data, samples, cert)
    fc.require((directory/'RESULTS.md').read_bytes() == render_report(cert).encode('utf-8'), 'Report differs from certificate')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, required=True)
    args = parser.parse_args()
    verify_directory(args.verify)
    print('Exact finite-field nodal and abstract stability replay passed; computed barcodes and infinite-field gates remain open.')


if __name__ == '__main__': main()
