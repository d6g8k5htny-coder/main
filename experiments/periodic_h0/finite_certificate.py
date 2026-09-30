"""Stdlib exact replay of derivative bounds for defined dyadic Fourier fields.

This module certifies no FFT evaluation, Gaussian sampling law, spectral tail,
or persistence implementation. Its input coefficients define the finite object.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

BITS = 128
MODEL = 'exact_dyadic_rounded_coefficients'
SOURCE_FILES = {'core': 'finite_certificate.py', 'extractor': 'run_certificate.py',
                'generator': 'experiment.py', 'configuration': 'refinement_config.json',
                'dependencies': 'requirements.txt'}
RECEIPT_TEXT = {
    'extraction': 'Actual field_grid coefficient array captured at the ifft2 input for n=64, cutoff24, default model factors. FFT replaced by a zero return only during capture; no numerical field evaluation claimed.',
    'historical_link': 'Same source configuration and seed labels as refinement8, deterministically reconstructed now. Historical executions did not archive their coefficient arrays; this is not a retroactive coefficient receipt.',
    'seed_meaning': 'Reconstruction labels, not a certificate of Gaussian distribution or independence.',
    'scientific_effect': 'Finite rounded-polynomial derivative certificate only; no lifetime-law acceptance.'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject_constant)


def rational(value):
    require(type(value) is str, 'Coefficients must be canonical rational strings')
    try:
        q = Q(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid rational coefficient') from exc
    require(str(q) == value, 'Noncanonical rational coefficient')
    require(q.denominator & (q.denominator - 1) == 0, 'Coefficient must be dyadic')
    return q


def sqrt_bounds(q, bits=BITS):
    require(type(q) is Q and q >= 0, 'Nonnegative rational radicand required')
    require(type(bits) is int and bits > 0, 'Positive precision required')
    scale = 1 << bits
    floor = isqrt((q.numerator * scale * scale) // q.denominator)
    exact = floor * floor * q.denominator == q.numerator * scale * scale
    return Q(floor, scale), Q(floor if exact else floor + 1, scale)


def _atan_bounds(denominator, terms):
    total = sum((Q((-1)**j, (2*j+1)*denominator**(2*j+1))
                 for j in range(terms)), Q(0))
    remainder = Q(1, (2*terms+1)*denominator**(2*terms+1))
    return (total, total + remainder) if terms % 2 == 0 else (total - remainder, total)


def pi_bounds():
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239). Alternating remainders.
    a, b = _atan_bounds(5, 64)
    c, d = _atan_bounds(239, 24)
    lo, hi = 16*a - 4*d, 16*b - 4*c
    scale = 1 << BITS
    lower = (lo * scale).numerator // (lo * scale).denominator
    upper = -((-hi * scale).numerator // (-hi * scale).denominator)
    return Q(lower, scale), Q(upper, scale)


def decimal_up(q, digits=15):
    require(type(q) is Q and q >= 0, 'Nonnegative rational required')
    require(type(digits) is int and digits > 0, 'Positive decimal precision required')
    scale = 10**digits
    integer = -((-q * scale).numerator // (-q * scale).denominator)
    return f'{integer // scale}.{integer % scale:0{digits}d}'


def validate(data):
    require(type(data) is dict and set(data) == {'schema_version', 'model', 'side', 'cutoff', 'grids', 'records'}, 'Unexpected coefficient schema')
    require(type(data['schema_version']) is int and data['schema_version'] == 1 and data['model'] == MODEL, 'Unknown coefficient model')
    require(type(data['side']) is int and data['side'] == 24, 'Only side24 supported')
    k = data['cutoff']
    require(type(k) is int and 1 <= k <= 64, 'Invalid finite cutoff')
    grids = data['grids']
    require(type(grids) is list and grids and all(type(n) is int and n > 2*k for n in grids) and grids == sorted(set(grids)), 'Invalid grids')
    expected = [(x, y) for x in range(k+1) for y in range(-k, k+1) if x > 0 or y > 0]
    records = data['records']; seeds = set()
    require(type(records) is list and records, 'No coefficient records')
    for row in records:
        require(type(row) is dict and set(row) == {'seed', 'dc', 'modes'}, 'Unexpected coefficient record')
        seed = row['seed']
        require(type(seed) is int and seed >= 0 and seed not in seeds, 'Invalid/duplicate seed label')
        seeds.add(seed)
        dc = row['dc']
        require(type(dc) is list and len(dc) == 2 and rational(dc[1]) == 0, 'DC must be real')
        rational(dc[0])
        modes = row['modes']
        require(type(modes) is list and len(modes) == len(expected), 'Incomplete finite mode set')
        for mode, pair in zip(modes, expected):
            require(type(mode) is list and len(mode) == 4, 'Malformed mode')
            require(type(mode[0]) is int and type(mode[1]) is int and tuple(mode[:2]) == pair, 'Mode convention, ordering or completeness violation')
            rational(mode[2]); rational(mode[3])
    require([r['seed'] for r in records] == sorted(seeds), 'Seed labels must be ordered')


def certify(data):
    validate(data)
    pi_lo, pi_hi = pi_bounds()
    frequency2 = (2*pi_hi / data['side'])**2
    rows = []
    for source in data['records']:
        amplitude = xx = yy = xy = Q(0)
        for x, y, real, imag in source['modes']:
            a, b = rational(real), rational(imag)
            rho = 2*sqrt_bounds(a*a+b*b)[1]
            amplitude += rho
            xx += rho*x*x; yy += rho*y*y; xy += rho*abs(x*y)
        bounds = {'amplitude_sum': amplitude, 'Mxx': frequency2*xx,
                  'Myy': frequency2*yy, 'Mxy': frequency2*xy,
                  'H': frequency2*(xx+yy)}
        grids = []
        for n in data['grids']:
            h2 = Q(data['side']**2, n*n)
            operator = h2*bounds['H']/4
            component = h2*(bounds['Mxx']+bounds['Myy'])/8 + h2*bounds['Mxy']/4
            spatial = min(operator, component)
            grids.append({'grid': n, 'operator_spatial_bound': str(operator),
                          'component_spatial_bound': str(component),
                          'spatial_bound': str(spatial), 'spatial_bound_decimal_up': decimal_up(spatial),
                          'nodal_error': None, 'diagram_error': None})
        rows.append({'seed': source['seed'], 'mode_pairs': len(source['modes']),
                     'bounds': {key: str(q) for key, q in bounds.items()},
                     'bounds_decimal_up': {key: decimal_up(q) for key, q in bounds.items()},
                     'grids': grids})
    return {'schema_version': 1, 'model': MODEL, 'coefficient_object_sha256': digest(data),
            'arithmetic': 'exact_rational_machin_pi_and_integer_square_root',
            'precision_bits': BITS, 'pi_interval': [str(pi_lo), str(pi_hi)],
            'records': rows, 'historical_fft_certified': False,
            'continuum_field_certified': False, 'gaussian_law_certified': False,
            'persistence_implementation_certified': False,
            'scope': 'Derivative bounds for exactly defined rounded finite polynomials. Spatial budget B only; nodal error eta is unknown. No diagram/count, spectral-tail or asymptotic-coefficient certificate.'}


def verify(data, certificate):
    require(canonical_bytes(certify(data)) == canonical_bytes(certificate), 'Certificate differs from exact replay')
    return True


def validate_receipt(receipt):
    expected = {'schema_version', 'utc', 'python', 'numpy', 'platform', 'sources', 'outputs'} | set(RECEIPT_TEXT)
    require(type(receipt) is dict and set(receipt) == expected, 'Unexpected receipt schema or fields')
    require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1, 'Unknown receipt schema version')
    for key, value in RECEIPT_TEXT.items():
        require(receipt[key] == value, 'Unexpected receipt scope: ' + key)
    value = receipt['utc']
    require(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00', value) is not None, 'Noncanonical receipt UTC')
    try:
        when = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('Invalid receipt UTC') from exc
    require(when.tzinfo == timezone.utc and when.isoformat(timespec='microseconds') == value, 'Invalid receipt UTC')
    python = receipt['python']; numpy = receipt['numpy']; platform = receipt['platform']
    require(type(python) is str and 0 < len(python) <= 4096 and '\x00' not in python
            and re.fullmatch(r'\d+\.\d+\.\d+[a-zA-Z0-9.+-]*(?:[ \n].*)?', python, re.DOTALL) is not None, 'Invalid observed Python environment')
    require(type(numpy) is str and len(numpy) <= 128 and re.fullmatch(r'\d+\.\d+\.\d+[a-zA-Z0-9.+-]*', numpy) is not None, 'Invalid observed NumPy environment')
    require(type(platform) is str and 0 < len(platform) <= 4096 and re.fullmatch(r'[A-Za-z0-9_.()+-]+', platform) is not None, 'Invalid observed platform environment')
    sources, outputs = receipt['sources'], receipt['outputs']
    require(type(sources) is dict and set(sources) == set(SOURCE_FILES), 'Unexpected source roles')
    require(type(outputs) is dict and set(outputs) == {'COEFFICIENTS.json', 'CERTIFICATE.json'}, 'Unexpected output roles')
    for role, item in sources.items():
        require(type(item) is dict and set(item) == {'filename', 'bytes', 'sha256'} and item['filename'] == SOURCE_FILES[role], 'Unexpected source metadata: ' + role)
    for role, item in outputs.items():
        require(type(item) is dict and set(item) == {'bytes', 'sha256'}, 'Unexpected output metadata: ' + role)
    for item in [*sources.values(), *outputs.values()]:
        require(type(item['bytes']) is int and item['bytes'] > 0, 'Invalid receipt byte count')
        require(type(item['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is not None, 'Invalid receipt SHA-256')
    return True


def render_report(certificate):
    finest = max(g['grid'] for g in certificate['records'][0]['grids'])
    lines = ['# Exact finite-polynomial derivative bounds', '',
             'These numbers enclose the derivative majorants of explicitly stored',
             'rounded finite Fourier polynomials. They do **not** certify historical FFT',
             'samples, the ideal Gaussian sampling law, the infinite field, or bar counts.',
             'Every displayed upper bound is rounded upward to 15 decimal places.', '',
             f'| Seed | Hessian operator majorant H | Component interpolation budget B at {finest}² |',
             '|---|---:|---:|']
    worst = Q(0)
    for row in certificate['records']:
        grid = next(g for g in row['grids'] if g['grid'] == finest)
        worst = max(worst, Q(grid['spatial_bound']))
        lines.append(f"| {row['seed']} | {row['bounds_decimal_up']['H']} | {grid['spatial_bound_decimal_up']} |")
    lines += ['', 'The componentwise interpolation bound is no larger than the operator bound',
              'for these inputs. `B` is only the spatial part of `epsilon = eta + B`:',
              'the nodal error `eta` is not supplied and is stored as null. Consequently',
              'the diagram-error field is also null. Setting either value to zero would',
              'change the claim and fails the exact replay.', '',
              'With *hypothetical exact nodal samples*, the largest certified spatial',
              f'budget at {finest}² is at most {decimal_up(worst)}. The clean bin upper',
              'bound would then require `a > 2 B`. This conditional observation does not',
              'certify any historical bin or select an asymptotic confirmation window.', '',
              '[Derivation and limitations](../../FINITE_CERTIFICATE.md) ·',
              '[Exact coefficients](COEFFICIENTS.json) · [Certificate](CERTIFICATE.json) ·',
              '[Execution and source identities](RUN.json)', '',
              'Replay from the repository root without numerical dependencies:', '',
              '```sh',
              'python -B -S experiments/periodic_h0/finite_certificate.py --verify experiments/periodic_h0/results/certificate8',
              '```', '']
    return '\n'.join(lines)


def verify_directory(directory):
    directory = Path(directory)
    data = load_json(directory/'COEFFICIENTS.json')
    certificate = load_json(directory/'CERTIFICATE.json')
    receipt = load_json(directory/'RUN.json')
    validate_receipt(receipt)
    binding = directory/'RUN.sha256'
    require(binding.is_file(), 'Missing receipt byte binding')
    expected_binding = (hashlib.sha256((directory/'RUN.json').read_bytes()).hexdigest()+'\n').encode('ascii')
    require(binding.read_bytes() == expected_binding, 'Complete receipt byte identity mismatch')
    verify(data, certificate)
    for filename in ('COEFFICIENTS.json', 'CERTIFICATE.json'):
        content = (directory/filename).read_bytes()
        require(receipt['outputs'][filename] == {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}, 'Output byte identity mismatch')
    base = Path(__file__).resolve().parent
    for role, filename in SOURCE_FILES.items():
        content = (base/filename).read_bytes()
        require(receipt['sources'][role] == {'filename': filename, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}, 'Source byte identity mismatch: ' + role)
    config = load_json(base/'refinement_config.json')
    require(data['side'] == config['side'] and data['cutoff'] == config['cutoff'] and data['grids'] == config['grids'] and [r['seed'] for r in data['records']] == config['seeds'], 'Configuration identity mismatch')
    require((directory/'RESULTS.md').read_bytes() == render_report(certificate).encode(), 'Rendered report differs from certificate')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, required=True)
    args = parser.parse_args()
    verify_directory(args.verify)
    print('Exact finite-polynomial derivative certificate replay passed; nodal/FFT/tail gates remain open.')


if __name__ == '__main__':
    main()
