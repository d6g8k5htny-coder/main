"""Exploratory pilot of fifty declared finite laws, with exact grid H0 checks.

Only the integer quantized vertex filtration has exact endpoints here. The
floating generator, IID premise, finite-polynomial/continuum error, asymptotic
window and theorem applicability are unverified. There is no blind-run option.
The CLI preserves completed generation before final record verification; it
does not provide crash recovery or checkpoints for partial generation.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.universality import models
from experiments.periodic_h0 import exact_h0


EXPLORATORY_NAMESPACE = 'universality/exploratory/v1'
DEFAULT_EDGES = [Q(1, 64), Q(1, 32), Q(1, 16), Q(1, 8), Q(1, 4),
                 Q(1, 2), Q(1), Q(2), Q(4)]
BIN_CONVENTION = 'Every bin half open [a,b), including the final bin'
QUANTIZATION = 'round(float_value*scale), nearest with ties to even; no continuum error certificate'
SAMPLER_DESCRIPTIONS = {
    'declared_float_sampler': 'random.Random MT19937; 52-bit open dyadic uniforms; float Box-Muller and libm',
    'injected_uncertified_sampler': 'Caller-supplied sampler; source, law and randomness unverified; arithmetic/test input only',
}
SCOPES = [
    'Exploratory public deterministic seeds; no held-out field or blind confirmation was consumed',
    'Square two-dimensional torus; finite square Fourier cutoff; normalized ideal variance one',
    'Python PRNG and floating transcendental functions are not a certified IID continuous sampler',
    'Exact barcode and BFS verification refer only to retained integer quantized vertex samples',
    'No nodal error enclosure, continuum persistence transfer, model admissibility or coefficient is supplied',
    'No slope, asymptotic window, scientific confirmation or observed confidence interval is claimed',
    'Verifier checks retained grids and derived records; a fresh simulation is required to test generator replay',
    'Injected-sampler records are arithmetic/test inputs, not executions of the declared candidate-law generator',
]


def _strict_positive(value, name):
    models.require(type(value) is int and value > 0, name+' must be a positive strict integer')


def _qtext(value):
    return str(value)


def seed_for(model_id, replicate):
    models.require(type(model_id) is str and type(replicate) is int and replicate >= 0,
                   'Invalid exploratory field identity')
    payload = f'{EXPLORATORY_NAMESPACE}/{model_id}/{replicate}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')


def bin_counts(intervals, scale, edges):
    _strict_positive(scale, 'Scale')
    models.require(type(edges) in (tuple, list) and len(edges) >= 2
                   and all(type(x) is Q and x > 0 for x in edges)
                   and all(a < b for a, b in zip(edges, edges[1:])),
                   'Positive increasing rational half-open bins required')
    models.require(type(intervals) is list and all(type(p) is list and len(p) == 2
                   and all(type(x) is int for x in p) and p[0] > p[1] for p in intervals),
                   'Strict positive integer barcode endpoints required')
    lengths = [Q(b-d, scale) for b, d in intervals]
    return [sum(a <= length < b for length in lengths) for a, b in zip(edges, edges[1:])]


def summarize_rows(rows, bins, *, side, vertex_count):
    """Bounds on the planned quantized sample average, retaining failed rows.

    These are deterministic unresolved-computation bounds, not confidence
    intervals or finite-polynomial/continuum expected-count bounds.
    """
    _strict_positive(bins, 'Bin count')
    _strict_positive(vertex_count, 'Vertex count')
    models.require(type(rows) is list and len(rows) > 0, 'All planned rows must be present')
    models.require(type(side) in (int, float) and math.isfinite(side) and side > 0,
                   'Positive finite side required')
    totals, failed = [0]*bins, 0
    for row in rows:
        if row['failure'] is not None:
            models.require(row['counts'] is None, 'A failed computation cannot report certified bin counts')
            failed += 1
        else:
            c = row['counts']
            models.require(type(c) is list and len(c) == bins
                           and all(type(x) is int and 0 <= x <= vertex_count-1 for x in c),
                           'Invalid quantized-grid count row')
            totals = [a+b for a, b in zip(totals, c)]
    area = Q(str(side))**2
    denominator = len(rows)*area
    upper_extra = failed*(vertex_count-1)
    return {'planned_fields': len(rows), 'successful_fields': len(rows)-failed,
            'failed_fields': failed, 'successful_bin_totals': totals,
            'mean_mass_bounds': [[_qtext(Q(t)/denominator),
                                  _qtext(Q(t+upper_extra)/denominator)] for t in totals],
            'meaning': 'Deterministic bounds on this complete planned quantized-grid sample average per area; no ensemble confidence claim'}


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ['experiments/universality/models.json', 'experiments/universality/models.py',
             'experiments/universality/run_pilot.py', 'experiments/periodic_h0/exact_h0.py']
    return {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}


def _validate_plan(*, n=16, cutoff=3, fields_per_model=2, scale=65536,
                   side=24, edges=None):
    """Validate the actual plan before generation or CLI custody reservation."""
    _strict_positive(fields_per_model, 'Fields per model')
    _strict_positive(scale, 'Quantization scale')
    models.require(type(n) is int and n >= 3 and type(cutoff) is int and n > 2*cutoff,
                   'Grid must satisfy n > 2K')
    models.normalized_spectrum('gaussian', cutoff)
    models.require(type(side) in (int, float) and math.isfinite(side) and side > 0,
                   'Positive finite torus side required')
    frozen_edges = list(DEFAULT_EDGES) if edges is None else list(edges)
    bin_counts([], scale, frozen_edges)
    return frozen_edges


def build_observations(*, n=16, cutoff=3, fields_per_model=2, scale=65536,
                       side=24, edges=None, sampler=None):
    """Generate retained rows and validate the record; propagate final errors.

    The CLI uses the internal constructor to save generated bytes before the
    final record check. Library callers retain this checked-return contract.
    """
    output = _generate_observations(n=n, cutoff=cutoff, fields_per_model=fields_per_model,
                                    scale=scale, side=side, edges=edges, sampler=sampler)
    verify_observations(output)
    return output


def _generate_observations(*, n=16, cutoff=3, fields_per_model=2, scale=65536,
                           side=24, edges=None, sampler=None):
    """Construct the full retained record; final consistency is not yet checked.

    Ordinary per-field exceptions stay in their rows. The completed record is
    returned for CLI custody; interrupted or partial generation is not saved.
    """
    edges = _validate_plan(n=n, cutoff=cutoff, fields_per_model=fields_per_model,
                           scale=scale, side=side, edges=edges)
    catalog = models.load_catalog()
    sampler_mode = 'declared_float_sampler' if sampler is None else 'injected_uncertified_sampler'
    sampler = models.sample_grid if sampler is None else sampler
    definitions = [models.definition(model, cutoff=cutoff, side=side, catalog=catalog)
                   for model in catalog['models']]
    rows = []
    for model, definition in zip(catalog['models'], definitions):
        for replicate in range(fields_per_model):
            seed = seed_for(model['id'], replicate)
            row = {'model_id': model['id'], 'model_sha256': models.digest(definition),
                   'replicate': replicate, 'seed': seed, 'float_samples_hex': None,
                   'quantized_samples': None, 'barcode': None, 'counts': None,
                   'connectivity_verified': False, 'failure': None}
            stage = 'sampling'
            try:
                values = sampler(model, seed, n=n, cutoff=cutoff, side=side)
                models.require(type(values) is list and len(values) == n*n
                               and all(type(v) is float and math.isfinite(v) for v in values),
                               'Sampler must return the entire finite float grid')
                row['float_samples_hex'] = [v.hex() for v in values]
                stage = 'quantization'
                row['quantized_samples'] = [round(v*scale) for v in values]
                stage = 'barcode'
                row['barcode'] = exact_h0.compute(row['quantized_samples'], n)
                stage = 'connectivity'
                exact_h0.verify_by_connectivity(row['quantized_samples'], n, row['barcode'])
                stage = 'bin_counts'
                row['counts'] = bin_counts(row['barcode']['intervals'], scale, edges)
                row['connectivity_verified'] = True
            except Exception as error:
                row['counts'] = None
                row['connectivity_verified'] = False
                row['failure'] = {'stage': stage, 'type': type(error).__name__, 'message': str(error)}
            rows.append(row)
    summaries = {model['id']: summarize_rows([r for r in rows if r['model_id'] == model['id']],
                                             len(edges)-1, side=side, vertex_count=n*n)
                 for model in catalog['models']}
    output = {'schema_version': 1, 'purpose': 'exploratory_software_pilot',
              'scientific_status_authority': False, 'source_sha256': _source_hashes(),
              'catalog_sha256': hashlib.sha256(models.CATALOG_PATH.read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                              'machine': platform.machine(), 'system': platform.system(),
                              'sampler_mode': sampler_mode, 'sampler': SAMPLER_DESCRIPTIONS[sampler_mode]},
              'config': {'dimension': 2, 'side': side, 'grid': n, 'cutoff': cutoff,
                         'fields_per_model': fields_per_model, 'quantization_scale': scale,
                         'seed_namespace': EXPLORATORY_NAMESPACE, 'bin_edges': [_qtext(x) for x in edges],
                         'bin_convention': BIN_CONVENTION, 'quantization': QUANTIZATION},
              'model_definitions': definitions, 'rows': rows, 'summaries': summaries,
              'scopes': list(SCOPES)}
    return output


def _require_keys(record, keys, scope):
    models.require(type(record) is dict and set(record) == keys,
                   'Unexpected retained '+scope+' schema')


def _validate_environment_declarations(environment):
    """Validate portable declaration formats without authenticating the host.

    Python's major.minor.micro token may include a lowercase a/b/rc prerelease
    and a trailing '+' development marker. Other platform names remain open.
    """
    for field in ('python', 'implementation', 'machine', 'system'):
        value = environment[field]
        models.require(type(value) is str and bool(value)
                       and value == value.strip() and value.isprintable(),
                       'Environment '+field+' requires a canonical nonempty scalar string')
    number = r'(?:0|[1-9][0-9]*)'
    version = number+r'\.'+number+r'\.'+number+r'(?:(?:a|b|rc)'+number+r')?\+?'
    models.require(re.fullmatch(version, environment['python']) is not None,
                   'Environment python requires a canonical major.minor.micro version')


def _replay_float_samples(row, n):
    encoded = row['float_samples_hex']
    models.require(type(encoded) is list and len(encoded) == n*n
                   and all(type(x) is str for x in encoded), 'Incomplete floating samples')
    floats = [float.fromhex(x) for x in encoded]
    models.require(all(math.isfinite(x) and text == x.hex() for text, x in zip(encoded, floats)),
                   'Floating samples require canonical finite hex serialization')
    return floats


def _replay_quantization(row, n, scale, floats):
    samples = row['quantized_samples']
    models.require(type(samples) is list and len(samples) == n*n
                   and all(type(x) is int for x in samples)
                   and samples == [round(x*scale) for x in floats], 'Quantization replay mismatch')


def _replay_failed_intermediates(row, n, scale):
    """Check the completed state implied by a failed atomic pipeline stage.

    This binds retained intermediates, not exception authenticity or causal
    history. At a connectivity failure, source replay checks the returned
    barcode without claiming that the attempted independent check succeeded.
    A bin-count failure must also replay the already completed independent check.
    """
    stage = row['failure']['stage']
    if stage == 'sampling':
        models.require(all(row[key] is None for key in
                           ('float_samples_hex', 'quantized_samples', 'barcode')),
                       'Sampling failure has unexpected completed intermediates')
        return
    floats = _replay_float_samples(row, n)
    if stage == 'quantization':
        models.require(row['quantized_samples'] is None and row['barcode'] is None,
                       'Quantization failure has unexpected completed intermediates')
        return
    _replay_quantization(row, n, scale, floats)
    if stage == 'barcode':
        models.require(row['barcode'] is None, 'Barcode failure has a returned barcode')
        return
    exact_h0._result(row['barcode'])
    models.require(row['barcode'] == exact_h0.compute(row['quantized_samples'], n),
                   'Completed barcode source replay mismatch')
    if stage == 'bin_counts':
        exact_h0.verify_by_connectivity(row['quantized_samples'], n, row['barcode'])


def verify_observations(output):
    """Replay identity, completeness, exact grids and derived reports.

    Supplied arrays are not authenticated as generator outputs by this check.
    Compare a freshly generated run separately to test execution replay.
    Failed rows replay only their completed intermediates; recorded exception
    authenticity and the claimed causal history are not established.
    """
    _require_keys(output, {'schema_version', 'purpose', 'scientific_status_authority',
                          'source_sha256', 'catalog_sha256', 'environment', 'config',
                          'model_definitions', 'rows', 'summaries', 'scopes'}, 'observation')
    models.require(type(output.get('schema_version')) is int
                   and output.get('schema_version') == 1
                   and output.get('purpose') == 'exploratory_software_pilot'
                   and output.get('scientific_status_authority') is False, 'Wrong observation scope')
    models.require(output.get('source_sha256') == _source_hashes(), 'Executed-source identity mismatch')
    models.require(output.get('catalog_sha256') == hashlib.sha256(models.CATALOG_PATH.read_bytes()).hexdigest(),
                   'Catalog identity mismatch')
    models.require(output.get('scopes') == SCOPES, 'Changed observation scope')
    environment = output['environment']
    _require_keys(environment, {'python', 'implementation', 'machine', 'system',
                                'sampler_mode', 'sampler'}, 'environment')
    _validate_environment_declarations(environment)
    sampler_mode = environment.get('sampler_mode')
    models.require(sampler_mode in SAMPLER_DESCRIPTIONS
                   and environment.get('sampler') == SAMPLER_DESCRIPTIONS[sampler_mode],
                   'Sampler provenance declaration mismatch')
    config = output['config']
    _require_keys(config, {'dimension', 'side', 'grid', 'cutoff', 'fields_per_model',
                          'quantization_scale', 'seed_namespace', 'bin_edges',
                          'bin_convention', 'quantization'}, 'configuration')
    n, cutoff, count, scale, side = (config[k] for k in ('grid', 'cutoff', 'fields_per_model', 'quantization_scale', 'side'))
    models.require(type(config['dimension']) is int and config['dimension'] == 2
                   and config['seed_namespace'] == EXPLORATORY_NAMESPACE,
                   'Only the declared exploratory planar namespace is supported')
    models.require(config['bin_convention'] == BIN_CONVENTION and config['quantization'] == QUANTIZATION,
                   'Endpoint or quantization convention mismatch')
    edges = _validate_plan(n=n, cutoff=cutoff, fields_per_model=count, scale=scale,
                           side=side, edges=[Q(x) for x in config['bin_edges']])
    catalog = models.load_catalog()
    expected = [models.definition(model, cutoff=cutoff, side=side, catalog=catalog)
                for model in catalog['models']]
    models.require(output['model_definitions'] == expected, 'Model definition or normalization mismatch')
    hashes = {x['model']['id']: models.digest(x) for x in expected}
    rows = output['rows']
    models.require(type(rows) is list and len(rows) == 50*count, 'Missing planned field rows')
    for row in rows:
        _require_keys(row, {'model_id', 'model_sha256', 'replicate', 'seed', 'float_samples_hex',
                           'quantized_samples', 'barcode', 'counts', 'connectivity_verified',
                           'failure'}, 'field row')
    planned = [(model['id'], replicate) for model in catalog['models'] for replicate in range(count)]
    models.require([(r['model_id'], r['replicate']) for r in rows] == planned,
                   'Missing, duplicate or reordered field identity')
    for row in rows:
        models.require(row['model_sha256'] == hashes[row['model_id']]
                       and row['seed'] == seed_for(row['model_id'], row['replicate']),
                       'Field identity is not bound to its law and exploratory replicate')
        if row['failure'] is not None:
            failure = row['failure']
            models.require(type(failure) is dict and set(failure) == {'stage', 'type', 'message'}
                           and all(type(v) is str for v in failure.values())
                           and failure['stage'] in ('sampling', 'quantization', 'barcode', 'connectivity', 'bin_counts')
                           and row['counts'] is None and row['connectivity_verified'] is False,
                           'Invalid retained failure')
            _replay_failed_intermediates(row, n, scale)
            continue
        floats = _replay_float_samples(row, n)
        _replay_quantization(row, n, scale, floats)
        exact_h0.verify_by_connectivity(row['quantized_samples'], n, row['barcode'])
        models.require(row['connectivity_verified'] is True
                       and row['counts'] == bin_counts(row['barcode']['intervals'], scale, edges),
                       'Exact-grid verification or bin count mismatch')
    summaries = {model['id']: summarize_rows([r for r in rows if r['model_id'] == model['id']],
                                             len(edges)-1, side=side, vertex_count=n*n)
                 for model in catalog['models']}
    models.require(output['summaries'] == summaries, 'Planned denominator or mass normalization mismatch')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory; existing paths are refused')
    parser.add_argument('--verify', type=Path,
                        help='Check retained consistency; failed field rows return nonzero')
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--cutoff', type=int, choices=(2, 3), default=3,
                        help='Supported distinct-spectrum square Fourier cutoffs: 2 or 3')
    parser.add_argument('--fields-per-model', type=int, default=2)
    args = parser.parse_args(argv)
    models.require((args.output is None) != (args.verify is None), 'Choose exactly one of --output or --verify')
    if args.verify is not None:
        try:
            output = json.loads(args.verify.read_text(encoding='utf-8'), object_pairs_hook=models._unique_object)
            models.require(verify_observations(output) is True,
                           'Retained observation verification did not return True')
        except Exception as error:
            print('Retained record verification failed: '+str(error), file=sys.stderr)
            return 2
        failures = sum(row['failure'] is not None for row in output['rows'])
        if failures:
            print(f'RETAINED_RECORD_CONSISTENT_WITH_FAILURES; retained failed fields: {failures}; '
                  'no generator-authenticity or continuum claim')
            return 1
        print('RETAINED_QUANTIZED_GRID_VERIFICATION_PASS; no generator-authenticity or continuum claim')
        return 0
    # Reject an invalid plan before reserving execution custody. A generation
    # or runtime failure after this valid-plan reservation keeps its directory.
    _validate_plan(n=args.grid, cutoff=args.cutoff, fields_per_model=args.fields_per_model)
    args.output.mkdir(parents=True, exist_ok=False)
    output = _generate_observations(n=args.grid, cutoff=args.cutoff,
                                    fields_per_model=args.fields_per_model)
    payload = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+'\n'
    target = args.output/'observations.json'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    failures = sum(row['failure'] is not None for row in output['rows'])
    # Completed observations are saved before this final check. Never erase or rebind
    # them after rejection; report verification separately from raw custody.
    observation_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    validation = {'schema_version': 1, 'scientific_status_authority': False,
                  'observations_file': 'observations.json',
                  'observations_sha256': observation_sha256,
                  'final_record_verification': 'FAIL', 'planned_fields': len(output['rows']),
                  'failed_fields': failures, 'error': None,
                  'scope': 'Retained-record consistency only; no generator-authenticity or continuum claim'}
    try:
        models.require(verify_observations(output) is True,
                       'Final observation verification did not return True')
        validation['final_record_verification'] = 'PASS'
        exit_code = 0 if failures == 0 else 1
    except Exception as error:
        validation['error'] = {'type': type(error).__name__, 'message': str(error)}
        print('Final record verification failed: '+str(error), file=sys.stderr)
        exit_code = 2
    validation['exit_code'] = exit_code
    validation_target = args.output/'validation.json'
    with validation_target.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(validation, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'observations': str(target), 'sha256': observation_sha256,
                      'validation': str(validation_target),
                      'final_record_verification': validation['final_record_verification'],
                      'models': 50, 'planned_fields': len(output['rows']), 'failed_fields': failures,
                      'purpose': output['purpose']}, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
