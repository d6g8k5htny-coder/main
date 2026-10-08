"""Audit the frozen default successful pilot and its validation receipt.

This checks current source identity, exact retained grids, the observation-byte
digest, and successful computation custody. It does not authenticate generator
outputs or provide ideal-field, continuum, blind, or scientific acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.universality import models, run_pilot


def _reject_constant(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def _read_record(payload):
    return json.loads(payload, object_pairs_hook=models._unique_object,
                      parse_constant=_reject_constant)


def _require_keys(record, keys, scope):
    models.require(type(record) is dict and set(record) == keys,
                   'Unexpected published '+scope+' schema')


def verify_published_artifact(directory):
    directory = Path(directory)
    payload = (directory/'observations.json').read_bytes()
    observations = _read_record(payload)
    _require_keys(observations, {'schema_version', 'purpose', 'scientific_status_authority',
                                'source_sha256', 'catalog_sha256', 'environment', 'config',
                                'model_definitions', 'rows', 'summaries', 'scopes'}, 'observation')
    models.require(run_pilot.verify_observations(observations) is True,
                   'Retained record verification did not return True')
    _require_keys(observations['environment'], {'python', 'implementation', 'machine', 'system',
                                               'sampler_mode', 'sampler'}, 'environment')
    _require_keys(observations['config'], {'dimension', 'side', 'grid', 'cutoff', 'fields_per_model',
                                          'quantization_scale', 'seed_namespace', 'bin_edges',
                                          'bin_convention', 'quantization'}, 'configuration')
    catalog = models.load_catalog()
    expected_config = {
        'dimension': catalog['domain']['dimension'], 'side': catalog['domain']['side'],
        **catalog['default_parameters'], 'seed_namespace': run_pilot.EXPLORATORY_NAMESPACE,
        'bin_edges': [str(edge) for edge in run_pilot.DEFAULT_EDGES],
        'bin_convention': run_pilot.BIN_CONVENTION, 'quantization': run_pilot.QUANTIZATION,
    }
    models.require(models._matches_declared_metadata(observations['config'], expected_config),
                   'Published configuration must match the frozen default plan')
    for row in observations['rows']:
        _require_keys(row, {'model_id', 'model_sha256', 'replicate', 'seed', 'float_samples_hex',
                            'quantized_samples', 'barcode', 'counts', 'connectivity_verified',
                            'failure'}, 'field row')
    models.require(observations['environment']['sampler_mode'] == 'declared_float_sampler',
                   'Injected arithmetic/test sampler records cannot pass the publication gate')
    failed = sum(row['failure'] is not None for row in observations['rows'])
    models.require(failed == 0, 'Retained field failures prevent a successful published artifact check')
    validation = _read_record((directory/'validation.json').read_bytes())
    expected = {'schema_version': 1, 'scientific_status_authority': False,
                'observations_file': 'observations.json',
                'observations_sha256': hashlib.sha256(payload).hexdigest(),
                'final_record_verification': 'PASS', 'planned_fields': len(observations['rows']),
                'failed_fields': 0, 'error': None, 'exit_code': 0,
                'scope': 'Retained-record consistency only; no generator-authenticity or continuum claim'}
    # JSON comparison preserves strict scalar types (Python bool equals 0/1).
    models.require(type(validation) is dict
                   and models.canonical_bytes(validation) == models.canonical_bytes(expected),
                   'Validation receipt does not match successful retained observation bytes')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing observations.json and validation.json')
    args = parser.parse_args(argv)
    verify_published_artifact(args.directory)
    print('PUBLISHED_RETAINED_ARTIFACT_CHECK_PASS; source/byte/record checks only; no generator-authenticity or continuum claim')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
