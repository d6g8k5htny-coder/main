"""Frozen full-field/global H0 attempt, conditional on reviewed analytic guards.

Importing this driver loads only the standard library. Scientific imports occur
through prepare_sources after all thirteen consumed sources have been frozen.
Actual author: OpenAI/Codex; source exposed, organizational/blind credit zero.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

POP = 'experiments/universality/two_parameter_cusp_population/'
SOURCE_PATHS = (
    POP+'PROOF.md', POP+'CONTROLLED_FALSIFICATION.md', POP+'ANALYTIC_GUARDS.md',
    'experiments/periodic_h0/finite_certificate.py', 'experiments/periodic_h0/gaussian_tail.py',
    POP+'GLOBAL_SELECTOR_CONTRACT.md', 'docs/plans/2026-10-09-global-h0-selector.md',
    *(POP+name for name in ('surgery_field.py', 'test_surgery_field.py',
        'global_h0_selector.py', 'test_global_h0_selector.py',
        'run_global_h0_selector.py', 'test_global_h0_runner.py')))
CONSUMED_SOURCE_PINS = {
    POP+'PROOF.md': {'bytes': 38350, 'sha256': '9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326'},
    POP+'CONTROLLED_FALSIFICATION.md': {'bytes': 15296, 'sha256': 'a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e'},
    POP+'ANALYTIC_GUARDS.md': {'bytes': 10494, 'sha256': '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9'},
    'experiments/periodic_h0/finite_certificate.py': {'bytes': 14843, 'sha256': '0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b'},
    'experiments/periodic_h0/gaussian_tail.py': {'bytes': 4688, 'sha256': '07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8'},
    POP+'GLOBAL_SELECTOR_CONTRACT.md': {'bytes': 27262, 'sha256': '553e06924f8640803ab397d5dee1474d1cf376fcda42c78325e2531d1e0bf1a4'}}
REVIEW_PIN = {'bytes': 22303, 'sha256': 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08',
              'status': 'PASS_BOUNDED_ANALYTIC_GUARDS'}
SOUNDNESS_GATES = ('accepted_guard_binding', 'critical_completeness', 'field_width',
    'field_source_binding', 'core_identity_overlap', 'strict_height_order', 'positive_signs', 'component_incidence')
MODEL_ID = 'explicit_periodic_cusp_v1'
B, R1, DELTA = Q(4, 9), Q(1, 32), Q(1, 2**26)
FIELD_WIDTH = Q(1, 2**104)
_UNSET = object()


def _identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def canonical_bytes(value):
    """Exact rational strings and strict metadata types; no floating numbers."""
    def encode(item):
        if type(item) is Q:
            return str(item)
        if item is None or type(item) in (str, int, bool):
            return item
        if type(item) in (tuple, list):
            return [encode(v) for v in item]
        if type(item) is dict and all(type(k) is str for k in item):
            return {k: encode(v) for k, v in item.items()}
        raise ValueError('Unsupported exact JSON value: '+type(item).__name__)
    return (json.dumps(encode(value), sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def load_exact_json(raw):
    def reject(value):
        raise ValueError('Floating or nonfinite JSON token: '+value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_float=reject, parse_constant=reject)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed exact JSON') from exc


def frozen_fixtures():
    a, u0, v0 = Q(1, 2**18), Q(3, 2**32), Q(1, 2**47)
    controls = []
    for theta in (Q(1, 4), Q(1, 2), Q(3, 4), Q(0), Q(1), Q(3), Q(2)):
        for sign in (1, -1):
            controls.append(((3+theta**2)*a**2, sign*2*(1-theta**2)*a**3))
    controls += [(-u0/2, v0/2), (Q(0), v0/2), (u0/2, v0/2), (u0/2, Q(0))]
    return tuple({'fixture_id': f'fixture_{i:02d}', 'u': u, 'v': v}
                 for i, (u, v) in enumerate(controls, 1))


def literal_expectation(fixture_id):
    lifetimes = (Q(1, 2**76), Q(1, 2**76), Q(1, 2**73), Q(1, 2**73),
        Q(27, 2**76), Q(27, 2**76), None, None, Q(1, 2**70), Q(1, 2**70),
        None, None, Q(3, 2**74), Q(3, 2**74), None, None, None, Q(9, 2**68))
    counts = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)
    younger = (0, 2, 0, 2, 0, 2, None, None, None, None, None, None, 2, 0, None, None, None, None)
    ids = tuple(f'fixture_{i:02d}' for i in range(1, 19))
    if type(fixture_id) is not str or fixture_id not in ids:
        raise ValueError('Unknown frozen fixture ID')
    index = ids.index(fixture_id)
    return {'lifetimes': () if lifetimes[index] is None else (lifetimes[index],),
            'population_count': counts[index], 'younger_root_ordinal': younger[index]}


def compare_expectations(fixture_id, selection, verification, soundness, *, expected_lookup=literal_expectation):
    if (selection.get('status') != 'CERTIFIED' or
            selection.get('actual_barcode', {}).get('status') != 'CERTIFIED' or
            verification.get('status') != 'PASS' or
            any(soundness.get(name) != 'PASS' for name in SOUNDNESS_GATES)):
        return {'status': 'NOT_COMPARED', 'mismatches': []}
    expected = expected_lookup(fixture_id)
    bars = selection['actual_barcode']['finite']
    actual = tuple(bar['lifetime_exact'] for bar in bars)
    mismatches = []
    if actual != expected['lifetimes']:
        mismatches.append({'quantity': 'actual_finite_lifetimes', 'computed': actual, 'expected': expected['lifetimes']})
    count = selection['population_projection']['count']
    if type(count) is not int or count != expected['population_count']:
        mismatches.append({'quantity': 'population_count', 'computed': count, 'expected': expected['population_count']})
    younger = expected['younger_root_ordinal']
    if younger is not None and (len(bars) != 1 or type(bars[0]['birth_root']) is not int or bars[0]['birth_root'] != younger):
        mismatches.append({'quantity': 'younger_root_ordinal', 'computed': tuple(bar['birth_root'] for bar in bars), 'expected': younger})
    return {'status': 'MISMATCH' if mismatches else 'PASS', 'mismatches': mismatches}


def _source_path(base, name):
    base = Path(base).resolve()
    if type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Repository-relative source path required')
    path = (base/name).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Source escapes repository: '+name)
    return path


def freeze_sources(paths, *, base, expected):
    manifest = {}
    for name in paths:
        if name in manifest:
            raise ValueError('Duplicate source path: '+name)
        try:
            actual = _identity(_source_path(base, name).read_bytes())
        except OSError as exc:
            raise ValueError('Unavailable source: '+name) from exc
        if name in expected and actual != expected[name]:
            raise ValueError('Consumed source mismatch: '+name)
        manifest[name] = actual
    if not set(expected) <= manifest.keys():
        raise ValueError('Pinned source omitted from manifest')
    return manifest


def source_drift(manifest, *, base):
    changed = []
    for name, expected in manifest.items():
        try:
            actual = _identity(_source_path(base, name).read_bytes())
        except (OSError, ValueError):
            actual = None
        if actual != expected:
            changed.append(name)
    return sorted(changed)


def prepare_sources(*, base, importer):
    manifest = freeze_sources(SOURCE_PATHS, base=base, expected=CONSUMED_SOURCE_PINS)
    return manifest, importer()


def bind_accepted_premise(premise, manifest):
    if type(premise) is not dict:
        raise ValueError('Accepted analytic premise unavailable')
    review = premise.get('review', {})
    source = CONSUMED_SOURCE_PINS.get(POP+'ANALYTIC_GUARDS.md')
    analytic = premise.get('analytic_source', {})
    if (premise.get('status') != 'ACCEPTED' or
            premise.get('record_role') != 'historical_accepted_analytic_premise' or
            source is None or type(analytic) is not dict or analytic != source or
            type(analytic.get('bytes')) is not int or type(analytic.get('sha256')) is not str or
            manifest.get(POP+'ANALYTIC_GUARDS.md') != source or
            type(review) is not dict or any(review.get(k) != v for k, v in REVIEW_PIN.items()) or
            type(review.get('bytes')) is not int or type(review.get('sha256')) is not str or
            type(review.get('status')) is not str or
            review.get('nonauthor') is not True or review.get('source_exposure') != 'source-exposed' or
            premise.get('accepted_by') != 'OpenAI/Codex root' or
            premise.get('scope') != 'bounded analytic guards for explicit_periodic_cusp_v1' or
            any(type(premise.get(k)) is not int or premise[k] != 0 for k in
                ('organization_independence_credit', 'blind_independence_credit'))):
        raise ValueError('Malformed or unbound accepted analytic premise')
    return deepcopy(premise)


def normalize_computed_predicates(record):
    def visit(value):
        if type(value) is dict:
            result = {}
            for key, item in value.items():
                if key == 'accepted_premise':
                    result[key] = deepcopy(item)
                elif key in ('checks', 'gates') and type(item) is dict:
                    result[key] = {name: {'status': 'PASS' if v is True else 'FAIL' if v is False else 'NOT_EVALUATED',
                                          'computed_value': v}
                                   if type(v) is bool or v is None else visit(v) for name, v in item.items()}
                else:
                    result[key] = visit(item)
            return result
        if type(value) in (list, tuple):
            return type(value)(visit(item) for item in value)
        return deepcopy(value)
    return visit(record)


def withdraw_acceptance(record, status, *, phase, detail):
    if status not in ('INCONCLUSIVE_SOURCE_DRIFT', 'INCONCLUSIVE_BUDGET', 'INCONCLUSIVE_PRECISION', 'FAIL_IMPLEMENTATION'):
        raise ValueError('Invalid global withdrawal status')
    def visit(value):
        if type(value) is dict:
            result = {key: deepcopy(item) if key == 'accepted_premise' else visit(item)
                      for key, item in value.items()}
            if 'status' in result:
                result.setdefault('computed_status_before_withdrawal', result['status'])
                result['status'] = status
            return result
        if type(value) in (tuple, list):
            return type(value)(visit(item) for item in value)
        return deepcopy(value)
    result = visit(record)
    result.setdefault('withdrawals', []).append({'status': status, 'phase': phase, 'detail': detail})
    return result


def finalize_acceptance(record, *, source_drift, start, now, seconds=900):
    if type(seconds) is not int or not 0 < seconds <= 900:
        raise ValueError('Positive strict reduced deadline required')
    if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in (start, now)) or now < start:
        raise ValueError('Valid monotonic clock floats required')
    result = deepcopy(record)
    # Deadline first, source drift last: drift has overall precedence; both reasons survive.
    if now-start >= seconds:
        result = withdraw_acceptance(result, 'INCONCLUSIVE_BUDGET', phase='final_deadline', detail='Whole-attempt deadline exhausted')
    if source_drift:
        result = withdraw_acceptance(result, 'INCONCLUSIVE_SOURCE_DRIFT', phase='final_sources', detail=list(source_drift))
    return result


def write_exclusive_artifacts(output_dir, *, certificate, results_md, run_metadata):
    if type(results_md) is not str or type(run_metadata) is not dict or 'outputs' in run_metadata:
        raise ValueError('Report text and unambiguous run metadata required')
    certificate_raw, results_raw = canonical_bytes(certificate), results_md.encode('utf-8')
    run = deepcopy(run_metadata)
    run['outputs'] = {'CERTIFICATE.json': _identity(certificate_raw), 'RESULTS.md': _identity(results_raw)}
    run_raw = canonical_bytes(run)
    output = Path(output_dir)
    output.mkdir()  # Exclusive even for an empty existing directory or a dangling symlink.
    for name, raw in (('CERTIFICATE.json', certificate_raw), ('RESULTS.md', results_raw), ('RUN.json', run_raw)):
        with (output/name).open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return run


# Portable historical evidence from the controller, not a host-path dependency.
ACCEPTED_PREMISE = {'accepted_by': 'OpenAI/Codex root', 'analytic_source': {'bytes': 10494, 'sha256': '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9'}, 'blind_independence_credit': 0, 'historical_delivery': {'native_head': '519b235f4d3df381caccede3b00c03c943543efd', 'native_pr': 346, 'private_receipt_file_id': '1JFDDy0MQXL7RVyGn-AF1p8f7kIV9LpBL', 'receipt_sha256': '13bf595bf3a138bcf169be3495d7fd37519def7be4489372f985ac528503c14f', 'work_events_delivery': 1350, 'work_events_release': 1351}, 'human_personal_reading': 'NOT_CLAIMED', 'interpretation': 'Root consumes the separately reviewed unchanged analytic premise for this explicit model. This is historical acceptance evidence, not a live field/selector run verdict, human reading or scientific promotion. Driver embeds portable metadata; no outside-host path is an executable dependency.', 'organization_independence_credit': 0, 'record_role': 'historical_accepted_analytic_premise', 'review': {'actual_author': 'OpenAI/Codex /root', 'actual_reviewer': 'OpenAI/Codex /root/analytic_guards_review', 'bytes': 22303, 'nonauthor': True, 'sha256': 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08', 'source_exposure': 'source-exposed', 'status': 'PASS_BOUNDED_ANALYTIC_GUARDS'}, 'scientific_promotion': 'NONE', 'scope': 'bounded analytic guards for explicit_periodic_cusp_v1', 'status': 'ACCEPTED'}


class _DriverFailure(Exception):
    def __init__(self, status, phase, detail):
        self.status, self.phase, self.detail = status, phase, detail
        super().__init__(detail)


def _scientific_imports():
    """Called only after freezing sources; helpers import no scientific code."""
    local = str(Path(__file__).resolve().parent)
    if local not in sys.path:
        sys.path.insert(0, local)
    from surgery_field import Budget, guard_certificate, chart_field_bounds, torus_field_bounds
    from global_h0_selector import select_h0, verify_h0
    return dict(Budget=Budget, guard_certificate=guard_certificate,
        chart_field_bounds=chart_field_bounds, torus_field_bounds=torus_field_bounds,
        select_h0=select_h0, verify_h0=verify_h0)


def _chart_anchors():
    return ((Q(0), Q(0)),
        *((sign*scale*R1, Q(0)) for scale in (Q(1, 4), Q(1, 2), Q(3, 4), Q(1), Q(2), Q(3)) for sign in (1, -1)),
        (Q(0), R1/2), (Q(0), 3*R1/4), (R1/2, R1/2))


def _torus_anchors():
    return ((Q(0), Q(0)), (Q(0), Q(1, 2)), (Q(1, 2), Q(0)), (Q(1, 2), Q(1, 2)),
        (Q(1, 4), Q(0)), (Q(0), Q(1, 4)), (Q(-1, 2), Q(1, 8)), (Q(1, 8), Q(-1, 2)))


def _ordered_interval(value):
    return (type(value) is tuple and len(value) == 2 and all(type(v) is Q for v in value)
            and value[0] <= value[1])


def _evaluate_field(row, handles, budget):
    """Retain completed anchors immediately; the Budget owns full call records."""
    u, v, selection = row['u'], row['v'], row['selection']
    evaluations = row['field_evaluations'] = []
    start_index = len(budget.snapshot()['field_records'])
    row['field_record_start'] = start_index
    checks = {'core_identity_overlap': True, 'anchor_seed_values': True, 'periodic_translation': True}
    row['field_checks'] = {'status': 'COMPUTING', 'checks': checks}

    def evaluate(route, point, *, height=None):
        if route == 'chart':
            interval = handles['chart_field_bounds'](u, v, (point[0], point[0]), (point[1], point[1]), budget=budget)
        elif route == 'root_box':
            interval = handles['chart_field_bounds'](u, v, point, (Q(0), Q(0)), budget=budget)
        else:
            interval = handles['torus_field_bounds'](u, v, point, budget=budget)
        record = {'route': route, 'input': point, 'returned_interval': interval, 'status': 'COMPUTED'}
        evaluations.append(record)  # Attach before any fallible check or checkpoint.
        if not _ordered_interval(interval):
            raise _DriverFailure('FAIL_IMPLEMENTATION', 'driver/field_interval', 'Malformed or reversed field interval')
        if interval[1]-interval[0] > FIELD_WIDTH:
            raise _DriverFailure('INCONCLUSIVE_PRECISION', 'driver/field_width', 'Field width exceeds frozen target')
        if height is not None:
            shifted = (B+height[0], B+height[1])
            record['algebraic_field_interval'] = shifted
            checks['core_identity_overlap'] &= max(shifted[0], interval[0]) <= min(shifted[1], interval[1])
        return interval

    for point in _chart_anchors():
        interval = evaluate('chart', point)
        s, z = point
        expected = None
        if s*s+z*z <= R1*R1/4:
            budget.charge_polynomial(phase='driver/core_anchor')
            expected = B-s**4/4-(1+s*s/4)*z*z+u*s*s/2+v*s
            checks['core_identity_overlap'] &= interval[0] <= expected <= interval[1]
        elif s*s+z*z >= Q(1, 128):
            budget.charge_polynomial(phase='driver/collar_anchor')
            expected = B-s*s-z*z
            checks['anchor_seed_values'] &= interval[0] <= expected <= interval[1]
        if expected is not None:
            evaluations[-1]['exact_identity_value'] = expected
    roots = selection['critical_roots']['roots']
    heights = {height['root_ordinal']: height for height in selection['critical_heights']}
    for root in roots:
        box = root['isolator']
        if not _ordered_interval(box) or box[0] < -R1/2 or box[1] > R1/2:
            raise _DriverFailure('FAIL_IMPLEMENTATION', 'driver/core_box', 'Critical box outside exact core')
        evaluate('root_box', box, height=heights[root['ordinal']]['centered_interval'])
        evaluations[-1]['root_ordinal'] = root['ordinal']
    seed = {(Q(0), Q(1, 2)): Q(2, 9), (Q(1, 2), Q(0)): Q(-2, 9),
            (Q(1, 2), Q(1, 2)): Q(-4, 9)}
    for point in _torus_anchors():
        interval = evaluate('torus', point)
        translated = evaluate('torus', (point[0]+1, point[1]-1))
        checks['periodic_translation'] &= interval == translated
        if point in seed:
            checks['anchor_seed_values'] &= interval[0] <= seed[point] <= interval[1]
            evaluations[-2]['exact_seed_value'] = seed[point]
    records = budget.snapshot()['field_records'][start_index:]
    row['field_record_count'] = len(records)
    torus = [record for record in records if record['route'] == 'torus']
    checks['periodic_translation'] &= len(torus) == 16 and all(
        a.get('canonical_point') == b.get('canonical_point') and a.get('returned_interval') == b.get('returned_interval')
        for a, b in zip(torus[::2], torus[1::2]))
    row['field_checks']['status'] = 'PASS' if all(checks.values()) else 'FAIL'
    return records


def _soundness(row, records, manifest, premise):
    """Summarize real replay/field checks; producer success flags are insufficient."""
    selection, verification = row['selection'], row['verification']
    vc = verification.get('checks', {})
    verified = verification.get('status') == 'PASS' and bool(vc) and all(value is True for value in vc.values())
    guard = selection['guard']
    guard_checks = guard.get('checks', {})
    bound = (guard.get('status') == 'ACCEPTED_ANALYTIC_PREMISE' and guard.get('model_id') == MODEL_ID
        and guard.get('analytic_source') == premise['analytic_source']
        and guard.get('accepted_review') == REVIEW_PIN and bool(guard_checks)
        and all(value is True for value in guard_checks.values()))
    def replay(prefix):
        matches = [value for name, value in vc.items() if name.startswith(prefix)]
        return verified and bool(matches) and all(value is True for value in matches)
    def sourced(record):
        model = record.get('model', {})
        return (record.get('u') == row['u'] and record.get('v') == row['v']
            and record.get('fixture_id') == row['fixture_id'] and record.get('status') == 'CERTIFIED'
            and record.get('gates', {}).get('identity') is True
            and model.get('model_id') == MODEL_ID and model.get('analytic_source') == premise['analytic_source']
            and model.get('accepted_review') == REVIEW_PIN
            and model.get('implementation_source') == {'filename': 'surgery_field.py', **manifest[POP+'surgery_field.py']}
            and model.get('dependencies') == {name: manifest['experiments/periodic_h0/'+name]
                for name in ('finite_certificate.py', 'gaussian_tail.py')})
    field_checks = row['field_checks']['checks']
    gates = {
        'accepted_guard_binding': bound and replay('guard/source_bound_rational_replay') and replay('guard/old_criticals'),
        'critical_completeness': selection['critical_roots'].get('status') == 'CERTIFIED'
            and replay('verify/critical/complete') and replay('heights/values_and_inertia'),
        'field_width': bool(records) and all(_ordered_interval(record.get('returned_interval'))
            and record['returned_interval'][1]-record['returned_interval'][0] <= FIELD_WIDTH
            and record.get('gates', {}).get('ordered') is True and record.get('gates', {}).get('width') is True for record in records),
        'field_source_binding': bool(records) and all(sourced(record) for record in records),
        'core_identity_overlap': guard_checks.get('core_critical_height') is True
            and field_checks['core_identity_overlap'] and field_checks['anchor_seed_values'] and field_checks['periodic_translation'],
        'strict_height_order': replay('groups/partition') and replay('groups/barrier_band'),
        'positive_signs': replay('levels/regular_sign'),
        'component_incidence': replay('levels/components') and replay('graph/incidence_witnesses') and replay('graph/barcode')}
    return {name: 'PASS' if gates[name] else 'FAIL' for name in SOUNDNESS_GATES}


def _grid_obstruction():
    a = Q(1, 2**18)
    return {'status': 'RETAINED_ANALYTIC_CERTIFICATE', 'source': POP+'GLOBAL_SELECTOR_CONTRACT.md',
        'fixture_id': 'fixture_03', 'grid_size': (1024, 1024), 'grid_execution': 'NOT_RUN',
        'centered_level': 9*a**4/128, 'younger_centered_height': 9*a**4/64,
        'saddle_centered_height': -23*a**4/64, 'continuous_components': 2, 'vertex_superlevel_count': 0,
        'bound_coefficient': -Q(1, 4)+Q(3, 2**13)+Q(1, 2**17),
        'checks': {'level_between_criticals': -23*a**4/64 < 9*a**4/128 < 9*a**4/64,
                   'grid_bound_negative': -Q(1, 4)+Q(3, 2**13)+Q(1, 2**17) < 0},
        'scope': 'Previously derived exact obstruction; no new grid or other discrete-bar claim'}


def _results_markdown(certificate):
    lines = ['# Complete periodic field and exact global H0 selector', '',
        'Driver disposition: **'+certificate['status']+'**.', '',
        'Full-field execution: '+certificate['field_execution']['status']+'.',
        'Global selector/replay execution: '+certificate['selector_execution']['status']+'.', '',
        'Conditional on the bound historical analytic guards. Source-exposed same-team review; organizational and blind credit zero.', '',
        '| Fixture | Disposition | Computed positive finite lifetimes | Population count | Comparison |',
        '| --- | --- | --- | --- | --- |']
    for row in certificate['fixtures']:
        selection = row.get('selection', {})
        bars = selection.get('actual_barcode', {}).get('finite', [])
        lifetimes = ', '.join(str(bar['lifetime_exact']) if bar['lifetime_exact'] is not None
            else str(bar['lifetime_interval']) for bar in bars) or 'none computed'
        lines.append('| '+row['fixture_id']+' | '+row['status']+' | '+lifetimes+' | '+
            str(selection.get('population_projection', {}).get('count', 'uncomputed'))+' | '+
            row.get('comparison', {}).get('status', 'NOT_COMPARED')+' |')
    lines += ['', 'Computed evidence and withdrawn prior statuses remain in CERTIFICATE.json. RUN.json binds the two other artifacts.', '',
        'The outside recorder must establish exit zero and complete-command wall time under 900 seconds, including writes and flush.', '',
        'Gaussian universality, random/blind campaigns, same-field spatial Poisson, higher homology, Lean/kernel and scientific promotion remain NOT_RUN.', '']
    return '\n'.join(lines)


def run_attempt(*, base=None, output_dir=None, importer=None, premise=_UNSET):
    """Execute the declared lifecycle; injection allows temporary orchestration tests."""
    start = time.monotonic()
    base = Path(base) if base is not None else Path(__file__).resolve().parents[3]
    output = Path(output_dir) if output_dir is not None else base/'experiments/universality/results/global_selector_1'
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    certificate = {'schema_version': 1, 'model_id': MODEL_ID, 'status': 'NOT_RUN',
        'fixtures': [], 'planned_fixtures': frozen_fixtures(), 'work': {}, 'phase_times': [],
        'field_execution': {'status': 'NOT_RUN', 'completed_fixtures': 0},
        'selector_execution': {'status': 'NOT_RUN', 'completed_fixtures': 0},
        'grid_obstruction': _grid_obstruction(),
        'lineage': {'actual_author': 'OpenAI/Codex /root/global_runner_implementation',
            'source_exposure': 'source-exposed', 'organization_independence_credit': 0, 'blind_independence_credit': 0},
        'remaining_scopes': {name: 'NOT_RUN' for name in ('gaussian_universality', 'random_blind_campaign',
            'same_field_spatial_poisson', 'higher_homology', 'lean_kernel', 'scientific_promotion')}}
    budget, manifest, phase = None, {}, 'source_freeze'
    def timed(name, previous):
        now = time.monotonic()
        certificate['phase_times'].append({'phase': name, 'seconds': format(now-previous, '.9f')})
        return now
    phase_start = start
    try:
        if sys.version_info[:2] != (3, 12):
            raise _DriverFailure('NOT_RUN', 'runtime', 'Python 3.12 runtime unavailable')
        manifest, handles = prepare_sources(base=base, importer=importer or _scientific_imports)
        certificate['sources_before'] = manifest
        phase_start = timed('source_freeze_and_imports', phase_start)
        phase = 'accepted_premise'
        supplied = ACCEPTED_PREMISE if premise is _UNSET else premise
        if supplied is None:
            raise _DriverFailure('NOT_RUN', phase, 'Accepted analytic premise unavailable')
        accepted = bind_accepted_premise(supplied, manifest)
        certificate['accepted_premise'] = accepted
        budget = handles['Budget'](start)
        budget.checkpoint('driver/guard')
        certificate['guard'] = handles['guard_certificate']()
        phase_start = timed('accepted_premise_and_guard', phase_start)
        for fixture in certificate['planned_fixtures']:
            phase = 'fixture/'+fixture['fixture_id']
            row = {**fixture, 'status': 'COMPUTING'}
            certificate['fixtures'].append(row)
            budget.begin_fixture(fixture['fixture_id'])
            u, v = fixture['u'], fixture['v']
            row['selection'] = handles['select_h0'](u, v, budget=budget)
            if row['selection']['status'] != 'CERTIFIED':
                raise _DriverFailure(row['selection']['status'], phase, 'Selection did not certify')
            # The verifier receives the actual raw bool/null predicate certificate.
            row['verification'] = handles['verify_h0'](u, v, row['selection'], budget=budget)
            if row['verification']['status'] != 'PASS':
                raise _DriverFailure(row['verification']['status'], phase, 'Independent verification did not pass')
            certificate['selector_execution']['completed_fixtures'] += 1
            records = _evaluate_field(row, handles, budget)
            soundness = _soundness(row, records, manifest, accepted)
            # Explicit status records permit complete recursive withdrawal later.
            row['soundness'] = {name: {'status': status} for name, status in soundness.items()}
            row['comparison'] = compare_expectations(fixture['fixture_id'], row['selection'], row['verification'],
                soundness, expected_lookup=literal_expectation)
            if row['comparison']['status'] == 'NOT_COMPARED':
                raise _DriverFailure('FAIL_IMPLEMENTATION', phase, 'Required soundness gate failed')
            if row['comparison']['status'] == 'MISMATCH':
                row['implementation_model_prediction_falsifier'] = True
                raise _DriverFailure('FAIL_IMPLEMENTATION', phase, 'Sound post-selection expectation mismatch')
            row['status'] = 'CERTIFIED'
            certificate['field_execution']['completed_fixtures'] += 1
            budget.checkpoint('driver/fixture_finish')
            phase_start = timed(phase, phase_start)
        certificate['field_execution']['status'] = 'CERTIFIED'
        certificate['selector_execution']['status'] = 'CERTIFIED'
        certificate['status'] = 'CERTIFIED'
    except Exception as exc:
        status = getattr(exc, 'status', 'INCONCLUSIVE_SOURCE_DRIFT' if phase == 'source_freeze' and isinstance(exc, ValueError) else 'FAIL_IMPLEMENTATION')
        if status not in ('NOT_RUN', 'INCONCLUSIVE_BUDGET', 'INCONCLUSIVE_SOURCE_DRIFT', 'INCONCLUSIVE_PRECISION', 'FAIL_IMPLEMENTATION'):
            status = 'FAIL_IMPLEMENTATION'
        certificate['failure'] = {'status': status, 'phase': getattr(exc, 'phase', phase), 'detail': str(exc)}
        certificate['status'] = status
    # One final whole-attempt ledger; nested scientific work remains scoped.
    if budget is not None:
        certificate['work'] = budget.snapshot()
        certificate['work']['ledger_scope'] = 'whole_attempt'
    certificate = normalize_computed_predicates(certificate)
    if certificate['status'] not in ('CERTIFIED', 'NOT_RUN'):
        certificate = withdraw_acceptance(certificate, certificate['status'], phase='attempt_failure', detail=certificate['failure'])
    certificate['sources_after'] = {}
    for name in manifest:
        try:
            certificate['sources_after'][name] = _identity(_source_path(base, name).read_bytes())
        except (OSError, ValueError):
            certificate['sources_after'][name] = None
    certificate['elapsed_before_serialization_seconds'] = format(time.monotonic()-start, '.9f')
    canonical_bytes(certificate)  # Preflight exact encoding before any output mutation.
    drift = source_drift(manifest, base=base)
    certificate['source_drift'] = drift
    certificate = finalize_acceptance(certificate, source_drift=drift, start=start, now=time.monotonic())
    run = {'schema_version': 1, 'model_id': MODEL_ID, 'status': certificate['status'],
        'command': 'python3 -B -S '+POP+'run_global_h0_selector.py',
        'python_executable': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
        'utc_before_writes': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
        'elapsed_before_writes_seconds': format(time.monotonic()-start, '.9f'),
        'sources_before': manifest, 'sources_after': certificate['sources_after'], 'source_drift': drift,
        'complete_command_acceptance': 'REQUIRES_EXTERNAL_RECORDER_EXIT_ZERO_AND_UNDER_900_SECONDS',
        'organization_independence_credit': 0, 'blind_independence_credit': 0}
    receipt = write_exclusive_artifacts(output, certificate=certificate,
        results_md=_results_markdown(certificate), run_metadata=run)
    return {'certificate': certificate, 'run': receipt}


def main():
    try:
        result = run_attempt()
    except (OSError, ValueError) as exc:
        print('Driver setup/output failure: '+str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes({'status': result['certificate']['status'],
        'fixtures_attempted': len(result['certificate']['fixtures']),
        'field_execution': result['certificate']['field_execution'],
        'selector_execution': result['certificate']['selector_execution']}))
    return 0 if result['certificate']['status'] == 'CERTIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
