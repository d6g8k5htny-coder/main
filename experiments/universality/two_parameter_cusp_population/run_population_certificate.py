"""Run one frozen exact-population attempt and retain its three artifacts.

Standard invocation (from repository root):
python3 -B -S experiments/universality/two_parameter_cusp_population/run_population_certificate.py
"""
from datetime import datetime, timezone
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[3]
POPULATION = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPOSITORY / 'experiments/universality/results/exact_population_1'
COMMAND = 'python3 -B -S experiments/universality/two_parameter_cusp_population/run_population_certificate.py'
FIXED_HASHES = {
    'experiments/universality/two_parameter_cusp_population/CONTROLLED_FALSIFICATION.md':
        'a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e',
    'experiments/universality/two_parameter_cusp_population/PROOF.md':
        '9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326',
    'experiments/periodic_h0/finite_certificate.py':
        '0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b',
    'experiments/periodic_h0/gaussian_tail.py':
        '07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8',
    'experiments/periodic_h0/mean_adaptive_confidence.py':
        'd83b4d530d00aafab1ebf5aa262869911db94d3b088309df3d878089285922a5',
    'experiments/periodic_h0/finite_count_loss.py':
        '85fca300f2dcb1af6102a3d82205e80cf5cb7f345b945186f2deb14d6b7f4317'}
SOURCE_PATHS = [REPOSITORY / p for p in FIXED_HASHES] + [
    POPULATION / name for name in ('population_certificate.py',
                                  'test_population_certificate.py',
                                  'run_population_certificate.py')]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_sources(paths=SOURCE_PATHS, base=REPOSITORY):
    base = Path(base)
    result = {}
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        key = str(path.relative_to(base))
        _require(key not in result, 'Duplicate frozen source')
        result[key] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def source_drift(manifest, base=REPOSITORY):
    changed = []
    for name, metadata in manifest.items():
        path = Path(base) / name
        try:
            raw = path.read_bytes()
            same = len(raw) == metadata['bytes'] and hashlib.sha256(raw).hexdigest() == metadata['sha256']
        except OSError:
            same = False
        if not same:
            changed.append(name)
    return changed


def _json_value(value):
    if type(value) is Q:
        return str(value)
    if type(value) in (str, int, bool) or value is None:
        return value
    if type(value) in (tuple, list):
        return [_json_value(item) for item in value]
    if type(value) is dict:
        _require(all(type(key) is str for key in value), 'String JSON keys required')
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError('Noncanonical certificate value: ' + type(value).__name__)


def canonical_bytes(value):
    return (json.dumps(_json_value(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def load_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    def reject(value):
        raise ValueError('Float or nonfinite certificate value: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_float=reject, parse_constant=reject)


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _overall(statuses):
    for status in ('FAIL_IMPLEMENTATION', 'FAIL_ROUTE_CONSISTENCY',
                   'INCONCLUSIVE_SOURCE_DRIFT', 'INCONCLUSIVE_BUDGET',
                   'INCONCLUSIVE_PRECISION', 'FALSIFIED'):
        if status in statuses:
            return status
    return 'PASS' if statuses and all(x == 'PASS' for x in statuses) else 'INCONCLUSIVE_PRECISION'


def _empty_certificate(status, error):
    return {'schema_version': 1, 'status': status, 'rows': [], 'error': error,
            'scope_statuses': {'exact_non_gaussian_population': status,
                              'rational_guard_consequents': 'NOT_RUN',
                              'analytic_cutoff_derivative_premise': 'REVIEW_REQUIRED',
                              'complete_surgery_field': 'NOT_RUN',
                              'independent_global_h0_selector': 'NOT_RUN',
                              'gaussian_universality': 'NOT_RUN', 'blind_50_campaign': 'NOT_RUN',
                              'higher_homology': 'NOT_RUN', 'same_field_spatial_poisson': 'NOT_RUN',
                              'lean_kernel': 'NOT_RUN', 'scientific_promotion': 'NOT_RUN',
                              'organizational_independence_credit': 0}}


def apply_final_runtime_budget(certificate, start, time_budget):
    """Withdraw acceptance if posthash/serialization preparation exhausts time.

    The payload may retain useful completed calculations, but no earlier accepted
    prediction remains accepted once the whole-attempt deadline is exhausted.
    """
    if time.monotonic() < start + time_budget:
        return False
    certificate['computed_status_before_runtime_budget'] = certificate['status']
    certificate['status'] = 'INCONCLUSIVE_BUDGET'
    certificate['runtime_budget_exhausted_phase'] = 'final_posthash_and_serialization_preparation'
    certificate['scope_statuses']['exact_non_gaussian_population'] = 'INCONCLUSIVE_BUDGET'
    for row in certificate['rows']:
        row['computed_status_before_runtime_budget'] = row['status']
        row['computed_coefficient_status_before_runtime_budget'] = row['coefficient_status']
        row['status'] = 'INCONCLUSIVE_BUDGET'
        row['coefficient_status'] = 'INCONCLUSIVE_BUDGET'
        if row.get('finite_void'):
            row['finite_void']['computed_status_before_runtime_budget'] = row['finite_void']['status']
            row['finite_void']['status'] = 'INCONCLUSIVE_BUDGET'
    return True


def apply_source_drift(certificate, drift):
    if drift:
        certificate['pre_source_drift_disposition'] = certificate['status']
        certificate['status'] = 'INCONCLUSIVE_SOURCE_DRIFT'
        certificate['source_drift'] = drift
        certificate['scope_statuses']['exact_non_gaussian_population'] = 'INCONCLUSIVE_SOURCE_DRIFT'
        for row in certificate['rows']:
            row['computed_status_before_source_drift'] = row['status']
            row['computed_coefficient_status_before_source_drift'] = row['coefficient_status']
            row['status'] = 'INCONCLUSIVE_SOURCE_DRIFT'
            row['coefficient_status'] = 'NOT_ACCEPTED_SOURCE_DRIFT'
            if row.get('finite_void'):
                row['finite_void']['computed_status_before_source_drift'] = row['finite_void']['status']
                row['finite_void']['status'] = 'NOT_ACCEPTED_SOURCE_DRIFT'


def full_declared_campaign(certificate, max_evaluations, time_budget, drift):
    return (not drift and max_evaluations == 80000 and time_budget == 900 and len(certificate['rows']) == 8
            and all(row.get('direct') is not None and row['direct']['status'] == 'CERTIFIED'
                    for row in certificate['rows']))


def _campaign(pc, max_evaluations, deadline):
    controls, guards = pc.negative_controls(), pc.guard_witnesses()
    b, target = pc.coefficient_interval(), pc.poisson_target()
    sound = controls['status'] == 'PASS' and guards['status'] == 'PASS_RATIONAL_GUARDS'
    rows, statuses = [], []
    if not sound:
        statuses.append('FAIL_IMPLEMENTATION')
    for j in range(1, 9):
        tau, n = Q(1, 1 << (3 * j)), 4 ** j
        row = {'j': j, 'tau': tau, 'n': n, 'schedule_identity': Q(n, 4 ** j),
               'status': 'INCONCLUSIVE_BUDGET', 'coefficient_status': 'NOT_RUN',
               'direct': None, 'cdf_mass': None, 'cdf_normalized': None,
               'intersection_normalized': None, 'void_interval': None}
        if time.monotonic() >= deadline:
            row['reason'] = 'Whole-run monotonic deadline exhausted before row'
            rows.append(row)
            statuses.append(row['status'])
            continue
        try:
            direct = pc.direct_mass(j, max_evaluations=max_evaluations, deadline=deadline)
            row['direct'] = direct
            if direct['status'] != 'CERTIFIED':
                row['status'] = direct['status']
            else:
                cdf = pc.cdf_mass(tau)
                normalized = tuple(n * x for x in cdf)
                observed = (max(direct['normalized_interval'][0], normalized[0]),
                            min(direct['normalized_interval'][1], normalized[1]))
                row.update({'cdf_mass': cdf, 'cdf_normalized': normalized,
                            'cdf_width': normalized[1] - normalized[0],
                            'cdf_width_gate': normalized[1] - normalized[0] <= pc.WIDTH_LIMIT})
                if not row['cdf_width_gate']:
                    row['status'] = 'INCONCLUSIVE_PRECISION'
                elif observed[0] > observed[1]:
                    row['status'] = 'FAIL_ROUTE_CONSISTENCY'
                elif not sound:
                    row['status'] = 'FAIL_IMPLEMENTATION'
                else:
                    mass = tuple(x / n for x in observed)
                    error = pc.error_interval(j)
                    envelopes = pc.coefficient_envelopes(b, error)
                    void = pc.void_interval(mass, n)
                    row.update({'intersection_normalized': observed,
                                'intersection_mass': mass, 'coefficient_interval': b,
                                'error_interval': error, 'predicted_envelopes': envelopes,
                                'outer_envelope_distance': pc.interval_distance(observed, envelopes['outer']),
                                'coefficient_status': pc.classify_coefficient(observed, b, error),
                                'void_interval': void, 'void_width': void[1] - void[0],
                                'void_width_gate': void[1] - void[0] <= pc.WIDTH_LIMIT,
                                'target_interval': target,
                                'target_width_gate': target[1] - target[0] <= pc.WIDTH_LIMIT})
                    row['status'] = row['coefficient_status']
                    if not row['void_width_gate'] or not row['target_width_gate']:
                        row['status'] = 'INCONCLUSIVE_PRECISION'
                    if j == 8:
                        gates = {'error_allowance': error[1] <= Q(41, 524288),
                                 'mass_gate': observed[1] < Q(1, 3),
                                 'allowance_identity': Q(41, 524288) + Q(1, 1048576)
                                 + 2 * pc.WIDTH_LIMIT == Q(115, 1048576) < Q(1, 8192)}
                        row['finite_void'] = {
                            'gates': gates, 'poissonization_error_upper': Q(1, 1048576),
                            'declared_allowance': Q(115, 1048576),
                            'endpoint_distance': max(abs(void[0] - target[1]), abs(void[1] - target[0])),
                            'minimum_distance': pc.interval_distance(void, target),
                            'status': pc.classify_void(void, target) if all(gates.values())
                            else 'INCONCLUSIVE_PRECISION'}
                        row['status'] = _overall([row['status'], row['finite_void']['status']])
            if time.monotonic() >= deadline:
                row['status'] = 'INCONCLUSIVE_BUDGET'
                row['reason'] = 'Whole-run monotonic deadline exhausted during row'
        except ValueError as exc:
            row['status'] = 'FAIL_IMPLEMENTATION'
            row['error'] = str(exc)
        rows.append(row)
        statuses.append(row['status'])
    return {'schema_version': 1, 'status': _overall(statuses), 'rows': rows,
            'negative_controls': controls, 'rational_guards': guards,
            'coefficient_interval': b, 'poisson_target_interval': target,
            'scope_statuses': {
                'exact_non_gaussian_population': _overall(statuses),
                'rational_guard_consequents': guards['status'],
                'analytic_cutoff_derivative_premise': 'REVIEW_REQUIRED',
                'complete_surgery_field': 'NOT_RUN', 'independent_global_h0_selector': 'NOT_RUN',
                'gaussian_universality': 'NOT_RUN', 'blind_50_campaign': 'NOT_RUN',
                'higher_homology': 'NOT_RUN', 'same_field_spatial_poisson': 'NOT_RUN',
                'lean_kernel': 'NOT_RUN', 'scientific_promotion': 'NOT_RUN',
                'organizational_independence_credit': 0},
            'arithmetic': {'mathematical_values': 'Fraction', 'outward_bits': pc.BITS,
                           'split_root_bisections': 96, 'core_panels': pc.CORE_PANELS,
                           'tail_panels_per_octave': pc.TAIL_PANELS,
                           'point_rounding': 'outward fixed 128-bit dyadic',
                           'void_method': 'shared outward log_bounds and exp_neg_bounds'},
            'analytic_premises': ['consumed global-selection proof', 'iid-copy law',
                                 'core monotonicity and tail convexity derivations'],
            'exposure': 'Source-exposed OpenAI/Codex provider/team; no organizational independence'}


def render_report(certificate):
    lines = ['# Frozen exact cusp-population attempt', '',
             'Disposition: **' + certificate['status'] + '**.', '',
             'This deterministic interval calculation addresses the fixed whole-rectangle',
             'non-Gaussian population, eight coefficient predictions and one row-8 finite',
             'iid-copy void prediction. It has no random seed or numerical field draw.', '',
             '| j | tau | Direct normalized interval | CDF normalized interval | Coefficient | Row disposition | Evaluations |',
             '|---|---|---|---|---|---|---|']
    if certificate.get('error'):
        lines[4:4] = ['Execution error: ' + certificate['error'] + '.', '']
    if certificate.get('runtime_budget_exhausted_phase'):
        lines[4:4] = ['Final runtime budget exhausted after source hashing and artifact',
                      'serialization preparation. Previously computed disposition: ' +
                      certificate['computed_status_before_runtime_budget'] + '.', '']
    def display(value):
        return 'NOT_RUN' if value is None else '[' + ', '.join(str(x) for x in value) + ']'
    for row in certificate['rows']:
        direct = row.get('direct')
        lines.append('| ' + ' | '.join((str(row['j']), str(row['tau']),
                     display(direct['normalized_interval'] if direct else None),
                     display(row.get('cdf_normalized')), row['coefficient_status'],
                     row['status'], str(direct['evaluations'] if direct else 0))) + ' |')
    lines += ['', 'All mathematical endpoints and exact arithmetic, discretization and',
              'root-sliver widths are retained in CERTIFICATE.json. A partial sum is',
              'reported as partial work and never used as a probability enclosure.', '']
    for row in certificate['rows']:
        if row.get('reason') or row.get('error'):
            lines.append('Row ' + str(row['j']) + ': ' + row.get('reason', row.get('error', '')) + '.')
        if row.get('finite_void'):
            void = row['finite_void']
            lines += ['Row-8 void interval: ' + display(row['void_interval']) + '.',
                      'Target exp(-B): ' + display(row['target_interval']) + '.',
                      'Maximum cross-endpoint distance: ' + str(void['endpoint_distance']) +
                      '; strict threshold 1/8192; disposition ' + void['status'] + '.']
    lines += ['', 'Scope statuses:', '']
    lines += ['- ' + name + ': ' + str(status) for name, status in certificate['scope_statuses'].items()]
    lines += ['', 'The rational guards check consequences of analytic cutoff derivative',
              'premises. Construction/evaluation of the complete surgery field and an',
              'independent global H0 selector remain NOT_RUN. Shared outward elementary',
              'primitives and source-exposed same-provider reviews provide zero',
              'organizational independence credit. PASS does not admit a theorem or',
              'promote scientific status. Source drift, failure or budget exhaustion',
              'remains attached to this original attempt.', '']
    return '\n'.join(lines).encode()


def run(output_dir=DEFAULT_OUTPUT, max_evaluations=80000, time_budget=900):
    _require(type(max_evaluations) is int and 0 <= max_evaluations <= 80000,
             'Evaluation budget may only be reduced')
    _require(type(time_budget) is int and 0 <= time_budget <= 900,
             'Whole-run budget may only be reduced')
    output_dir = Path(output_dir)
    # Refuse any existing attempt, including a partial or empty attempt directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    start_utc, start = _utc(), time.monotonic()
    before, pre_drift = {}, []
    try:
        before = freeze_sources()
        pre_drift = [name for name, expected in FIXED_HASHES.items()
                     if before[name]['sha256'] != expected]
        if pre_drift:
            certificate = _empty_certificate('INCONCLUSIVE_SOURCE_DRIFT',
                                             'Frozen contract/proof/dependency identity mismatch before evaluation')
        else:
            # Freeze prior to loading executable population source for this invocation.
            import population_certificate as pc
            certificate = _campaign(pc, max_evaluations, start + time_budget)
    except (ImportError, OSError) as exc:
        certificate = _empty_certificate('SETUP_BLOCKED', type(exc).__name__ + ': ' + str(exc))
    except Exception as exc:
        certificate = _empty_certificate('FAIL_IMPLEMENTATION', type(exc).__name__ + ': ' + str(exc))
    drift = sorted(set(pre_drift + source_drift(before)))
    after = {}
    for path in SOURCE_PATHS:
        name = str(path.relative_to(REPOSITORY))
        try:
            after.update(freeze_sources([path]))
        except OSError as exc:
            after[name] = {'error': type(exc).__name__ + ': ' + str(exc)}
    apply_source_drift(certificate, drift)
    end_utc, elapsed = _utc(), time.monotonic() - start
    exit_code = 0 if certificate['status'] == 'PASS' else (1 if certificate['status'].startswith('FAIL')
                                                         or certificate['status'] == 'FALSIFIED' else 2)
    certificate_bytes = canonical_bytes(certificate)
    report_bytes = render_report(certificate)
    run_record = {'schema_version': 1, 'start_utc': start_utc, 'end_utc': end_utc,
                  'runtime_seconds': format(elapsed, '.6f'), 'command': COMMAND,
                  'timer_scope': 'Acceptance deadline covers source freezing, import, campaign, '
                  'post-source hashing and serialization preparation of all three artifacts. '
                  'runtime_seconds is sampled before final RUN encoding; exclusive artifact '
                  'writes and flush are excluded. Final deadline is checked after RUN encoding.',
                  'final_runtime_budget_gate': 'PASS',
                  'exit_code': exit_code, 'status': certificate['status'],
                  'sources_before': before, 'sources_after': after, 'source_drift': drift,
                  'performer': 'OpenAI/Codex population certificate runner',
                  'exposure': 'Source-exposed same provider/team',
                  'organizational_independence_credit': 0,
                  'python': sys.version, 'full_declared_campaign':
                  full_declared_campaign(certificate, max_evaluations, time_budget, drift),
                  'invocation_scope': 'declared full attempt' if max_evaluations == 80000
                  and time_budget == 900 else 'reduced-budget focused control',
                  'parameters': {'d': 2, 'L': Q(1), 'A': Q(1, 1 << 16),
                                 'u0': Q(3, 1 << 32), 'v0': Q(2, 1 << 48),
                                 'rectangle_density': Q(1 << 80, 24),
                                 'whole_rectangle_bar_mass': Q(1, 5),
                                 'evaluation_budget_per_row': max_evaluations,
                                 'wall_clock_budget_seconds': time_budget,
                                 'rows': list(range(1, 9)), 'rng': 'NONE'},
                  'actual_route_evaluations': sum(row['direct']['evaluations']
                                                 for row in certificate['rows'] if row.get('direct')),
                  'outputs': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                              for name, raw in (('CERTIFICATE.json', certificate_bytes),
                                                ('RESULTS.md', report_bytes))}}
    # All three byte payloads must exist before the final acceptance check. This
    # catches time spent in final source hashing, reporting and JSON preparation.
    run_bytes = canonical_bytes(run_record)
    if apply_final_runtime_budget(certificate, start, time_budget):
        # An exhausted attempt remains exhausted; do not recheck/loop while
        # preparing its retained inconclusive receipt.
        certificate_bytes = canonical_bytes(certificate)
        report_bytes = render_report(certificate)
        run_record.update({'status': certificate['status'], 'exit_code': 2,
                           'full_declared_campaign': False,
                           'final_runtime_budget_gate': 'INCONCLUSIVE_BUDGET',
                           'end_utc': _utc(),
                           'runtime_seconds': format(time.monotonic() - start, '.6f'),
                           'outputs': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                                       for name, raw in (('CERTIFICATE.json', certificate_bytes),
                                                         ('RESULTS.md', report_bytes))}})
        run_bytes = canonical_bytes(run_record)
    for name, raw in (('CERTIFICATE.json', certificate_bytes), ('RESULTS.md', report_bytes),
                      ('RUN.json', run_bytes)):
        with (output_dir / name).open('xb') as output:
            output.write(raw)
    return run_record


if __name__ == '__main__':
    try:
        result = run()
        print(result['status'] + '; direct evaluations=' + str(result['actual_route_evaluations'])
              + '; runtime_seconds=' + result['runtime_seconds'])
        raise SystemExit(result['exit_code'])
    except FileExistsError as exc:
        print('REFUSED_EXISTING_ATTEMPT: ' + str(exc), file=sys.stderr)
        raise SystemExit(3)
