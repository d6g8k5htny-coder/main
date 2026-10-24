"""Single-point mutation harness for PR main#350 test-adequacy review (READ-ONLY on wt350).

Each mutation: (id, file, old, new, expectation-note). Applied one at a time to the
scratch copy, the suite is run, failing test ids are recorded, the file is restored.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/p350/agents/tests-mutation')
POP = ROOT / 'experiments/universality/two_parameter_cusp_population'
PRISTINE = ROOT / 'pristine/pop'
CMD = [sys.executable, '-B', '-E', '-S', '-m', 'unittest', 'discover', '-s',
       'experiments/universality/two_parameter_cusp_population', '-p', 'test_*.py']

MUTATIONS = [
    # ---- surgery_field.py ----
    ('SF1_outward_inward', 'surgery_field.py',
     "    return Q(a.numerator // a.denominator, scale), Q(-((-b.numerator) // b.denominator), scale)",
     "    return Q(-((-a.numerator) // a.denominator), scale), Q(b.numerator // b.denominator, scale)",
     'interval bound inward instead of outward'),
    ('SF2_sine_remainder_dropped', 'surgery_field.py',
     "    lo, hi = _outward(total[0] - SINE_REMAINDER, total[1] + SINE_REMAINDER)",
     "    lo, hi = _outward(total[0], total[1])",
     'Taylor truncation remainder not added to sine enclosure'),
    ('SF3_height_guard_weakened', 'surgery_field.py',
     "        'height_guard': bounds['height_perturbation'] < DELTA/4,",
     "        'height_guard': bounds['height_perturbation'] < DELTA*4,",
     'guard constant changed (consequent made 16x weaker)'),
    ('SF4_bind_record_skips_guard', 'surgery_field.py',
     "    budget.checkpoint('field_source')\n    guard_certificate()\n",
     "    budget.checkpoint('field_source')\n",
     'field evaluation no longer re-binds analytic source on each call'),
    ('SF5_checkpoint_strict', 'surgery_field.py',
     "        if time.monotonic() >= self._deadline:",
     "        if time.monotonic() > self._deadline:",
     'deadline inequality flipped (exact deadline admitted)'),
    ('SF6_dependency_identity_dropped', 'surgery_field.py',
     "            _fc.BITS != BITS or _gt.EXP_BITS != 256 or _gt.fc is not _fc):",
     "            False):",
     'imported primitive identity/precision check dropped'),
    ('SF7_exp_dyadic_threshold', 'surgery_field.py',
     "    if reciprocal > 4096:",
     "    if reciprocal > 2048:",
     'dyadic exp shortcut threshold lowered (2^-4096 claim unsound for 2048<1/t<2839)'),
    ('SF8_width_gate_loosened', 'surgery_field.py',
     "    record['gates']['width'] = result[1]-result[0] <= FIELD_WIDTH",
     "    record['gates']['width'] = result[1]-result[0] <= 2**20*FIELD_WIDTH",
     'field width gate loosened by 2^20'),
    # ---- global_h0_selector.py ----
    ('GS1_variation_counts_zeros', 'global_h0_selector.py',
     "        if y:\n            signs.append(1 if y > 0 else -1)",
     "        signs.append(1 if y > 0 else -1)",
     'Sturm variation counts interior zeros as negative sign'),
    ('GS2_elder_tiebreak_flipped_producer', 'global_h0_selector.py',
     "                survivor = min(incoming, key=lambda birth: (rank[heights[birth]['group_id']], birth))",
     "                survivor = min(incoming, key=lambda birth: (rank[heights[birth]['group_id']], -birth))",
     'elder tie-break by ordinal flipped in producer only'),
    ('GS3_verify_completeness_trivial', 'global_h0_selector.py',
     "        _check(cnt == supplied, phase+'/complete'+str(i), checks)",
     "        _check(True, phase+'/complete'+str(i), checks)",
     'verifier root-completeness check made trivially true (name retained)'),
    ('GS4_projection_degenerate_flipped', 'global_h0_selector.py',
     "    reason = 'EXCLUDED_DEGENERATE' if discriminant == 0 else",
     "    reason = 'EXCLUDED_DEGENERATE' if discriminant < 0 else",
     'population projection labels degenerate/non-wedge swapped'),
    ('GS5_tie_branch_skipped', 'global_h0_selector.py',
     "    elif v == 0 and u > 0:\n        # This identity is proved",
     "    elif False:\n        # This identity is proved",
     'elder-rule equal-height tie identity branch skipped'),
    ('GS6_verify_guard_replay_dropped', 'global_h0_selector.py',
     "        _check(_same(certificate['guard'],guard_certificate()),'guard/source_bound_rational_replay',checks)\n",
     "",
     'verifier no longer replays guard certificate'),
    ('GS7_elder_rank_flipped_both', 'global_h0_selector.py',
     None, None, 'elder rank (older survives) flipped in BOTH producer and verifier'),
    ('GS8_verify_isolator_width_dropped', 'global_h0_selector.py',
     "        _check(domain[0] <= lo <= hi <= domain[1] and hi-lo <= ROOT_WIDTH, phase+'/enclosure'+str(i), checks)",
     "        _check(domain[0] <= lo <= hi <= domain[1], phase+'/enclosure'+str(i), checks)",
     'verifier isolator width bound dropped'),
    ('GS9_connectivity_loop_early_return', 'global_h0_selector.py',
     "        if value > 0:\n            if index == 0 or index == len(rootset['roots']):",
     "        if value > 0 and index > 1:\n            if index == 0 or index == len(rootset['roots']):",
     'regular-level connectivity loop skips the first positive component'),
    # ---- run_global_h0_selector.py ----
    ('RN1_withdrawal_on_failure_removed', 'run_global_h0_selector.py',
     "    if certificate['status'] not in ('CERTIFIED', 'NOT_RUN'):\n        certificate = withdraw_acceptance(certificate, certificate['status'], phase='attempt_failure', detail=certificate['failure'])\n",
     "",
     'withdrawn attempt keeps nested CERTIFIED statuses'),
    ('RN2_core_overlap_trivial', 'run_global_h0_selector.py',
     "            checks['core_identity_overlap'] &= max(shifted[0], interval[0]) <= min(shifted[1], interval[1])",
     "            checks['core_identity_overlap'] &= True",
     'root-box field/height overlap check trivialised'),
    ('RN3_core_anchor_identity_trivial', 'run_global_h0_selector.py',
     "            checks['core_identity_overlap'] &= interval[0] <= expected <= interval[1]",
     "            checks['core_identity_overlap'] &= True",
     'core anchor exact-identity containment trivialised'),
    ('RN4_positive_signs_gate_trivial', 'run_global_h0_selector.py',
     "        'positive_signs': replay('levels/regular_sign'),",
     "        'positive_signs': True,",
     'soundness gate positive_signs trivially PASS'),
    ('RN5_periodic_translation_trivial', 'run_global_h0_selector.py',
     "        checks['periodic_translation'] &= interval == translated",
     "        checks['periodic_translation'] &= True",
     'periodic translation check trivialised'),
    ('RN6_drift_recheck_dropped', 'run_global_h0_selector.py',
     "    drift = source_drift(manifest, base=base)",
     "    drift = []",
     'post-computation source drift re-check dropped'),
    ('RN7_fixture13_expectation_changed', 'run_global_h0_selector.py',
     "    younger = (0, 2, 0, 2, 0, 2, None, None, None, None, None, None, 2, 0, None, None, None, None)",
     "    younger = (0, 2, 0, 2, 0, 2, None, None, None, None, None, None, 0, 0, None, None, None, None)",
     'fixture_13 younger-root expectation changed'),
    ('RN8_deadline_strict', 'run_global_h0_selector.py',
     "    if now-start >= seconds:",
     "    if now-start > seconds:",
     'final deadline inequality flipped'),
    ('RN9_sourced_fixture_id_dropped', 'run_global_h0_selector.py',
     "            and record.get('fixture_id') == row['fixture_id'] and record.get('status') == 'CERTIFIED'",
     "            and record.get('status') == 'CERTIFIED'",
     'field record fixture binding dropped from soundness'),
    ('RN10_verified_all_true_dropped', 'run_global_h0_selector.py',
     "    verified = verification.get('status') == 'PASS' and bool(vc) and all(value is True for value in vc.values())",
     "    verified = verification.get('status') == 'PASS' and bool(vc)",
     'soundness no longer requires every verifier check True'),

    ('GS10_endpoint_extraction_skipped', 'global_h0_selector.py',
     "                if _eval(remaining, endpoint, budget, prefix + '/' + label) == 0:",
     "                if _eval(remaining, endpoint, budget, prefix + '/' + label) == 0 and False:",
     'root isolation: exact endpoint roots not extracted (positive control)'),
    ('GS11_separation_nonstrict', 'global_h0_selector.py',
     "        if any(a['isolator'][1] >= b['isolator'][0] for a, b in zip(result['roots'], result['roots'][1:])):",
     "        if any(a['isolator'][1] > b['isolator'][0] for a, b in zip(result['roots'], result['roots'][1:])):",
     'root isolation: touching isolators admitted (inequality flipped)'),
    ('GS12_root_width_accept_wider', 'global_h0_selector.py',
     "                    if number == 1 and hi-lo <= ROOT_WIDTH:",
     "                    if number == 1 and hi-lo <= 2**8*ROOT_WIDTH:",
     'root isolation: isolator accepted 256x wider than ROOT_WIDTH'),
    ('RN11_younger_comparison_dropped', 'run_global_h0_selector.py',
     "    if younger is not None and (len(bars) != 1 or type(bars[0]['birth_root']) is not int or bars[0]['birth_root'] != younger):",
     "    if False:",
     'compare_expectations: younger-root-ordinal comparison dropped'),
    ('RN12_count_comparison_dropped', 'run_global_h0_selector.py',
     "    if type(count) is not int or count != expected['population_count']:",
     "    if False:",
     'compare_expectations: population count comparison dropped'),
]


def apply(file, old, new):
    path = POP / file
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{file}: expected exactly one occurrence, found {count} for {old[:60]!r}')
    path.write_text(text.replace(old, new))


def restore(file):
    shutil.copy2(PRISTINE / file, POP / file)


def run_suite():
    proc = subprocess.run(CMD, cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\S+) \(([\w.]+)\)', out, re.M)))
    ids = sorted({f'{cls}.{name}' for name, cls in failed})
    ran = re.search(r'Ran (\d+) tests', out)
    summary = re.search(r'^(OK|FAILED.*)$', out, re.M)
    return {'returncode': proc.returncode, 'ran': int(ran.group(1)) if ran else None,
            'summary': summary.group(1) if summary else out[-500:], 'failing_tests': ids}


def main(selected=None):
    results = []
    for mid, file, old, new, note in MUTATIONS:
        if selected and mid not in selected:
            continue
        try:
            if mid == 'GS7_elder_rank_flipped_both':
                apply(file, "                survivor = min(incoming, key=lambda birth: (rank[heights[birth]['group_id']], birth))",
                      "                survivor = max(incoming, key=lambda birth: (rank[heights[birth]['group_id']], birth))")
                apply(file, "                survivor = sorted(incoming,key=lambda n:(rank[heights[n]['group_id']],n))[0]",
                      "                survivor = sorted(incoming,key=lambda n:(rank[heights[n]['group_id']],n))[-1]")
            else:
                apply(file, old, new)
            result = run_suite()
        finally:
            restore(file)
        result.update(id=mid, file=file, note=note)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('id', 'file', 'note', 'ran', 'summary', 'failing_tests')}), flush=True)
    (ROOT / ('mutation_results_extra.json' if selected else 'mutation_results.json')).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main(set(sys.argv[1:]) or None)
