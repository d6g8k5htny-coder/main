"""Single-point mutation harness. Each mutant is an isolated copy of root/ with
exactly one textual substitution (verified to occur exactly once), then the
declared CI step is run in that copy. Never touches the worktree."""
import os, re, shutil, subprocess, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'root'
MUT = HERE / 'mutants'
PC = 'experiments/universality/two_parameter_cusp_population/population_certificate.py'
RUN = 'experiments/universality/two_parameter_cusp_population/run_population_certificate.py'
CMD = ['python3.12', '-B', '-E', '-S', '-m', 'unittest', 'discover', '-s',
       'experiments/universality/two_parameter_cusp_population', '-p', 'test_population_certificate.py', '-v']

MUTATIONS = [
 # --- category: make an interval bound inward ---
 ('M01_outward_upper_floors', PC,
  "            Q(-((-upper.numerator) // upper.denominator), SCALE))",
  "            Q(upper.numerator // upper.denominator, SCALE))"),
 ('M02_sqrt_lower_rounds_up', PC,
  "    return sqrt_bounds(a[0], BITS)[0], sqrt_bounds(a[1], BITS)[1]",
  "    return sqrt_bounds(a[0], BITS)[1], sqrt_bounds(a[1], BITS)[1]"),
 ('M03a_void_interval_inward', PC,
  "        return exp_neg_bounds(xhi)[0], exp_neg_bounds(xlo)[1]",
  "        return exp_neg_bounds(xhi)[1], exp_neg_bounds(xlo)[0]"),
 ('M03b_poisson_target_inward', PC,
  "    return exp_neg_bounds(b[1])[0], exp_neg_bounds(b[0])[1]",
  "    return exp_neg_bounds(b[1])[1], exp_neg_bounds(b[0])[0]"),
 # --- category: drop a panel error term ---
 ('M04_drop_root_sliver', PC,
  "    normalized = core_lower + tail_low, core_upper + tail_upper_ceiling + sliver",
  "    normalized = core_lower + tail_low, core_upper + tail_upper_ceiling"),
 ('M05_core_upper_uses_left_endpoints', PC,
  "            if index > 0:\n                core_upper_floor += lo",
  "            if index < CORE_PANELS:\n                core_upper_floor += lo"),
 ('M06_tail_upper_uses_midpoint_sum', PC,
  "            tail_upper_floor += step * Q(endpoint_floor, 2 * SCALE)\n            tail_upper_ceiling += step * Q(endpoint_ceiling, 2 * SCALE)",
  "            tail_upper_floor += step * Q(midpoint_lower, SCALE)\n            tail_upper_ceiling += step * Q(midpoint_lower + midpoint_width, SCALE)"),
 ('M16_direct_status_always_certified', PC,
  "    result.update({'status': 'CERTIFIED' if all(gates.values()) else 'INCONCLUSIVE_PRECISION',",
  "    result.update({'status': 'CERTIFIED',"),
 # --- category: overlap predicate trivially true ---
 ('M17_overlap_elif_false', RUN,
  "                elif observed[0] > observed[1]:",
  "                elif False:"),
 ('M18_ignore_cdf_route_in_intersection', RUN,
  "                observed = (max(direct['normalized_interval'][0], normalized[0]),\n                            min(direct['normalized_interval'][1], normalized[1]))",
  "                observed = tuple(direct['normalized_interval'])"),
 # --- category: flip a falsification predicate inequality ---
 ('M08_falsify_low_side_at_inner', PC,
  "    if observed[1] < outer[0] or observed[0] > outer[1]:",
  "    if observed[1] < inner[0] or observed[0] > outer[1]:"),
 ('M08b_falsify_high_side_at_inner', PC,
  "    if observed[1] < outer[0] or observed[0] > outer[1]:",
  "    if observed[1] < outer[0] or observed[0] > inner[1]:"),
 ('M09_void_pass_nonstrict', PC,
  "    if max(abs(observed[0] - target[1]), abs(observed[1] - target[0])) < Q(1, 8192):",
  "    if max(abs(observed[0] - target[1]), abs(observed[1] - target[0])) <= Q(1, 8192):"),
 ('M09b_void_falsify_nonstrict', PC,
  "    if interval_distance(observed, target) > Q(1, 8192):",
  "    if interval_distance(observed, target) >= Q(1, 8192):"),
 ('M15_width_limit_256x', PC,
  "WIDTH_LIMIT = Q(1, 1 << 16)",
  "WIDTH_LIMIT = Q(1, 1 << 8)"),
 # --- category: skip a negative control ---
 ('M10_drop_reject_fixed_left_gate', PC,
  "             'reject_fixed_left': all(c['lifetime'] != 32 * a ** 4 for c in adversarial),\n",
  ""),
 ('M11_negative_controls_always_pass', PC,
  "    return {'status': 'PASS' if all(gates.values()) else 'FAIL_IMPLEMENTATION',\n            'controls': controls,",
  "    return {'status': 'PASS',\n            'controls': controls,"),
 ('M11b_adversarial_accepts_fixed_left', PC,
  "             'actual_theta2_gap': all(c['lifetime'] == 3 * a ** 4 / 4 for c in adversarial),",
  "             'actual_theta2_gap': all(c['lifetime'] == 32 * a ** 4 for c in adversarial),"),
 # --- category: drop the source-drift check ---
 ('M22_drop_post_run_drift_check', RUN,
  "    drift = sorted(set(pre_drift + source_drift(before)))",
  "    drift = sorted(set(pre_drift))"),
 ('M23_drop_pre_run_frozen_hash_check', RUN,
  "        pre_drift = [name for name, expected in FIXED_HASHES.items()\n                     if before[name]['sha256'] != expected]",
  "        pre_drift = []"),
 ('M29_apply_source_drift_noop', RUN,
  "def apply_source_drift(certificate, drift):\n    if drift:",
  "def apply_source_drift(certificate, drift):\n    if False:"),
 # --- category: change a schedule constant ---
 ('M12_schedule_n_halved_exponent', PC,
  "    return Q(1, 1 << (3 * j)), 4 ** j",
  "    return Q(1, 1 << (3 * j)), 2 ** j"),
 ('M13_schedule_tau_halved', PC,
  "    return Q(1, 1 << (3 * j)), 4 ** j",
  "    return Q(1, 1 << (3 * j + 1)), 4 ** j"),
 ('M14_tail_panels_256', PC,
  "TAIL_PANELS = 512",
  "TAIL_PANELS = 256"),
 ('M24_runner_schedule_n_diverges_from_pc', RUN,
  "        tau, n = Q(1, 1 << (3 * j)), 4 ** j",
  "        tau, n = Q(1, 1 << (3 * j)), 2 ** j"),
 # --- category: let a failed prediction still report PASS ---
 ('M19_row_status_pass_regardless', RUN,
  "                    row['status'] = row['coefficient_status']",
  "                    row['status'] = 'PASS'"),
 ('M20_overall_drops_falsified', RUN,
  "                   'INCONCLUSIVE_PRECISION', 'FALSIFIED'):",
  "                   'INCONCLUSIVE_PRECISION'):"),
 ('M21_overall_pass_if_any_pass', RUN,
  "    return 'PASS' if statuses and all(x == 'PASS' for x in statuses) else 'INCONCLUSIVE_PRECISION'",
  "    return 'PASS' if statuses and any(x == 'PASS' for x in statuses) else 'INCONCLUSIVE_PRECISION'"),
 ('M25_j8_mass_gate_trivial', RUN,
  "                                 'mass_gate': observed[1] < Q(1, 3),",
  "                                 'mass_gate': True,"),
 ('M26_exit_code_zero_on_falsified', RUN,
  "    exit_code = 0 if certificate['status'] == 'PASS' else",
  "    exit_code = 0 if certificate['status'] in ('PASS', 'FALSIFIED') else"),
 ('M28_ignore_void_width_gates', RUN,
  "                    if not row['void_width_gate'] or not row['target_width_gate']:",
  "                    if False:"),
 ('M27_full_campaign_ignores_budget_args', RUN,
  "    return (not drift and max_evaluations == 80000 and time_budget == 900 and len(certificate['rows']) == 8",
  "    return (not drift and len(certificate['rows']) == 8"),
 ('M31_cdf_closed_form_coefficient_error', PC,
  "    second = _subtract(_scale(_subtract(inverse_seventh, (Q(1), Q(1))), Q(36, 7)),",
  "    second = _subtract(_scale(_subtract(inverse_seventh, (Q(1), Q(1))), Q(36, 8)),"),
 ('M34_both_routes_missing_factor_two_TWO_POINT', PC,
  ("    result = _outward(_scale(_add(first, _multiply(coefficient, second)), Q(1, 15)))",
   "                           Q(n, 15) * rlo ** 2 * node * (9 - theta2) * base ** 2)"),
  ("    result = _outward(_scale(_add(first, _multiply(coefficient, second)), Q(1, 30)))",
   "                           Q(n, 30) * rlo ** 2 * node * (9 - theta2) * base ** 2)")),
 ('M34b_both_routes_missing_factor_two_THREE_POINT', PC,
  ("    result = _outward(_scale(_add(first, _multiply(coefficient, second)), Q(1, 15)))",
   "                           Q(n, 15) * rlo ** 2 * node * (9 - theta2) * base ** 2)",
   "                                            rinverse), Q(n, 15)))"),
  ("    result = _outward(_scale(_add(first, _multiply(coefficient, second)), Q(1, 30)))",
   "                           Q(n, 30) * rlo ** 2 * node * (9 - theta2) * base ** 2)",
   "                                            rinverse), Q(n, 30)))")),
 ('M30_final_runtime_budget_never_fires', RUN,
  "    if time.monotonic() < start + time_budget:\n        return False",
  "    if True:\n        return False"),
]

def make_mutant(name, rel, old, new):
    dest = MUT / name
    if dest.exists():
        shutil.rmtree(dest)
    (dest / 'experiments/periodic_h0').mkdir(parents=True)
    for p in (ROOT / 'experiments/periodic_h0').glob('*.py'):
        shutil.copy2(p, dest / 'experiments/periodic_h0' / p.name)
    shutil.copytree(ROOT / 'experiments/universality/two_parameter_cusp_population',
                    dest / 'experiments/universality/two_parameter_cusp_population')
    target = dest / rel
    src = target.read_text()
    pairs = list(zip(old, new)) if isinstance(old, tuple) else [(old, new)]
    for o, nw in pairs:
        n = src.count(o)
        if n != 1:
            raise SystemExit(f'{name}: pattern occurs {n} times in {rel}')
        src = src.replace(o, nw)
    target.write_text(src)
    return dest

def run_tests(dest):
    t = time.time()
    p = subprocess.run(CMD, cwd=dest, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    (dest / 'tests.log').write_text(out)
    failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+)', out, re.M)))
    m = re.search(r'^Ran (\d+) tests', out, re.M)
    ran = int(m.group(1)) if m else -1
    return p.returncode, ran, failed, round(time.time() - t, 1)

if __name__ == '__main__':
    only = sys.argv[1:]
    MUT.mkdir(exist_ok=True)
    results = []
    for name, rel, old, new in MUTATIONS:
        if only and name not in only:
            continue
        dest = make_mutant(name, rel, old, new)
        rc, ran, failed, secs = run_tests(dest)
        verdict = 'KILLED' if rc != 0 else 'SURVIVED'
        results.append({'mutant': name, 'file': rel, 'verdict': verdict, 'rc': rc,
                        'ran': ran, 'failed_tests': failed, 'seconds': secs})
        print(f"{verdict:8} {name:42} rc={rc} ran={ran} {secs}s failed={failed}", flush=True)
    (HERE / ('mutation_results_extra.json' if only else 'mutation_results.json')).write_text(json.dumps(results, indent=2))
