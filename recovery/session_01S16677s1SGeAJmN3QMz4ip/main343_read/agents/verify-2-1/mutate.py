"""Own single-substitution mutants of the PR's runner, run under the CI command. Usage: mutate.py <root> <out>"""
import re, shutil, subprocess, sys
from pathlib import Path
root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
RUN = 'experiments/universality/two_parameter_cusp_population/run_population_certificate.py'
CMD = ['python3.12', '-B', '-E', '-S', '-m', 'unittest', 'discover', '-s',
       'experiments/universality/two_parameter_cusp_population', '-p', 'test_population_certificate.py', '-v']
M = {
 'M17_overlap_elif_false': ("                elif observed[0] > observed[1]:", "                elif False:"),
 'M18_direct_only_intersection': ("                observed = (max(direct['normalized_interval'][0], normalized[0]),\n                            min(direct['normalized_interval'][1], normalized[1]))",
                                  "                observed = tuple(direct['normalized_interval'])"),
 'M19_row_status_pass': ("                    row['status'] = row['coefficient_status']", "                    row['status'] = 'PASS'"),
 'M20_overall_drops_falsified': ("                   'INCONCLUSIVE_PRECISION', 'FALSIFIED'):", "                   'INCONCLUSIVE_PRECISION'):"),
 'M25_mass_gate_true': ("                                 'mass_gate': observed[1] < Q(1, 3),", "                                 'mass_gate': True,"),
 'M28_ignore_void_width_gates': ("                    if not row['void_width_gate'] or not row['target_width_gate']:", "                    if False:"),
 'M26_exit_zero_on_falsified': ("    exit_code = 0 if certificate['status'] == 'PASS' else", "    exit_code = 0 if certificate['status'] in ('PASS', 'FALSIFIED') else"),
 'M24_runner_n_diverges': ("        tau, n = Q(1, 1 << (3 * j)), 4 ** j", "        tau, n = Q(1, 1 << (3 * j)), 2 ** j"),
}
out.mkdir(exist_ok=True)
for name, (old, new) in M.items():
    dest = out / name
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(root, dest)
    t = dest / RUN
    s = t.read_text()
    assert s.count(old) == 1, (name, s.count(old))
    t.write_text(s.replace(old, new))
    p = subprocess.run(CMD, cwd=dest, capture_output=True, text=True, timeout=600)
    o = p.stdout + p.stderr
    failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+)', o, re.M)))
    ran = re.search(r'^Ran (\d+) tests', o, re.M).group(1)
    print(f"{'KILLED' if p.returncode else 'SURVIVED':8} {name:32} rc={p.returncode} ran={ran} failed={failed}", flush=True)
