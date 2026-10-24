"""Independent single-point mutants of the runner, built from verify-2-0/pristine.
Each substitution is required to occur exactly once. Runs the CI discover command."""
import shutil, subprocess, sys, re, time
from pathlib import Path
V = Path(__file__).resolve().parents[1]
ROOT = V / 'pristine'
MUT = V / 'mutants'
RUN = 'experiments/universality/two_parameter_cusp_population/run_population_certificate.py'
CMD = ['python3.12', '-B', '-S', '-m', 'unittest', 'discover', '-s',
       'experiments/universality/two_parameter_cusp_population', '-p', 'test_population_certificate.py']
M = {
 'M17': ("                elif observed[0] > observed[1]:", "                elif False:"),
 'M18': ("                observed = (max(direct['normalized_interval'][0], normalized[0]),\n                            min(direct['normalized_interval'][1], normalized[1]))",
         "                observed = tuple(direct['normalized_interval'])"),
 'M19': ("                    row['status'] = row['coefficient_status']", "                    row['status'] = 'PASS'"),
 'M20': ("                   'INCONCLUSIVE_PRECISION', 'FALSIFIED'):", "                   'INCONCLUSIVE_PRECISION'):"),
 'M24': ("        tau, n = Q(1, 1 << (3 * j)), 4 ** j", "        tau, n = Q(1, 1 << (3 * j)), 2 ** j"),
 'M25': ("                                 'mass_gate': observed[1] < Q(1, 3),", "                                 'mass_gate': True,"),
 'M26': ("    exit_code = 0 if certificate['status'] == 'PASS' else", "    exit_code = 0 if certificate['status'] in ('PASS', 'FALSIFIED') else"),
 'M27': ("    return (not drift and max_evaluations == 80000 and time_budget == 900 and len(certificate['rows']) == 8",
         "    return (not drift and len(certificate['rows']) == 8"),
 'M28': ("                    if not row['void_width_gate'] or not row['target_width_gate']:", "                    if False:"),
 # extra, mine: classify_coefficient result thrown away entirely -> status PASS and overall PASS fallthrough
 'MX1': ("                                'coefficient_status': pc.classify_coefficient(observed, b, error),",
         "                                'coefficient_status': 'PASS',"),
 # extra, mine: finite void classification ignored
 'MX2': ("                            'status': pc.classify_void(void, target) if all(gates.values())\n                            else 'INCONCLUSIVE_PRECISION'}",
         "                            'status': 'PASS'}"),
}
def build(name):
    dest = MUT / name
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(ROOT, dest)
    f = dest / RUN
    s = f.read_text()
    old, new = M[name]
    assert s.count(old) == 1, (name, s.count(old))
    f.write_text(s.replace(old, new))
    return dest
if __name__ == '__main__':
    names = sys.argv[1:] or list(M)
    MUT.mkdir(exist_ok=True)
    for name in names:
        dest = build(name)
        t = time.time()
        p = subprocess.run(CMD, cwd=dest, capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
        (dest / 'tests.log').write_text(out)
        ran = re.search(r'^Ran (\d+) tests', out, re.M)
        failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+)', out, re.M)))
        print(f"{'KILLED' if p.returncode else 'SURVIVED':8} {name} rc={p.returncode} ran={ran.group(1) if ran else '?'} {round(time.time()-t,1)}s failed={failed} ok_line={[l for l in out.splitlines() if l.startswith('OK') or l.startswith('FAILED')]}", flush=True)
