"""Independent check (b): NO stubs. Run the real run() in a child process with a reduced evaluation
budget; from the parent, append one byte to PROOF.md while the campaign is executing.
Usage: python3.12 -B -E -S midrun_real_check.py <copy-root>"""
import json, shutil, subprocess, sys, tempfile, time
from pathlib import Path
POP = 'experiments/universality/two_parameter_cusp_population'
root = Path(sys.argv[1]).resolve()
child = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_population_certificate as runner
t0 = time.monotonic()
rec = runner.run(output_dir=Path(sys.argv[2]), max_evaluations=20000, time_budget=900)
print("child: status=%s exit_code=%s source_drift=%s elapsed=%.2fs" % (rec["status"], rec["exit_code"], rec["source_drift"], time.monotonic() - t0))
'''
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / 'copy'
    shutil.copytree(root, base)
    proof = base / POP / 'PROOF.md'
    p = subprocess.Popen([sys.executable, '-B', '-E', '-S', '-c', child, str(base / POP), str(base / 'attempt')],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)                                     # the campaign is in progress
    assert p.poll() is None, 'child finished before the edit; raise budget'
    proof.write_bytes(proof.read_bytes() + b'\n')
    t_edit = time.monotonic()
    out, _ = p.communicate(timeout=600)
    print(out.strip())
    print('parent: edited PROOF.md at +2.0s; child still running then; child total wait %.2fs after edit' % (time.monotonic() - t_edit))
    disk = json.loads((base / 'attempt/RUN.json').read_text())
    key = POP + '/PROOF.md'
    print('on-disk RUN.json status=%s source_drift=%s' % (disk['status'], disk['source_drift']))
    print('  sources_before[PROOF.md]=%s' % disk['sources_before'][key])
    print('  sources_after [PROOF.md]=%s' % disk['sources_after'][key])
    rows = json.loads((base / 'attempt/CERTIFICATE.json').read_text())['rows']
    print('  CERTIFICATE.json rows statuses=%s' % sorted({r['status'] for r in rows}))
