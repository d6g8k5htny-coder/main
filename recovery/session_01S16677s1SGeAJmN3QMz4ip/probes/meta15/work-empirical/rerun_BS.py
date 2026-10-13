"""Second CLI pass with the test-suite / original-diagnostic flags (-B -S, -B -O -S); compares to plain runs."""
import json, subprocess
from pathlib import Path
W = Path(__file__).resolve().parent
SCRIPT = W / 'repo/tools/architecture_conformance.py'
diffs = []; n = 0
for cdir in sorted((W / 'cases').iterdir()):
    ws = cdir / 'ws'
    for tag, flags, ref in (('cli_BS', ['-B', '-S'], 'cli_normal'), ('cli_BOS', ['-B', '-O', '-S'], 'cli_O')):
        p = subprocess.run(['/usr/bin/python3', *flags, str(SCRIPT), '--workspace', str(ws)], capture_output=True, text=True, timeout=120)
        (cdir / f'{tag}.rc').write_text(f'{p.returncode}\n'); (cdir / f'{tag}.stdout').write_text(p.stdout); (cdir / f'{tag}.stderr').write_text(p.stderr)
        n += 1
        ref_rc = int((cdir / f'{ref}.rc').read_text()); ref_out = (cdir / f'{ref}.stdout').read_text()
        ref_err_tail = ((cdir / f'{ref}.stderr').read_text().strip().splitlines() or [''])[-1]
        err_tail = (p.stderr.strip().splitlines() or [''])[-1]
        if p.returncode != ref_rc or p.stdout != ref_out or err_tail != ref_err_tail:
            diffs.append((cdir.name, tag, p.returncode, ref_rc))
print('processes', n, 'differences vs plain runs', len(diffs)); print(diffs[:20])
(W / 'raw/rerun_BS_summary.json').write_text(json.dumps({'processes': n, 'differences': diffs}, indent=1))
