"""Mutation controls run on 2026-10-13 against GROK-H04's ULW guard (Universal-Law-Workspace#16, head 8e0f5fde).
Run from this directory with CPython 3.11: python3 -I mutations.py. Expected output is recorded in the ULW#16 review 5509491588."""
import importlib.util, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location('r', Path(__file__).with_name('receipt.py')); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
def fixture(files):
    d = Path(tempfile.mkdtemp()); (d/'tests').mkdir()
    for fn, b in files.items(): (d/'tests'/fn).write_text(b)
    return d
empty, failing, passing = fixture(r.FIXTURES['empty']), fixture(r.FIXTURES['one_failing']), fixture(r.FIXTURES['one_passing'])
for wf in ('ci', 'pins'):
    lines = r.CAND[wf].splitlines(keepends=True)
    m1 = ''.join(l for l in lines if not l.startswith('grep '))
    m2 = ''.join(l for l in lines if not l.startswith('set -euo pipefail'))
    shell_full, shell_nopf = ['--noprofile','--norc','-eo','pipefail'], ['--noprofile','--norc','-e']
    print(wf, 'M1 no-grep      empty  ->', r.run(m1, empty, shell_full)['exit'], '(candidate 1, original 0)')
    print(wf, 'M2 no-pipefail  failing->', r.run(m2, failing, shell_nopf)['exit'], '(candidate 1, original 1)')
    print(wf, 'M2 no-pipefail  passing->', r.run(m2, passing, shell_nopf)['exit'], '(candidate 0)')
    print(wf, 'candidate under GitHub shell flags, script without its own set line, failing ->', r.run(m2, failing, shell_full)['exit'])
