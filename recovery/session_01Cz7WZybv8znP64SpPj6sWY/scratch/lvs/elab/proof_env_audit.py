"""Local replica of the proof packet's environment audit (check.py --execute minus leanchecker/controls)."""
import os, sys
from pathlib import Path
PACKET = Path('<SCRATCH>/repos/math/specifications/lean_statement_proofs_20261011')
sys.path.insert(0, str(PACKET))
import check
root = PACKET.resolve(); stmt = (root / check.STATEMENT_DIR).resolve()
registered, structures, _ = check.load_register(stmt)
module_files = check.check_tree(root)
modules = check.check_root(root, module_files)
scans = {m: check.scan_module((root / m).read_text(encoding='utf-8'), Path(m).stem, registered) for m in modules}
module_names = [check.ROOT_MODULE + '.' + Path(m).stem for m in modules]
expected = {t['name']: t['proves'] for m in modules for t in scans[m]['theorems']}
out = root / '.lake' / 'evidence_local'; out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
res = check.environment_audit(os.path.expanduser('~/.elan/bin/lake'), root, env, out, module_names, sorted(registered), expected)
print('ENVIRONMENT_AUDIT_PASS', res, 'theorems expected', len(expected))
