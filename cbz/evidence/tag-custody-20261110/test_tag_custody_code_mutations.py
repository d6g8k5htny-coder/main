"""Run real verifier code mutants against the retained corrupt-source cases."""
import ast
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'tag-custody-code-mutations'
OUT.mkdir(exist_ok=False)
source = ROOT / 'tag_custody_verifier_common.py'
records = []
for mutation, target in (('disable-require', 'require'), ('disable-source-check', 'check_source')):
    directory = OUT / mutation
    directory.mkdir()
    for name in ('verify_math_tag_native_custody.py', 'verify_main_tag_native_custody.py'):
        shutil.copy2(ROOT / name, directory / name)
    tree = ast.parse(source.read_text())
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == target]
    if len(found) != 1:
        raise ValueError('mutation target must exist exactly once')
    found[0].body = [ast.Pass()]
    ast.fix_missing_locations(tree)
    mutated = directory / source.name
    mutated.write_text(ast.unparse(tree) + '\n')
    for kind in ('math', 'main'):
        candidate = ROOT / 'tag-custody-check-after-repair' / (kind + '-corrupt-source.tar.gz')
        verifier = directory / ('verify_' + kind + '_tag_native_custody.py')
        for mode, flags in (('normal', []), ('optimized', ['-O'])):
            p = subprocess.run([sys.executable, '-B', *flags, '-S', str(verifier), str(candidate)],
                               capture_output=True, timeout=30)
            stem = mutation + '-' + kind + '-' + mode
            (OUT / (stem + '.stdout')).write_bytes(p.stdout)
            (OUT / (stem + '.stderr')).write_bytes(p.stderr)
            # The adverse oracle rejects this mutant only when the mutated
            # verifier falsely accepts the actual corrupted source input.
            records.append({'mutation': mutation, 'kind': kind, 'mode': mode,
                            'mutated_child_exit': p.returncode,
                            'adverse_oracle_killed': p.returncode == 0,
                            'mutated_source_sha256': hashlib.sha256(mutated.read_bytes()).hexdigest(),
                            'archive_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()})
report = {'scope': 'two specified helper-function mutations, corrupted source only; not exhaustive',
          'cases': records, 'kills': sum(row['adverse_oracle_killed'] for row in records)}
(OUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'children': len(records), 'kills': report['kills'], 'scope': report['scope']}))
if report['kills'] != len(records):
    raise SystemExit(1)
