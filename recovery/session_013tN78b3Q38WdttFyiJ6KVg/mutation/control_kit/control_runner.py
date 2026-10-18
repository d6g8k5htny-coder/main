"""Claude 54's mutant control for a test-only addition (standard library only).
Usage, from the repository root: python3 control_runner.py control_mutants.json FILE -- TEST COMMAND...
For each listed mutant of FILE: apply it (its old text must occur exactly once), run the test command, restore.
Report KILLED (command fails) or SURVIVED. Run once without the new test class (expect SURVIVED) and once with it (expect KILLED)."""
import json, subprocess, sys
spec, target = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index('--') + 1:]
rows = next(o['mutants'] for o in json.load(open(spec)) if o['file'] == target)
original = open(target, encoding='utf-8').read()
try:
    for mid, old, new in rows:
        if original.count(old) != 1: print(mid, 'PATTERN-NOT-UNIQUE'); continue
        open(target, 'w', encoding='utf-8').write(original.replace(old, new))
        code = subprocess.run(cmd, capture_output=True).returncode
        print(mid, 'KILLED' if code else 'SURVIVED', flush=True)
finally:
    open(target, 'w', encoding='utf-8').write(original)
