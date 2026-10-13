# Transient clarification probe (not a repository test): does check_lean_version at main#315 head 690002bb refuse
# FULLY FORMED wrong-version records that keep the exact pinned commit and "Release"? The committed WRONG_RUNTIMES
# fixtures omit the commit field for the wrong-version cases, so they do not isolate the version comparison.
import importlib.util, sys
spec = importlib.util.spec_from_file_location('gate', sys.argv[1]); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
C = gate.LEAN_COMMIT
full = lambda v, plat='x86_64-unknown-linux-gnu', rel='Release': f"Lean (version {v}, {plat}, commit {C}, {rel})\n"
cases = [('4.34.1', full('4.34.1'), 'ACCEPT'), ('4.34.1 aarch64', full('4.34.1', 'aarch64-unknown-linux-gnu'), 'ACCEPT'),
         ('4.34.10', full('4.34.10'), 'REFUSE'), ('4.34.100', full('4.34.100'), 'REFUSE'), ('4.34.1-rc1', full('4.34.1-rc1'), 'REFUSE'),
         ('4.34.1-nightly', full('4.34.1-nightly'), 'REFUSE'), ('4.34.2', full('4.34.2'), 'REFUSE'), ('4.35.1', full('4.35.1'), 'REFUSE'),
         ('4.34.0', full('4.34.0'), 'REFUSE'), ('4.34.1 Debug', full('4.34.1', rel='Debug'), 'REFUSE'), ('04.34.1', full('04.34.1'), 'REFUSE')]
ok = True
for name, text, expect in cases:
    try: gate.check_lean_version(text); got = 'ACCEPT'
    except ValueError as e: got = 'REFUSE'
    flag = 'ok' if got == expect else 'MISMATCH'; ok &= (got == expect)
    print(f'{flag:8} expect={expect:6} got={got:6} version={name!r}')
print('ALL_AS_EXPECTED' if ok else 'SOME_MISMATCH')
