"""Adversarial filesystem probes against check.py finish() using the companion's own Fixture (copy, not the worktree)."""
import hashlib, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path

TREE = Path('<SCRATCH>/lvs/c262/verify_fable_g3/tree')
TEST = TREE / 'companions/torus_filtration_20261011/test_torus_filtration.py'
spec = importlib.util.spec_from_file_location('tft', TEST)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
c = t.checker()
print('outer optimize', sys.flags.optimize)


def run_finish(f, out, timeout=60):
    try:
        r = subprocess.run([*t.child_python(), str(f.side / 'check.py'), 'finish', str(out)], cwd=f.root,
                           capture_output=True, text=True, env=t.child_env(), timeout=timeout)
        return r.returncode, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '', r.stdout[:80]
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', '', ''


def snapshot(out):
    return {p.name: (p.is_symlink(), p.is_dir(), hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and not p.is_symlink() else None)
            for p in out.iterdir()}


with tempfile.TemporaryDirectory() as tmp:
    f = t.Fixture(c, tmp)
    out = f.evidence()
    base = snapshot(out)
    print('fixture evidence members:', len(base), '== EVIDENCE_FILES:', set(base) == set(c.EVIDENCE_FILES))

    # P1 required member is a directory
    saved = (out / 'tests.log').read_bytes(); (out / 'tests.log').unlink(); (out / 'tests.log').mkdir()
    print('P1 dir member      ->', run_finish(f, out)[:2], '| receipt absent:', not (out / 'receipt.json').exists())
    (out / 'tests.log').rmdir(); (out / 'tests.log').write_bytes(saved)

    # P2 required member is a FIFO (must refuse without blocking)
    (out / 'tests.log').unlink(); os.mkfifo(out / 'tests.log')
    print('P2 fifo member     ->', run_finish(f, out, timeout=30)[:2], '| receipt absent:', not (out / 'receipt.json').exists())
    (out / 'tests.log').unlink(); (out / 'tests.log').write_bytes(saved)

    # P3 receipt.json is a FIFO
    os.mkfifo(out / 'receipt.json')
    print('P3 fifo receipt    ->', run_finish(f, out, timeout=30)[:2], '| still fifo:', (out / 'receipt.json').is_fifo())
    (out / 'receipt.json').unlink()

    # P4 receipt.json is a directory
    (out / 'receipt.json').mkdir()
    print('P4 dir receipt     ->', run_finish(f, out)[:2], '| still dir:', (out / 'receipt.json').is_dir())
    (out / 'receipt.json').rmdir()

    # P5 receipt.json is a live symlink to a required member: refuse, and the target bytes must not change
    os.symlink('tests.log', out / 'receipt.json')
    before = (out / 'tests.log').read_bytes()
    print('P5 live-symlink rcpt->', run_finish(f, out)[:2], '| target unchanged:', (out / 'tests.log').read_bytes() == before,
          '| still symlink:', (out / 'receipt.json').is_symlink())
    (out / 'receipt.json').unlink()

    # P6 OUT itself is a symlink to the evidence dir
    link = Path(tmp) / 'evidence-link'; os.symlink(out, link)
    print('P6 symlinked OUT   ->', run_finish(f, link)[:2], '| receipt absent:', not (out / 'receipt.json').exists())
    link.unlink()

    # P7 hidden file
    (out / '.keep').write_text('')
    print('P7 hidden file     ->', run_finish(f, out)[:2])
    (out / '.keep').unlink()

    # P8 hardlinked member (a regular file by inode; observe only)
    (out / 'tests.log').unlink(); os.link(out / 'tests-optimized.log', out / 'tests.log')
    r8 = run_finish(f, out)
    print('P8 hardlink member ->', r8[:2], '| receipt written:', (out / 'receipt.json').exists())
    if (out / 'receipt.json').exists():
        (out / 'receipt.json').unlink()
    (out / 'tests.log').unlink(); (out / 'tests.log').write_bytes(saved)

    # P9 relative OUT path (as replay.sh would not do, but the CLI allows)
    rel = os.path.relpath(out, f.root)
    r9 = run_finish(f, Path(rel))
    print('P9 relative OUT    ->', r9[:2], '| receipt written:', (out / 'receipt.json').exists())
    receipt = json.loads((out / 'receipt.json').read_text())
    print('   receipt log_sha256 n =', len(receipt['log_sha256']), '| has receipt.json:', 'receipt.json' in receipt['log_sha256'],
          '| keys == EVIDENCE_FILES:', set(receipt['log_sha256']) == set(c.EVIDENCE_FILES))
    frozen = (out / 'receipt.json').read_bytes()

    # P10 second finish, then newly failing input, then membership drift with a receipt present: all refuse at the receipt gate
    print('P10 second finish  ->', run_finish(f, out)[:2], '| bytes frozen:', (out / 'receipt.json').read_bytes() == frozen)
    (out / 'axioms.status').write_text('1\n')
    print('P10 + failing input->', run_finish(f, out)[:2], '| bytes frozen:', (out / 'receipt.json').read_bytes() == frozen)
    (out / 'tests.log').unlink()
    print('P10 + missing log  ->', run_finish(f, out)[:2], '| bytes frozen:', (out / 'receipt.json').read_bytes() == frozen)
    (out / 'tests.log').write_bytes(saved); (out / 'axioms.status').write_text('0\n')
    after = snapshot(out); after.pop('receipt.json')
    print('P10 evidence bytes unchanged across all refusals:', after == base)
