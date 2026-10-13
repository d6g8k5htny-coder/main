"""Apply one single-point mutation to the scratch copy, run the CI command in both modes,
report failing tests, restore the module byte-identically from the read-only worktree."""
import hashlib, re, shutil, subprocess, sys, time
from pathlib import Path
SP = Path('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad')
WT, COPY = SP/'wt350', SP/'p350patch/copy'
POP = 'experiments/universality/two_parameter_cusp_population'
MUTATIONS = {
    'GS3_verify_completeness_trivial': ('global_h0_selector.py',
        "        _check(cnt == supplied, phase+'/complete'+str(i), checks)",
        "        _check(True, phase+'/complete'+str(i), checks)"),
    'RN2_core_overlap_trivial': ('run_global_h0_selector.py',
        "            checks['core_identity_overlap'] &= max(shifted[0], interval[0]) <= min(shifted[1], interval[1])",
        "            checks['core_identity_overlap'] &= True"),
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(mode_flags, log):
    cmd = ['python3.12', '-B', '-E', *mode_flags, '-S', '-m', 'unittest', 'discover', '-s', POP, '-p', 'test_*.py', '-v']
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=COPY, capture_output=True, text=True, timeout=900)
    wall = time.monotonic()-t0
    out = proc.stdout+proc.stderr
    Path(log).write_text(out)
    failed = sorted({f'{cls}.{name}' for name, cls in re.findall(r'^(?:FAIL|ERROR): (\S+) \(([\w.]+)\)', out, re.M)})
    ran = re.search(r'^Ran (\d+) tests', out, re.M)
    summary = re.search(r'^(OK|FAILED.*)$', out, re.M)
    return dict(cmd=' '.join(cmd), rc=proc.returncode, ran=int(ran.group(1)) if ran else None,
                summary=summary.group(1) if summary else out[-300:], failing=failed, wall=round(wall, 2))
for mid in sys.argv[1:]:
    fname, old, new = MUTATIONS[mid]
    target, pristine = COPY/POP/fname, WT/POP/fname
    before = sha(target)
    assert before == sha(pristine), f'{fname}: copy differs from pristine before mutation'
    text = target.read_text()
    assert text.count(old) == 1, (mid, text.count(old))
    target.write_text(text.replace(old, new))
    print(f'=== {mid}: {fname} mutated (sha256 {sha(target)[:16]}..., pristine {before[:16]}...)')
    try:
        for label, flags in (('normal', []), ('-O', ['-O'])):
            r = run(flags, SP/f'p350patch/logs/kill_{mid}_{label.strip("-") or "normal"}.log')
            print(f'  [{label}] {r["cmd"]}')
            print(f'  [{label}] rc={r["rc"]} ran={r["ran"]} summary={r["summary"]} wall={r["wall"]}s')
            print(f'  [{label}] failing tests: {r["failing"]}')
    finally:
        shutil.copyfile(pristine, target)
        for pyc in COPY.rglob('__pycache__'): shutil.rmtree(pyc)
    after = sha(target)
    print(f'  restored: sha256 {after} == pristine {sha(pristine)} -> {"OK" if after == sha(pristine) == before else "MISMATCH"}')
