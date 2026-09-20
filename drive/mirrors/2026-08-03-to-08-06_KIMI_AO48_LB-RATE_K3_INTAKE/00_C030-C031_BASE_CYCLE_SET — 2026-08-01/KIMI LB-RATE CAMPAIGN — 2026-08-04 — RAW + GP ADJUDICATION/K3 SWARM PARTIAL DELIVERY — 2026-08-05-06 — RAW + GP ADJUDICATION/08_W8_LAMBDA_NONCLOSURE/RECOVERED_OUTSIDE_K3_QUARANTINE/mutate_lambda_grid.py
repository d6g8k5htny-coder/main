"""Mutation-test driver for verify_lambda_grid_v2.py (K3 Phase 3 requirement).
Each mutation must trip the fail-closed ck() (SystemExit, nonzero status).
Receipts printed to stdout; deterministic."""
import subprocess, sys, os, tempfile, hashlib, json

HERE = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(HERE, 'verify_lambda_grid_v2.py')
SRC = open(CERT).read()

def run_case(name, path, env_extra=None, timeout=600):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = p.stdout + p.stderr
        tripped = (p.returncode != 0) and ('FAIL-CLOSED TRIGGER' in out)
    except subprocess.TimeoutExpired:
        tripped = False
        out = 'TIMEOUT'
    print('mutation %s: fail-closed tripped = %s' % (name, tripped))
    if not tripped:
        print('MUTATION TEST FAILED (ck did not fire) for case ' + name)
        raise SystemExit(1)
    return True

def main():
    print('mutation tests for verify_lambda_grid_v2.py (receipts)')
    print('certificate sha256: ' + hashlib.sha256(open(CERT, 'rb').read()).hexdigest())
    tmp = tempfile.mkdtemp(prefix='lamgrid_mut_')

    # A: corrupted archived input (hash mismatch must trip input integrity)
    arch = open('/mnt/agents/upload/c027 sweep core40.json').read()
    bad = arch.replace('0.011820071962656182', '0.012820071962656182', 1)
    ck_bad = bad != arch
    badpath = os.path.join(tmp, 'corrupted_sweep.json')
    open(badpath, 'w').write(bad)
    print('mutation A setup: archived input corrupted =', ck_bad)
    run_case('A (corrupted input hash)', CERT, env_extra={'SWEEP_JSON_PATH': badpath})

    # B: script copy with embedded expected hash adjusted to the corrupted file
    #    -> cross-validation must trip instead
    hb = hashlib.sha256(open(badpath, 'rb').read()).hexdigest()
    srcB = SRC.replace('0003756d4075bbfa881edfeac68d6cca7242c54cee2c551f2814fd57759b4c18', hb)
    ckB = srcB != SRC
    pathB = os.path.join(tmp, 'cert_B.py')
    open(pathB, 'w').write(srcB)
    print('mutation B setup: embedded hash swapped =', ckB)
    run_case('B (corrupted archived value, hash bypass)', pathB,
             env_extra={'SWEEP_JSON_PATH': badpath})

    # C: script copy with corrupted DEN closed form (den_check must trip)
    srcC = SRC.replace("mp.sqrt(2) * b * mp.exp(-b * b / 4) / mp.sqrt(2 * mp.pi)",
                       "mp.sqrt(2) * b * mp.exp(-b * b / 3) / mp.sqrt(2 * mp.pi)")
    ckC = srcC != SRC
    pathC = os.path.join(tmp, 'cert_C.py')
    open(pathC, 'w').write(srcC)
    print('mutation C setup: DEN closed form corrupted =', ckC)
    run_case('C (corrupted DEN closed form)', pathC)

    # D: script copy with corrupted tail threshold (P0 tail check must trip)
    srcD = SRC.replace("ck(tail < mpf('1e-60'), 'P0 kernel tail exceeds 1e-60')",
                       "ck(tail < mpf('1e-200'), 'P0 kernel tail exceeds 1e-60')")
    ckD = srcD != SRC
    pathD = os.path.join(tmp, 'cert_D.py')
    open(pathD, 'w').write(srcD)
    print('mutation D setup: tail threshold corrupted =', ckD)
    run_case('D (corrupted kernel tail threshold)', pathD)

    print('ALL MUTATION TESTS PASSED (every mutation tripped ck)')

main()
