import sys, json, time
sys.path.insert(0, sys.argv[1])
from moment_kernel import certify, KernelHold
import verify_certificate as V
for lo, hi in [("6/5","6/5"), ("6/5","7/5"), ("5/4","100/1"), ("2/1","164/1")]:
    req = dict(L_lo=lo, L_hi=hi, cutoff=64, bits=128, pi_terms=32, exp_pairs=24, input_role="TEST_ONLY")
    t = time.monotonic()
    try:
        c = certify(req)
    except KernelHold as e:
        print(lo, hi, "HOLD", e.reason, round(time.monotonic()-t, 3)); continue
    tp = time.monotonic()-t
    s = json.dumps(c)
    t = time.monotonic()
    try:
        V.verify(json.loads(s)); r = "ACCEPT"
    except ValueError as e:
        r = "REJECT " + str(e)
    print(lo, hi, "produce", round(tp, 3), "verify", round(time.monotonic()-t, 3), r, "bytes", len(s))
