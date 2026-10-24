"""Produce point certificates at L in {1/2,1,2,3} under default and max settings; scan cutoff M."""
import sys, json, os
from fractions import Fraction as Q
KDIR = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, KDIR)
from moment_kernel import certify, KernelHold
from verify_certificate import verify
Ls = ["1/2", "1/1", "2/1", "3/1"]
settings = {"default": dict(bits=80, pi_terms=20, exp_pairs=16),
            "max": dict(bits=128, pi_terms=32, exp_pairs=24)}
best = {}
for L in Ls:
    for sname, s in settings.items():
        rows = []
        for M in [1,2,3,4,5,6,8,10,12,16,20,24,32,48,64]:
            req = dict(L_lo=L, L_hi=L, cutoff=M, input_role="TEST_ONLY", **s)
            try:
                c = certify(req)
            except KernelHold as e:
                rows.append((M, "HOLD", e.reason)); continue
            verify(json.loads(json.dumps(c)))
            w = {k: Q(v[1]) - Q(v[0]) for k, v in c["bounds"].items()}
            rows.append((M, "OK", float(w["mu2"]), float(w["mu6"])))
            key = (L, sname)
            score = sum(w[k] / max(Q(c["bounds"][k][1]), Q(1, 10**300)) for k in ("mu2","mu4","mu6"))
            if key not in best or score < best[key][0]:
                best[key] = (score, M, c)
        print(L, sname, rows)
for (L, sname), (score, M, c) in best.items():
    fn = os.path.join(OUT, "cert_L%s_%s.json" % (L.replace("/", "_"), sname))
    with open(fn, "w") as f: json.dump(c, f)
    print("BEST", L, sname, "M=", M, fn)
