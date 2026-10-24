"""Stdlib-only driver (run with python3 -I). Produces certificates with the delivered
producer copy and verifies them with the delivered checker copy (separate module).
argv: workdir requests.json outdir"""
import sys, json, os, time
work, reqfile, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, work)
import moment_kernel as mk
import verify_certificate as vc
os.makedirs(outdir, exist_ok=True)
reqs = json.load(open(reqfile))
summary = []
for i, (tag, req) in enumerate(reqs):
    t = time.monotonic()
    try:
        c = mk.certify(req)
    except mk.KernelHold as e:
        summary.append({"tag": tag, "req": req, "status": "HOLD", "reason": e.reason, "t": time.monotonic()-t})
        continue
    dt = time.monotonic()-t
    try:
        vc.verify(json.loads(json.dumps(c)))
        v = "ACCEPT"
    except ValueError as e:
        v = "REJECT:" + str(e)
    path = os.path.join(outdir, "%03d.json" % i)
    with open(path, "w") as f:
        json.dump(c, f, sort_keys=True)
    summary.append({"tag": tag, "req": req, "status": c["status"], "outcome": c["outcome"], "verify": v, "path": path, "t": dt})
json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
for s in summary:
    print(s["tag"], s["status"], s.get("reason", s.get("outcome")), s.get("verify", ""), "%.3fs" % s["t"])
