import subprocess, re, sys, json
from decimal import Decimal as D
# printed 15-digit value, kind, delta = (true endpoint - printed) per 01Fi 419/6243732292 (sign: ceil hi.b above print, floor lo.a below print)
V = {
 "3.66282864761194": ("ceil (0,0.0025]", D("4.8e-15")),
 "3.66435028046302": ("ceil (0.0025,0.005]", D("3.1e-15")),
 "3.67360595202419": ("ceil (0.005,0.01]", D("3.4e-16")),
 "3.74767948915996": ("ceil (0.04,0.05] / uniform", D("1.04e-15")),
 "2.58288222715895": ("floor (0,0.0025]", D("-2.1e-15")),
 "2.58174158877406": ("floor (0.0025,0.005]", D("-2.0e-15")),
 "2.41792545239159": ("floor (0.03,0.04]", D("-9.7e-16")),
}
trees = [(sys.argv[i], sys.argv[i+1]) for i in range(1, len(sys.argv), 2)]  # (gitdir, ref)
out = []
for gitdir, ref in trees:
    for p, (cell, dl) in V.items():
        pre = p[:8]  # 6 decimals
        r = subprocess.run(["git", "-C", gitdir, "grep", "-n", "-F", pre, ref, "--"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            loc, _, txt = line.partition(":")[2].partition(":")
            path = line[len(ref)+1:].rsplit(":", 1)[0] if False else None
            m0 = line[len(ref)+1:]
            fpath, lno, body = m0.split(":", 2)
            for m in re.finditer(re.escape(pre) + r"\d*", body):
                q = m.group(0)
                true = D(p) + dl
                ceil = cell.startswith("ceil")
                ok = (D(q) >= true) if ceil else (D(q) <= true)
                out.append(dict(tree=gitdir.split("/")[-1]+"@"+ref, file=fpath, line=int(lno), cell=cell, quoted=q, exact=(q == p), conservative=ok))
json.dump(out, sys.stdout, indent=0)
