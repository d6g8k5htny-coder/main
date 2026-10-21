import subprocess, re, sys, json
from decimal import Decimal as D
# O7 (419/6245249123) indicative endpoints after matmax/vecmax fix; cell 1 uncertified.
T = [
 ("3.7476", "ceil (0.04,0.05] / uniform", D("3.74772007069189177")),
 ("3.7477", "ceil (0.04,0.05] / uniform", D("3.74772007069189177")),
 ("2.3065", "floor (0.04,0.05] / uniform", D("2.30654071417810586")),
 ("2.3066", "floor (0.04,0.05] / uniform", D("2.30654071417810586")),
 ("3.7241", "ceil (0.03,0.04]", D("3.72414748332433") + D("3.77e-11")),
 ("2.4179", "floor (0.03,0.04]", D("2.41792545239159") - D("9.7e-16") - D("4.31e-11")),
 ("3.6628", "ceil cell1 (0,0.0025]", None),
 ("2.5828", "floor cell1 (0,0.0025]", None),
 ("2.5829", "floor cell1 (0,0.0025]", None),
]
trees = [(sys.argv[i], sys.argv[i+1]) for i in range(1, len(sys.argv), 2)]
out = []
for gitdir, ref in trees:
    for pre, cell, end in T:
        r = subprocess.run(["git", "-C", gitdir, "grep", "-n", "-F", pre, ref, "--"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            fpath, lno, body = line[len(ref)+1:].split(":", 2)
            for m in re.finditer(r"(?<![\d.])" + re.escape(pre) + r"\d*(?![\d])", body):
                q = m.group(0)
                if end is None: verdict = "UNCERTIFIED_CELL1"
                elif cell.startswith("ceil"): verdict = "ok" if D(q) >= end else "NC"
                else: verdict = "ok" if D(q) <= end else "NC"
                out.append(dict(tree=gitdir.rstrip("/").split("/")[-1] + "@" + ref, file=fpath, line=int(lno), cell=cell, quoted=q, verdict=verdict))
json.dump(out, sys.stdout, indent=0)
