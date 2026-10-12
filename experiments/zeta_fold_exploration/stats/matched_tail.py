"""Matched-quantile comparison: Hill CDF exponent of lobe areas a vs of the fold-only proxy r^3
(r = unfolded gap) and of a/sigma_local, at the same lower-tail fraction q.  If the curvature /
amplitude marks were irrelevant, s(a) = s(r^3) = s_gap/3.  Cluster bootstrap 95% CI for the difference.
Resolution: detects mark effects of size |delta s| >~ 0.02 at q >= 0.001; it cannot verify E[kappa^-1] < inf.
Output: results/stats/matched_tail.json"""
import json, numpy as np
from analyze import load, local_sigma
from stats_paths import out_path
rng = np.random.default_rng(7)
out = {}
for nm in ["gauss", "cue128", "cue12", "zetaA", "zetaB"]:
    area, gap, peak, sid, cid = load(nm)
    a = np.abs(area); r3 = (gap / gap.mean()) ** 3; an = a / local_sigma(peak, sid)
    _, cl = np.unique(cid, return_inverse=True); ncl = cl.max() + 1
    res = {}
    for q in (1e-2, 3e-3, 1e-3):
        stats = []
        for x in (a, r3, an):
            th = np.quantile(x, q); m = x < th
            stats.append((np.bincount(cl[m], minlength=ncl).astype(float),
                          np.bincount(cl[m], weights=np.log(th / x[m]), minlength=ncl)))
        S = np.array([[s[0], s[1]] for s in stats])
        est = lambda w: [(S[i, 0] @ w) / (S[i, 1] @ w) for i in range(3)]
        e0 = est(np.ones(ncl))
        bs = np.array([est(np.bincount(rng.integers(0, ncl, ncl), minlength=ncl).astype(float)) for _ in range(400)])
        d = bs[:, 0] - bs[:, 1]
        res[f"q={q:g}"] = dict(n=int(S[0, 0].sum()), s_area=round(e0[0], 4), s_r3=round(e0[1], 4),
                               s_area_over_localamp=round(e0[2], 4),
                               diff_area_minus_r3=round(e0[0] - e0[1], 4),
                               diff_ci95=[round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])],
                               gap_quantile_r=round(float(np.quantile(gap / gap.mean(), q)), 4))
    out[nm] = res
    print(nm, json.dumps(res))
json.dump(out, open(out_path("matched_tail.json"), "w"), indent=1)
