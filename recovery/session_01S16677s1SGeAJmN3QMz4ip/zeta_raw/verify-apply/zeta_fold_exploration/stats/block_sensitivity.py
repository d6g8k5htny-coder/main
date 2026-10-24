"""Block-size sensitivity of the zeta cluster-bootstrap CIs (numerics-review follow-up).

log|zeta| is approximately log-correlated, so blocks of 2000 consecutive lobes need not be
independent at the amplitude level. This recomputes the main statistics with blocks of 500,
2000 and 20000 lobes on the existing samples (no new data):

  s_hill@x            Hill CDF exponent of bars below x * median (exponent statistic)
  flat_coef@x         F(x)/x, F = fraction of bars below x * median: the coefficient of the
                      short-bar CDF if the density were exactly flat (amplitude-sensitive)
  lowamp_frac@0.01    fraction of bars below 0.01 * median whose local amplitude lies in the
                      lowest decile (mark diagnostic; baseline for close gaps is ~4%, see
                      localamp_baseline.py)

Percentile 95% intervals, B = 400 cluster-bootstrap replicates, RNG seed 31337 per block size.
CIs are sampling-only. With 20000-lobe blocks zeta A has only 7 clusters, so its intervals at
that block size are not meaningful and are reported for completeness only.

Reads results/raw/zeta_A.npz and zeta_B.npz; writes results/stats/block_sensitivity.json.
"""
import json

import numpy as np

from analyze import barcode, load, local_sigma
from stats_paths import out_path

B = 400
BLOCKS = (500, 2000, 20000)
CUTS = (0.01, 0.003)


def ci(vals):
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def run(name):
    area, gap, peak, sid, _ = load(name)
    ell, blobe, bm_lobe = barcode(area, sid)
    L = ell / np.median(ell)
    sig = local_sigma(peak, sid)
    q10s = np.quantile(sig, 0.1)
    out = dict(dataset=name, n_lobes=int(area.size), n_bars=int(ell.size), blocks={})
    for bs in BLOCKS:
        rng = np.random.default_rng(31337)
        cid = np.arange(area.size) // bs
        ncl = int(cid.max()) + 1
        clb = cid[np.clip(bm_lobe, 0, area.size - 1)]
        nb = np.bincount(clb, minlength=ncl).astype(float)
        per = {}
        for x in CUTS:
            m = L < x
            n_c = np.bincount(clb[m], minlength=ncl).astype(float)
            sl_c = np.bincount(clb[m], weights=np.log(x / L[m]), minlength=ncl)
            per[x] = (n_c, sl_c)
        sel = (L < 0.01) & (blobe >= 0)
        low = np.zeros(sel.size, bool)
        low[sel] = sig[blobe[sel]] < q10s
        s_c = np.bincount(clb[sel], minlength=ncl).astype(float)
        l_c = np.bincount(clb[low], minlength=ncl).astype(float)

        def stats(w):
            r = {}
            for x in CUTS:
                n_c, sl_c = per[x]
                r[f"s_hill@{x:g}"] = (n_c @ w) / (sl_c @ w)
                r[f"flat_coef@{x:g}"] = (n_c @ w) / (nb @ w) / x
            r["lowamp_frac@0.01"] = (l_c @ w) / (s_c @ w)
            return r

        point = stats(np.ones(ncl))
        reps = [stats(np.bincount(rng.integers(0, ncl, ncl), minlength=ncl).astype(float)) for _ in range(B)]
        out["blocks"][str(bs)] = dict(
            n_clusters=ncl,
            **{k: dict(value=round(float(v), 4), ci95=ci([r[k] for r in reps])) for k, v in point.items()})
    return out


if __name__ == "__main__":
    res = {nm: run(nm) for nm in ("zetaB", "zetaA")}
    print(json.dumps(res, indent=1))
    with open(out_path("block_sensitivity.json"), "w") as f:
        json.dump(res, f, indent=1)
