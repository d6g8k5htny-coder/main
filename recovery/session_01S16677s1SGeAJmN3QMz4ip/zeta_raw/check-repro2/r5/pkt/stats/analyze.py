"""H0 superlevel barcodes of antiderivatives F = int g and short-bar / small-gap exponents.

For a 1D integrand g with simple zeros z_0<...<z_K, the critical points of F are the z_k and
F(z_{k+1}) - F(z_k) = signed lobe integral. Superlevel H0 persistence of F on [z_0, z_K] is the
persistence of the path graph with vertex values F(z_k) (F is monotone between them). Elder rule
via union-find. Bars whose birth vertex is a path endpoint are dropped (boundary artifacts);
the essential bar is dropped.

Exponent conventions: density ~ x^alpha near 0, CDF ~ x^s, s = alpha + 1.
  Hill/MLE on x < x_hi: s_hat = n / sum log(x_hi/x_i)  (exact MLE for a pure power law on (0,x_hi]).
  LS: slope of log ECDF vs log x on log-spaced points in [lo, hi].
Cluster bootstrap (resampling whole samples / zeta blocks), percentile 95% CI, B = 400.
The CIs cover sampling error only, not the finite-cutoff estimator bias (see README).
Zeta blocks are 2000 lobes; block_sensitivity.py checks 500 / 2000 / 20000.

Usage: python3 stats/analyze.py [names...] > results/stats/analyze.log
Reads results/raw/*.npz; writes results/stats/summary.json, ecdf_*.csv and results/raw/ecdf_samples.npz.
"""
import os, sys, json, time
import numpy as np

from stats_paths import out_path, raw_path
RNG = np.random.default_rng(424242)
NBOOT = 400


# ---------------------------------------------------------------------------
def barcode(area, sid):
    """Return arrays (birth_vertex_lobe_index info) for finite bars of all paths.
    Outputs: ell, adj_lobe (lobe index if birth/death vertices adjacent else -1), lobe_left_of_max."""
    starts = np.r_[0, np.nonzero(np.diff(sid))[0] + 1]
    ends = np.r_[starts[1:], area.size]
    nv = area.size + 2 * starts.size  # K+1 vertices per path + 1 separator per path
    vals = np.empty(nv); vlobe = np.empty(nv, dtype=np.int64); endpoint = np.zeros(nv, bool)
    pos = 0
    for s, e in zip(starts, ends):
        K = e - s
        vals[pos] = 0.0
        vals[pos + 1: pos + K + 1] = np.cumsum(area[s:e])
        vlobe[pos: pos + K + 1] = np.arange(s, e + 1)  # vertex j sits left of lobe s+j
        endpoint[pos] = True; endpoint[pos + K] = True
        vals[pos + K + 1] = -np.inf; vlobe[pos + K + 1] = -1
        pos += K + 2
    order = np.argsort(-vals, kind="stable")
    parent = np.full(nv, -1, dtype=np.int64)
    birth = np.full(nv, -1, dtype=np.int64)
    bm, bd = [], []
    par = parent  # local alias
    for v in order.tolist():
        if vals[v] == -np.inf:
            break
        roots = []
        for u in (v - 1, v + 1):
            if 0 <= u < nv and par[u] >= 0:
                r = u
                while par[r] != r:
                    par[r] = par[par[r]]
                    r = par[r]
                roots.append(r)
        if not roots:
            par[v] = v; birth[v] = v
        elif len(roots) == 1 or roots[0] == roots[1]:
            par[v] = roots[0]
        else:
            r1, r2 = roots
            b1, b2 = birth[r1], birth[r2]
            if vals[b1] >= vals[b2]:
                old, young = r1, r2
            else:
                old, young = r2, r1
            bm.append(birth[young]); bd.append(v)
            par[young] = old; par[v] = old
    bm = np.array(bm, dtype=np.int64); bd = np.array(bd, dtype=np.int64)
    keep = ~endpoint[bm]
    bm, bd = bm[keep], bd[keep]
    ell = vals[bm] - vals[bd]
    adj = np.abs(bm - bd) == 1
    lobe = np.where(adj, np.minimum(vlobe[bm], vlobe[bd]), -1)
    return ell, lobe, vlobe[bm]


def local_sigma(peak, sid, w=6):
    """median of peaks of lobes k-w..k+w excluding k, within the same path."""
    n = peak.size
    pad = np.full(n + 2 * w, np.nan)
    pad[w:w + n] = peak
    sidp = np.full(n + 2 * w, -1); sidp[w:w + n] = sid
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    wsid = np.lib.stride_tricks.sliding_window_view(sidp, 2 * w + 1)
    win = np.where(wsid == sid[:, None], win, np.nan)
    win = win.copy(); win[:, w] = np.nan
    return np.nanmedian(win, axis=1)


# ---------------------------------------------------------------------------
def boot_weights(ncl, B=NBOOT):
    for _ in range(B):
        yield np.bincount(RNG.integers(0, ncl, ncl), minlength=ncl).astype(float)


def hill_boot(x, cl, ncl, xhis):
    """x normalized; returns dict xhi -> (s_hat, lo, hi, n)."""
    out = {}
    stats = []
    for xh in xhis:
        m = x < xh
        n_c = np.bincount(cl[m], minlength=ncl).astype(float)
        sl_c = np.bincount(cl[m], weights=np.log(xh / x[m]), minlength=ncl)
        stats.append((n_c, sl_c))
    S = np.array([[st[0], st[1]] for st in stats])  # (nx, 2, ncl)
    ests = {xh: [] for xh in xhis}
    for wgt in boot_weights(ncl):
        v = S @ wgt  # (nx, 2)
        for i, xh in enumerate(xhis):
            ests[xh].append(v[i, 0] / v[i, 1] if v[i, 1] > 0 else np.nan)
    for i, xh in enumerate(xhis):
        n = S[i, 0].sum(); s = n / S[i, 1].sum() if S[i, 1].sum() > 0 else np.nan
        lo, hi = np.nanpercentile(ests[xh], [2.5, 97.5])
        out[f"{xh:g}"] = dict(s=round(float(s), 4), ci95=[round(float(lo), 4), round(float(hi), 4)], n=int(n))
    return out


def ls_boot(x, cl, ncl, ranges, npts=13):
    out = {}
    tot_c = np.bincount(cl, minlength=ncl).astype(float)
    for lo, hi in ranges:
        g = np.logspace(np.log10(lo), np.log10(hi), npts)
        idx = np.searchsorted(g, x, side="left")  # x <= g[j] iff idx <= j
        C = np.zeros((npts, ncl))
        for j in range(npts):
            m = idx <= j
            C[j] = np.bincount(cl[m], minlength=ncl)
        lg = np.log(g)

        def slope(cnt, tot):
            if np.any(cnt <= 0):
                return np.nan
            y = np.log(cnt / tot)
            return np.polyfit(lg, y, 1)[0]
        s0 = slope(C.sum(1), tot_c.sum())
        bs = [slope(C @ w, tot_c @ w) for w in boot_weights(ncl)]
        l, h = np.nanpercentile(bs, [2.5, 97.5])
        out[f"{lo:g}-{hi:g}"] = dict(s=round(float(s0), 4), ci95=[round(float(l), 4), round(float(h), 4)],
                                     n_in_range=int(np.sum((x >= lo) & (x <= hi))), n_below_lo=int(np.sum(x < lo)))
    return out


def local_slopes(x, edges):
    """log-log slope of ECDF between successive edges."""
    n = x.size
    F = np.array([np.sum(x <= e) for e in edges]) / n
    rows = []
    for i in range(len(edges) - 1):
        if F[i] > 0:
            rows.append((edges[i], edges[i + 1], float(np.log(F[i + 1] / F[i]) / np.log(edges[i + 1] / edges[i])), int(F[i] * n)))
    return rows


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
def analyze(name, area, gap, peak, sid, cid):
    t0 = time.time()
    res = dict(dataset=name, n_lobes=int(area.size))
    a = np.abs(area)
    ell, blobe, bm_lobe = barcode(area, sid)
    res["n_bars"] = int(ell.size)
    res["frac_bars_adjacent_pair"] = round(float(np.mean(blobe >= 0)), 4)
    med_ell = np.median(ell); med_a = np.median(a)
    L = ell / med_ell
    A = a / med_a
    r = gap / gap.mean()
    sig = local_sigma(peak, sid)
    hrel = peak / sig
    c = a / (gap * peak)
    # clusters for bootstrap
    ucl, cl_l = np.unique(cid, return_inverse=True)
    ncl = ucl.size
    cl_b = cl_l[np.clip(bm_lobe, 0, cid.size - 1)]
    res["n_clusters"] = int(ncl)
    xh_bars = [0.1, 0.03, 0.01, 0.003]
    res["bars_hill_s"] = hill_boot(L, cl_b, ncl, xh_bars)
    res["bars_ls_s"] = ls_boot(L, cl_b, ncl, [(0.01, 0.1), (0.001, 0.01), (0.001, 0.1)])
    res["lobes_hill_s"] = hill_boot(A, cl_l, ncl, xh_bars)
    res["gaps_hill_s"] = hill_boot(r, cl_l, ncl, [0.4, 0.3, 0.2, 0.15])
    # amplitude-normalized lobe areas a/sigma
    An = a / sig; An = An / np.median(An)
    res["lobes_over_localamp_hill_s"] = hill_boot(An, cl_l, ncl, [0.03, 0.01, 0.003])
    # fold geometry check among small gaps
    m = r < 0.35
    if m.sum() > 50:
        res["fold_check_small_gaps"] = dict(
            n=int(m.sum()),
            slope_log_area_vs_log_gap=round(float(np.polyfit(np.log(r[m]), np.log(a[m]), 1)[0]), 3),
            slope_log_hrel_vs_log_gap=round(float(np.polyfit(np.log(r[m]), np.log(hrel[m]), 1)[0]), 3),
            slope_log_area_over_sigma_vs_log_gap=round(float(np.polyfit(np.log(r[m]), np.log(a[m] / sig[m]), 1)[0]), 3))
    # correlations
    res["spearman_all_lobes"] = dict(log_area_vs_log_gap=round(spearman(a, r), 3),
                                     log_area_vs_log_localamp=round(spearman(a, sig), 3),
                                     log_gap_vs_log_localamp=round(spearman(r, sig), 3))
    # short-bar decomposition
    q10r = np.quantile(r, 0.1); q10s = np.quantile(sig, 0.1)
    med = dict(r=np.median(r), h=np.median(hrel), s=np.median(sig), c=np.median(c))
    dec = {}
    for thr in (0.1, 0.03, 0.01):
        sel = (L < thr) & (blobe >= 0)
        k = blobe[sel]
        if k.size < 5:
            continue
        d_r = np.log10(r[k] / med["r"]); d_h = np.log10(hrel[k] / med["h"])
        d_s = np.log10(sig[k] / med["s"]); d_c = np.log10(c[k] / med["c"])
        dec[f"bars<{thr:g}*median"] = dict(
            n=int(sel.sum()), frac_adjacent=round(float(sel.sum() / max(1, np.sum(L < thr))), 4),
            median_gap=round(float(np.median(r[k])), 3),
            frac_gap_below_q10=round(float(np.mean(r[k] < q10r)), 3),
            frac_localamp_below_q10=round(float(np.mean(sig[k] < q10s)), 3),
            mean_log10_dev=dict(gap=round(float(d_r.mean()), 3), peak_over_localamp=round(float(d_h.mean()), 3),
                                localamp=round(float(d_s.mean()), 3), shape=round(float(d_c.mean()), 3),
                                total=round(float((d_r + d_h + d_s + d_c).mean()), 3)),
            spearman_ell_vs_gap=round(spearman(L[sel], r[k]), 3) if k.size > 20 else None,
            spearman_ell_vs_localamp=round(spearman(L[sel], sig[k]), 3) if k.size > 20 else None)
    res["short_bar_decomposition"] = dec
    res["local_slopes_bars"] = local_slopes(L, np.logspace(-4, 0, 9))
    res["local_slopes_gaps"] = local_slopes(r, [0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0])
    res["seconds"] = round(time.time() - t0, 1)
    ecdf = dict(L=L, A=A, r=r, An=An)
    return res, ecdf


def load(name):
    if name.startswith("zeta"):
        w = name[4]
        d = np.load(raw_path(f"zeta_{w}.npz"))
        area = d["lobe_raw"] if name.endswith("raw") else d["lobe_unf"]
        u = d["unfold"]
        gap = np.diff(u) if not name.endswith("raw") else np.diff(d["zeros"])
        peak = d["peak"]
        n = area.size
        sid = np.zeros(n, dtype=np.int64)
        cid = np.arange(n) // 2000
        return area, gap, peak, sid, cid
    d = np.load(raw_path(f"{name}.npz"))
    return d["area"], d["gap"], d["peak"], d["sid"], d["cid"]


def write_ecdf_csv(allecdf):
    grid = np.logspace(-4, 1, 51)
    with open(out_path("ecdf_bars_lobes.csv"), "w") as f:
        cols = []
        for nm, e in allecdf.items():
            cols += [f"{nm}_bars", f"{nm}_lobes"]
        f.write("x_over_median," + ",".join(cols) + "\n")
        sorted_ = {nm: (np.sort(e["L"]), np.sort(e["A"])) for nm, e in allecdf.items()}
        for x in grid:
            row = [f"{x:.6g}"]
            for nm in allecdf:
                Ls, As = sorted_[nm]
                row += [f"{np.searchsorted(Ls, x, 'right') / Ls.size:.6g}", f"{np.searchsorted(As, x, 'right') / As.size:.6g}"]
            f.write(",".join(row) + "\n")
    grid = np.logspace(-2, 0.5, 41)
    with open(out_path("ecdf_gaps.csv"), "w") as f:
        f.write("gap_over_mean," + ",".join(allecdf) + "\n")
        sr = {nm: np.sort(e["r"]) for nm, e in allecdf.items()}
        for x in grid:
            f.write(f"{x:.6g}," + ",".join(f"{np.searchsorted(sr[nm], x, 'right') / sr[nm].size:.6g}" for nm in allecdf) + "\n")


if __name__ == "__main__":
    names = sys.argv[1:] or ["gauss", "cue128", "cue12", "zetaA", "zetaB", "zetaB_raw"]
    out = {}; ecdfs = {}
    for nm in names:
        try:
            data = load(nm)
        except FileNotFoundError as e:
            print("skip", nm, e); continue
        r, e = analyze(nm, *data)
        out[nm] = r; ecdfs[nm] = e
        print(json.dumps(r, indent=1), flush=True)
    with open(out_path("summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_ecdf_csv(ecdfs)
    np.savez_compressed(raw_path("ecdf_samples.npz"), **{f"{k}_{kk}": v for k, e in ecdfs.items() for kk, v in e.items() if kk in ("L", "r")})
