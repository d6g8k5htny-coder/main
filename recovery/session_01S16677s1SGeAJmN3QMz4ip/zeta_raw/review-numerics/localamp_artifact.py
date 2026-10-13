"""Is 'only 3.8% of zeta short bars lie in the lowest local-amplitude decile' a property of zeta,
or of local_sigma (median peak of the 12 neighbouring lobes, excluding lobe k)?  Level rigidity
makes a short gap's neighbours larger, which raises local_sigma regardless of bar length.
Test: condition only on the gap (no bar lengths) and on a neighbour-free amplitude proxy."""
import sys, numpy as np
sys.dont_write_bytecode = True
ST = '/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats'
sys.path.insert(0, ST)
from analyze import load, local_sigma
for nm in ['gauss', 'cue12', 'cue128', 'zetaB']:
    area, gap, peak, sid, cid = load(nm)
    r = gap / gap.mean()
    sig = local_sigma(peak, sid)
    q10s = np.quantile(sig, 0.1)
    # far-neighbour proxy: median peak of lobes k-12..k-7 and k+7..k+12 (skips the 6 nearest each side)
    w = 12; n = peak.size
    pad = np.full(n + 2 * w, np.nan); pad[w:w + n] = peak
    sp = np.full(n + 2 * w, -1); sp[w:w + n] = sid
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).copy()
    ws = np.lib.stride_tricks.sliding_window_view(sp, 2 * w + 1)
    win = np.where(ws == sid[:, None], win, np.nan)
    win[:, 6:19] = np.nan
    far = np.nanmedian(win, axis=1)
    q10f = np.nanquantile(far, 0.1)
    out = []
    for thr in (0.1, 0.2):
        m = r < thr
        out.append('gap<%.1f: n=%d  P(local_sigma in low decile)=%.3f  P(far proxy in low decile)=%.3f' %
                   (thr, m.sum(), np.mean(sig[m] < q10s), np.nanmean(far[m] < q10f)))
    print(nm, ' | '.join(out))
