"""Containment of #447's certified (mu, Sigma) enclosures, with a direct law accurate enough to
resolve the tight cell-endpoint slack.

Usage: python3 -I law_probe2.py PROBE_PROGRAM H2_DIR OUT.jsonl
Phase A (the certificate's own dps 100): store every cell enclosure (first_cell_law,
neumann_law(c1, c2)) and point enclosure (neumann_law(r, r), or on the first cell the order-J
Taylor polynomial at r plus the certified tails MT, ST).
Phase B: rebuild the H2 lattice with ce.configure at (dps, kmax) = (250, 30), (320, 30) and
(320, 40), and evaluate the
program's own direct point law (ce.cov in mp, no series machinery) at every sample r. The
(320, 30) value is the reference; the larger of its differences from the other two estimates
its rounding and spectral-truncation error.
Phase C: containment, with absolute slack min(d - lo, hi - d) per entry against that error.
"""
import importlib.util
import json
import sys
from fractions import Fraction as F

prog, h2, out_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('h3_band_v2_probe', prog)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
mp, iv, mpf = m.mp, m.iv, m.mpf
rows = []


def entries(muE, SigE):
    out = [(('mu', a), muE[a]) for a in range(6)]
    out += [(('Sig', a, b), SigE[a][b]) for a in range(6) for b in range(6)]
    return out


def hook(L):
    ce = L['ce']
    neumann_law, first_cell_law, direct = L['neumann_law'], L['first_cell_law'], L['direct_point_law']
    CELLS, J, MT, ST = L['CELLS'], L['J'], L['MT'], L['ST']
    mu_ser, Sig_ser, horner = L['mu_ser'], L['Sig_ser'], L['horner']
    # phase A: enclosures at the certificate's precision, endpoints kept as exact mpf
    cases = []
    for c1, c2 in CELLS:
        cell = '(%s,%s]' % (m.frac_label(c1), m.frac_label(c2))
        if c1 == 0:
            pts = [c2, c2 / 2, c2 / 16, F(1, 10 ** 6), F(1, 10 ** 8)]
            muC, SigC, _, _, _ = first_cell_law()
        else:
            pts = [c1, (c1 + c2) / 2, c2]
            muC, SigC, _, _, _ = neumann_law(c1, c2)
        encC = [(k, (mpf(e.a), mpf(e.b))) for k, e in entries(muC, SigC)]
        for r in pts:
            if c1 == 0:
                t = m.iv_frac(r)
                muP = [horner([mu_ser[j][a] for j in range(J + 1)], t) + iv.mpf((-MT, MT)) for a in range(6)]
                SigP = [[horner([Sig_ser[j][a][b] for j in range(J + 1)], t) + iv.mpf((-ST, ST)) for b in range(6)]
                        for a in range(6)]
            else:
                muP, SigP, _, _, _ = neumann_law(r, r)
            encP = [(k, (mpf(e.a), mpf(e.b))) for k, e in entries(muP, SigP)]
            cases.append((cell, r, encC, encP))
    # phase B: direct law with the lattice rebuilt at higher precision
    ref = {}
    for dps, kmax in ((250, 30), (320, 30), (320, 40)):
        ce.configure(dps=dps, kmax=kmax)
        for cell, r, _, _ in cases:
            muD, SigD = direct(mpf(r.numerator) / r.denominator)
            ref.setdefault(str(r), {})[(dps, kmax)] = [+x for x in muD] + [+SigD[a][b] for a in range(6) for b in range(6)]
    mp.dps = 320
    # phase C: containment with absolute slack against the direct-law error estimate
    for cell, r, encC, encP in cases:
        d320 = ref[str(r)][(320, 30)]
        err_dps = max(abs(x - y) for x, y in zip(ref[str(r)][(250, 30)], d320))
        err_k = max(abs(x - y) for x, y in zip(ref[str(r)][(320, 40)], d320))
        err = max(err_dps, err_k)
        for name, enc in (('cell enclosure', encC), ('point enclosure', encP)):
            inside = True
            slack, slack_at, width_at = mpf('inf'), None, None
            for (key, (lo, hi)), d in zip(enc, d320):
                inside = inside and (lo <= d <= hi)
                s = min(d - lo, hi - d)
                if s < slack:
                    slack, slack_at, width_at = s, key, hi - lo
            rows.append({'check': name, 'cell': cell, 'r': str(r), 'inside': bool(inside),
                         'min_abs_slack': mp.nstr(slack, 4), 'at': str(slack_at), 'width_there': mp.nstr(width_at, 4),
                         'direct_err_dps': mp.nstr(err_dps, 4), 'direct_err_kmax': mp.nstr(err_k, 4),
                         'direct_err_est': mp.nstr(err, 4), 'resolved': bool(slack > 1000 * err)})
            print(json.dumps(rows[-1]), flush=True)
    ce.configure(dps=100, kmax=30)


m.PROBE_HOOK = hook
rc = m.main(['ceil', '--h2-dir', h2])
with open(out_path, 'w') as fh:
    for row in rows:
        fh.write(json.dumps(row) + '\n')
print('PROBE2 rc', rc, 'rows', len(rows), 'not inside', sum(1 for x in rows if not x['inside']),
      'unresolved', sum(1 for x in rows if not x['resolved']))
sys.exit(rc)
