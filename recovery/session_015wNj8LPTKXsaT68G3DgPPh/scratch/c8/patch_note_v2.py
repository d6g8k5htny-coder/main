import json, re, sys
from fractions import Fraction as Fr
D = '/home/user/wt_c8/frontiers/c8_normalizer_floor_planar_20260930/'
res = json.load(open(D + 'RESULTS.json'))
lo, hi = res['bands']['low'], res['bands']['high']
def parse(k):
    p = k.replace('r=[', '').replace('],b=[', ',').replace('],k=[', ',').replace(']', '').split(',')
    return [Fr(x) for x in p]
def dec(x, d=4):
    x = Fr(x); n = x.numerator * 10 ** d // x.denominator; s = str(n).rjust(d + 1, '0'); return s[:-d] + '.' + s[-d:]
def minr(rec, rmax, bk=None):
    best = None
    for k, v in rec['table'].items():
        p = parse(k); fl = Fr(v['floor'])
        if p[1] <= rmax and (bk is None or (p[2] == bk[0] and p[4] == bk[1])):
            if best is None or fl < best[0]: best = (fl, k)
    return best
zl, zh = Fr(lo['z_star']), Fr(hi['z_star']); zc = Fr(res['combined']['z_star'])
ratios = []
for rec in (lo, hi):
    for k, row in rec['float_controls'].items():
        ratios.append(Fr(rec['table'][k]['floor']) / Fr(row['r1,b0,k0']['mc']))
rmin, rmax_ = dec(min(ratios), 2), dec(max(ratios), 2)
nctrl = len(lo['float_controls']) + len(hi['float_controls'])
worst = min((c['mc'] - float(Fr(rec['table'][k]['floor']))) / c['se'] for rec in (lo, hi) for k, row in rec['float_controls'].items() for c in row.values())
n = open(D + 'NOTE.md').read()
def sub1(old, new):
    global n
    assert n.count(old) == 1, old[:80]
    n = n.replace(old, new)
# section 0
i0 = n.index('**Certified inequality.**'); i1 = n.index('This is the constant whose existence is [LP] (5.5)')
new0 = f"""**Certified inequality.** For every frame and every `(b, k)` with `0 <= b <= 1`, `1/2 <= k <= 2`:

    L >= 12,   1/64 <= r <= 1/2 :    Z_r / r^2  >=  z_*  =  {dec(zc)} ,
    L >= 10,   1/8  <= r <= 1/2 :    Z_r / r^2  >=  {dec(zh)}   (high band),
    L >= 12,   1/64 <= r <= 1/8 :    Z_r / r^2  >=  {dec(zl)}   (low band),

with the sharper sub-box floors of `RESULTS.json` (`bands.high.table`: 192 x 8 x 6 boxes on `[1/8, 1/2]`;
`bands.low.table`: 56 x 8 x 6 boxes on `[1/64, 1/8]`; `r` in steps of `1/512`, `b` of `1/8`, `k` of `1/4`): in particular
`Z_r / r^2 >= {dec(minr(lo, Fr(1,32))[0])}` for `r <= 1/32`, `>= {dec(minr(lo, Fr(1,16))[0])}` for `r <= 1/16`, `>= {dec(minr(hi, Fr(3,16))[0])}` for `1/8 <= r <= 3/16`,
`>= {dec(minr(hi, Fr(1,4))[0])}` for `1/8 <= r <= 1/4`, `>= {dec(minr(hi, Fr(3,8))[0])}` for `1/8 <= r <= 3/8` (all `b`, `k` in the band), and for every box the floor
scales as `k_lo^2 x q(box)` with `q` certified for `Z_r / (r^2 k^2)`. The high-band minimum is attained at
`{hi['argmin']}`, the low-band minimum at `{lo['argmin']}`. The low band needs `L >= 12`
because the periodization remainder enters the preconditioned functionals with the factor `r^-10` (section 2).

"""
n = n[:i0] + new0 + n[i1:]
sub1("`2` times the floor across the band (`0.50` to `0.85` as a ratio floor / MC).", f"`2` times the floor across the bands (`{rmin}` to `{rmax_}` as a ratio floor / MC).")
# section 2 torus paragraph
sub1("`eps_0 = 4.41e-17`; the cross-block Cholesky entries this produces are of size `1e-9` to `1e-6` and are absorbed below.",
     f"`eps_0 = 4.41e-17`; the cross-block Cholesky entries this produces are of size `1e-9` to `1e-6` and are absorbed below.\nThe preconditioned functionals carry coefficients up to `r^-5`, so a raw perturbation `eps` becomes `eps r^-10` in the\n`O(1)` covariances: `5e-6` at `L = 10`, `r = 1/8`, but `60` at `L = 10`, `r = 1/64`, where the interval Cholesky fails; the low\nband is therefore declared for `L >= 12`, where `eps(12, 1/8) = {lo['torus_epsilon_at_rmax']:.2e}` gives `3e-8` at `r = 1/64`.")
# section 3
i3 = n.index('## 3. Values'); i4 = n.index('## 4. Precision')
rows_hi = []
for lab, rmx in (('[1/8, 3/16]', Fr(3,16)), ('[1/8, 1/4]', Fr(1,4)), ('[1/8, 3/8]', Fr(3,8)), ('[1/8, 1/2]', Fr(1,2))):
    fl, k = minr(hi, rmx); p = parse(k); rows_hi.append('| `%s` | `%s` | `b in [%s, %s], k in [%s, %s]` |' % (lab, dec(fl), p[2], p[3], p[4], p[5]))
rows_lo = []
for lab, rmx in (('[1/64, 1/32]', Fr(1,32)), ('[1/64, 1/16]', Fr(1,16)), ('[1/64, 1/8]', Fr(1,8))):
    fl, k = minr(lo, rmx); p = parse(k); rows_lo.append('| `%s` | `%s` | `b in [%s, %s], k in [%s, %s]` |' % (lab, dec(fl), p[2], p[3], p[4], p[5]))
def bktable(rec, r1, r2, lab1, lab2):
    bs = sorted(set(parse(k)[2] for k in rec['table'])); ks = sorted(set(parse(k)[4] for k in rec['table']))
    out = [f"| box (`b`, `k`) | {lab1}: floor / MC (s.e.) | {lab2}: floor / MC (s.e.) | `z_0(b_lo, k_lo)` |", "|---|---|---|---|"]
    for b0 in (bs[0], bs[len(bs)//2], bs[-1]):
        for k0 in (ks[0], ks[len(ks)//2], ks[-1]):
            cells = []; z0 = None
            for rm in (r1, r2):
                fl, key = minr(rec, rm, (b0, k0)); mc = rec['float_controls'][key]['r1,b0,k0']
                cells.append('`%s` / `%.3f` (`%.3f`)' % (dec(fl), mc['mc'], mc['se'])); z0 = mc['z0']
            out.append('| `b in [%s, %s], k in [%s, %s]` | %s | %s | `%.3f` |' % (b0, b0 + (bs[1]-bs[0]), k0, k0 + (ks[1]-ks[0]), cells[0], cells[1], z0))
    return '\n'.join(out)
sec3 = f"""## 3. Values (`RESULTS.json`)

Combined floor: `z_* = {dec(zc)}` (`L >= 12`, `r in [1/64, 1/2]`); high band `{dec(zh)}` (attained at `{hi['argmin']}`,
valid for `L >= 10`); low band `{dec(zl)}` (attained at `{lo['argmin']}`). Floors by `r`-range (minimum over all
`b`, `k` sub-boxes with `r` in the range):

| `r` range | floor of `Z_r / r^2` | at `(b, k)` box |
|---|---|---|
{chr(10).join(rows_lo)}
{chr(10).join(rows_hi)}

Floors at fixed `(b, k)` corners (minimum over the `r`-bands in the range), against the floating Monte Carlo of
`Z_r / r^2` at the worst corner `(r_hi, b_lo, k_lo)` of the box attaining the floor and the limit `z_0(b_lo, k_lo)`.
High band:

{bktable(hi, Fr(1,4), Fr(1,2), '`r <= 1/4`', '`r <= 1/2`')}

Low band:

{bktable(lo, Fr(1,16), Fr(1,8), '`r <= 1/16`', '`r <= 1/8`')}

"""
n = n[:i3] + sec3 + n[i4:]
# section 4
m = re.search(r"give ratios floor / MC between `(\d\.\d\d)` and `(\d\.\d\d)` over\n\s*the `(\d+)` control boxes", n)
assert m, 'ratio sentence'
n = n.replace(m.group(0), f"give ratios floor / MC between `{rmin}` and `{rmax_}` over\n  the `{nctrl}` control boxes of the two bands (the low band reaches `0.85`–`0.90`)")
sub1("relative at two point bands (`--check` rule); the point-band enclosures have widths below `1e-6`.",
     "relative at three point bands (`r = 1/4, 3/8, 1/32`; `--check` rule); the point-band enclosures have widths below `1e-6`.")
# section 5
sub1("[LP] section 7, or any `d >= 3` normalizer. Planar only, `L >= 10` (the `L = 24` SIDE24 torus included), aligned or\nrotated frames, `r in [1/8, 1/2]`, `b in [0, 1]`, `k in [1/2, 2]`; nothing for `r < 1/8` (where the floor tends to\n`z_0`) or `r > 1/2`.",
     "[LP] section 7, or any `d >= 3` normalizer. Planar only, `L >= 12` on `[1/64, 1/2]` and `L >= 10` on `[1/8, 1/2]` (the\n`L = 24` SIDE24 torus included), aligned or rotated frames, `b in [0, 1]`, `k in [1/2, 2]`; nothing for `r < 1/64`\n(where the floor tends to `z_0`), `r > 1/2`, or `L < 10`.")
# section 6
sub1("re-derives the certified floor of seven sub-boxes\n(the argmin box and six boxes spread over the band, one per 32 `r`-bands) from scratch",
     "re-derives the certified floor of twelve sub-boxes\n(each band's argmin box, six high-band boxes one per 32 `r`-bands, four low-band boxes one per 18) from scratch")
sub1("checks that the stored `z_star` is the minimum of the table;", "checks that each band's stored `z_star` is the minimum of its table and the combined statement;")
sub1("preconditioned regression against the direct floating regression at two point bands", "preconditioned regression against the direct floating regression at three point bands")
sub1("regenerates **every** one of the 9216 sub-box floors and requires exact\nagreement with `RESULTS.json` (`--bands i,j,...` restricts to listed `r`-bands for a partial check);",
     "regenerates **every** one of the 11904 sub-box floors (9216 high, 2688 low) and requires exact\nagreement with `RESULTS.json` (`--band low,high` and `--rbands i,j,...` restrict to bands or listed `r`-bands for a partial check);")
sub1("(`workflow_dispatch`, two cores, about two hours)", "(`workflow_dispatch`, two cores, about three hours)")
sub1("The full run (`--procs 4`, about 64 minutes\non four cores) regenerates `RESULTS.json` deterministically", "The full run (`--procs 4`; 64 minutes for the high band and about 20 for the low band\non four cores) regenerates `RESULTS.json` deterministically")
assert 'n.a.' not in n and '__' not in n.replace('__init__', '')
open(D + 'NOTE.md', 'w').write(n)
# README
r = open(D + 'README.md').read()
old = """Proves, for the parent torus kernel of [LP] in `d = 2`, every orthonormal frame and every side `L >= 10`, and for every
`r in [1/8, 1/2]`, `b in [0, 1]`, `k in [1/2, 2]`, the explicit floor `Z_r / r^2 >= z_*` for the full normalizer"""
assert r.count(old) == 1
r = r.replace(old, f"""Proves, for the parent torus kernel of [LP] in `d = 2`, every orthonormal frame, every side `L >= 12`, and every
`r in [1/64, 1/2]`, `b in [0, 1]`, `k in [1/2, 2]`, the explicit floor `Z_r / r^2 >= z_* = {dec(zc)}` (the band
`[1/8, 1/2]` alone holds for `L >= 10`; the band `[1/64, 1/8]` has floor `{dec(zl)}`) for the full normalizer""")
r = r.replace("The value and the full table of per-sub-box floors are in `RESULTS.json` (`z_star`, `table`)", "The values and the full tables of per-sub-box floors are in `RESULTS.json` (`combined`, `bands.high`, `bands.low`)")
r = r.replace("`--check` replays the manifest-pinned floor exactly on seven\n  sub-boxes", "`--check` replays the manifest-pinned floor exactly on twelve\n  sub-boxes of the two bands")
r = r.replace("`--check-full --procs N` regenerates all 9216 sub-box floors and compares them exactly (about an hour on four cores;", "`--check-full --procs N` regenerates all 11904 sub-box floors and compares them exactly (about 85 minutes on four cores;")
open(D + 'README.md', 'w').write(r)
print('NOTE/README patched; z combined', dec(zc), 'low', dec(zl), 'high', dec(zh), 'ratios', rmin, rmax_, 'controls', nctrl, 'worst se', round(worst, 1))
