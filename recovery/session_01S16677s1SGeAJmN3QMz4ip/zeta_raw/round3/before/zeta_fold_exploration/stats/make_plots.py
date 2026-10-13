"""Static SVG plots (matplotlib is not installed): effective local exponents s(x) = d log ECDF / d log x
for normalized bar lengths and normalized zero gaps, with approximate Poisson error bars.
Also writes the plotted numbers to CSV.

Reads results/raw/ecdf_samples.npz (from analyze.py); writes results/stats/effective_exponent_*.svg
and results/stats/local_slopes.csv. With --png and Pillow (pinned in requirements-plot.txt),
also rasterizes PNG previews into results/raw/ (not committed)."""
import os
import sys
import numpy as np

from stats_paths import out_path, raw_path

D = np.load(raw_path("ecdf_samples.npz"))
SERIES = [("gauss", "Gaussian process (SE cov.)", "#2a78d6"),
          ("cue128", "CUE N=128", "#eb6834"),
          ("cue12", "CUE N=12", "#1baf7a"),
          ("zetaA", "Hardy Z, t in [1e4,1e5]", "#eda100"),
          ("zetaB", "Hardy Z, t in [1e5,1e6]", "#e87ba4")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def local_slopes(x, centers, half):
    xs = np.sort(x); n = xs.size
    out = []
    for c in centers:
        e1, e2 = c / half, c * half
        n1 = np.searchsorted(xs, e1, "right"); n2 = np.searchsorted(xs, e2, "right")
        if n1 < 10:
            out.append((c, np.nan, np.nan, n1)); continue
        lr = np.log(e2 / e1)
        out.append((c, np.log(n2 / n1) / lr, np.sqrt((n2 - n1) / (n1 * n2)) / lr, n1))
    return np.array(out)


def svg_plot(fname, title, subtitle, xlabel, ylabel, xlim, ylim, data, refs, xticks, yticks):
    W, H = 760, 500
    ml, mr, mt, mb = 70, 230, 70, 60
    pw, ph = W - ml - mr, H - mt - mb
    lx0, lx1 = np.log10(xlim[0]), np.log10(xlim[1])
    X = lambda x: ml + (np.log10(x) - lx0) / (lx1 - lx0) * pw
    Y = lambda y: mt + (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * ph
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{ml}" y="28" font-size="16" font-weight="600" fill="{INK}">{title}</text>',
         f'<text x="{ml}" y="48" font-size="12" fill="{INK2}">{subtitle}</text>']
    for t in xticks:
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{mt+ph}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{X(t):.1f}" y="{mt+ph+18}" font-size="11" fill="{INK2}" text-anchor="middle">{t:g}</text>')
    for t in yticks:
        s.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{ml-8}" y="{Y(t)+4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{t:g}</text>')
    s.append(f'<text x="{ml+pw/2}" y="{H-14}" font-size="12" fill="{INK2}" text-anchor="middle">{xlabel}</text>')
    s.append(f'<text transform="translate(18,{mt+ph/2}) rotate(-90)" font-size="12" fill="{INK2}" text-anchor="middle">{ylabel}</text>')
    for j, (yv, lab) in enumerate(refs):
        s.append(f'<line x1="{ml}" x2="{ml+pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        ly = mt + 10 + 22 * len(data) + 18 + 40 * j
        s.append(f'<line x1="{ml+pw+18}" x2="{ml+pw+40}" y1="{ly}" y2="{ly}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        for i2, part in enumerate(lab.split("|")):
            s.append(f'<text x="{ml+pw+46}" y="{ly+4+15*i2}" font-size="12" fill="{INK2}">{part}</text>')
    for k, (nm, lab, col, arr) in enumerate(data):
        ok = np.isfinite(arr[:, 1]) & (arr[:, 0] >= xlim[0]) & (arr[:, 0] <= xlim[1])
        a = arr[ok]
        jit = 10 ** ((k - 2) * 0.012)
        pts = " ".join(f"{X(c*jit):.1f},{Y(np.clip(v, *ylim)):.1f}" for c, v in zip(a[:, 0], a[:, 1]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for c, v, e, n in a:
            if np.isfinite(e):
                s.append(f'<line x1="{X(c*jit):.1f}" x2="{X(c*jit):.1f}" y1="{Y(np.clip(v-e,*ylim)):.1f}" y2="{Y(np.clip(v+e,*ylim)):.1f}" stroke="{col}" stroke-width="1.5"/>')
            s.append(f'<circle cx="{X(c*jit):.1f}" cy="{Y(np.clip(v,*ylim)):.1f}" r="4" fill="{col}" stroke="{SURF}" stroke-width="2"><title>{lab}: x={c:.3g}, s={v:.3f} +/- {e:.3f} (n below window={int(n)})</title></circle>')
        ly = mt + 10 + 22 * k
        s.append(f'<line x1="{ml+pw+18}" x2="{ml+pw+40}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<text x="{ml+pw+46}" y="{ly+4}" font-size="12" fill="{INK}">{lab}</text>')
    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="{INK2}" stroke-width="1"/>')
    s.append("</svg>")
    with open(out_path(fname), "w") as f:
        f.write("\n".join(s))


centers = np.logspace(-3.75, -0.25, 15)
bars = [(nm, lab, col, local_slopes(D[f"{nm}_L"], centers, 10 ** 0.25)) for nm, lab, col in SERIES]
svg_plot("effective_exponent_bars.svg",
         "Short-bar exponent of the antiderivative barcode",
         "local slope of log ECDF(ell), half-decade windows; density ~ ell^(s-1); error bars: approx. Poisson SE",
         "bar length / median bar length", "local CDF slope s", (1e-4, 1), (0.3, 1.2),
         bars, [(2 / 3, "s = 2/3: linear repulsion|(density ell^-1/3)"), (1.0, "s = 1: quadratic repulsion|(density ell^0)")],
         [1e-4, 1e-3, 1e-2, 1e-1, 1], [0.4, 0.6, 0.8, 1.0, 1.2])
gc = np.logspace(np.log10(0.06), np.log10(0.8), 12)
gaps = [(nm, lab, col, local_slopes(D[f"{nm}_r"], gc, 10 ** 0.1)) for nm, lab, col in SERIES]
svg_plot("effective_exponent_gaps.svg",
         "Small-gap exponent of nearest-neighbour zero spacings",
         "local slope of log ECDF(r) over 0.2-decade windows; gap density ~ r^(s-1); bars = approx. Poisson SE",
         "gap / mean gap", "local CDF slope s", (0.05, 1), (0.5, 3.5),
         gaps, [(2.0, "s = 2: linear repulsion|(density r^1)"), (3.0, "s = 3: quadratic repulsion|(density r^2)")],
         [0.05, 0.1, 0.2, 0.5, 1], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
with open(out_path("local_slopes.csv"), "w") as f:
    f.write("quantity,dataset,x_center,local_slope_s,approx_se,n_below_window\n")
    for q, arrs in (("bars", bars), ("gaps", gaps)):
        for nm, lab, col, a in arrs:
            for c, v, e, n in a:
                f.write(f"{q},{nm},{c:.5g},{v:.5g},{e:.5g},{int(n)}\n")
print("ok")


# ---- PNG previews rasterized with Pillow from the SVG primitives (no matplotlib available) ----
def svg_to_png(svgname, pngname, scale=2):
    import re
    from PIL import Image, ImageDraw, ImageFont
    txt = open(out_path(svgname)).read()
    W, H = map(int, re.search(r'width="(\d+)" height="(\d+)"', txt).groups())
    im = Image.new("RGB", (W * scale, H * scale), SURF)
    dr = ImageDraw.Draw(im)
    def font(sz, bold=False):
        for p in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                return ImageFont.truetype(p, sz * scale)
        return ImageFont.load_default(size=sz * scale)
    A = lambda m, k, d=None: (m.group(0).split(f'{k}="')[1].split('"')[0] if f'{k}="' in m.group(0) else d)
    for m in re.finditer(r'<(line|polyline|circle|rect|text)([^>]*)>(?:([^<]*)</text>)?', txt):
        tag = m.group(1)
        if tag == "line":
            x1, x2, y1, y2 = (float(A(m, k)) * scale for k in ("x1", "x2", "y1", "y2"))
            w = int(float(A(m, "stroke-width", 1)) * scale)
            if A(m, "stroke-dasharray"):
                L = np.hypot(x2 - x1, y2 - y1); n = int(L / (10 * scale)) + 1
                for i in range(n):
                    a0 = i * 10 * scale / L; a1 = min(1, a0 + 6 * scale / L)
                    dr.line([(x1 + a0 * (x2 - x1), y1 + a0 * (y2 - y1)), (x1 + a1 * (x2 - x1), y1 + a1 * (y2 - y1))], fill=A(m, "stroke"), width=w)
            else:
                dr.line([(x1, y1), (x2, y2)], fill=A(m, "stroke"), width=w)
        elif tag == "polyline":
            pts = [tuple(float(v) * scale for v in p.split(",")) for p in A(m, "points").split()]
            dr.line(pts, fill=A(m, "stroke"), width=int(2 * scale), joint="curve")
        elif tag == "circle":
            cx, cy, r = (float(A(m, k)) * scale for k in ("cx", "cy", "r"))
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=A(m, "fill"), outline=SURF, width=2 * scale)
        elif tag == "rect" and A(m, "x"):
            x, y, w, h = (float(A(m, k)) * scale for k in ("x", "y", "width", "height"))
            dr.rectangle([x, y, x + w, y + h], outline=A(m, "stroke"), width=scale)
        elif tag == "text":
            t = m.group(3) or ""
            f = font(int(A(m, "font-size", 12)), A(m, "font-weight") == "600")
            anchor = {"middle": "ms", "end": "rs"}.get(A(m, "text-anchor"), "ls")
            if A(m, "transform"):
                tx, ty = (float(v) for v in re.search(r"translate\(([\d.]+),([\d.]+)\)", A(m, "transform")).groups())
                bb = dr.textbbox((0, 0), t, font=f)
                tmp = Image.new("RGBA", (bb[2] + 4, bb[3] + 4), (0, 0, 0, 0))
                ImageDraw.Draw(tmp).text((0, 0), t, font=f, fill=A(m, "fill"))
                tmp = tmp.rotate(90, expand=True)
                im.paste(tmp, (int(tx * scale - tmp.width / 2 - 2 * scale), int(ty * scale - tmp.height / 2)), tmp)
            else:
                dr.text((float(A(m, "x")) * scale, float(A(m, "y")) * scale), t, font=f, fill=A(m, "fill"), anchor=anchor)
    im.save(raw_path(pngname))


if "--png" in sys.argv[1:]:
    svg_to_png("effective_exponent_bars.svg", "effective_exponent_bars.png")
    svg_to_png("effective_exponent_gaps.svg", "effective_exponent_gaps.png")
    print("png ok")
