"""Real zeros and critical points of H_t on [a,b] by fine scan + Newton/bisection refinement."""
import mpmath as mp

def refine_root(f, a, b, fa, fb, tol=None):
    """Illinois-type bracketing on [a,b] with f(a)f(b)<0; f returns mpf."""
    if tol is None:
        tol = mp.mpf(10) ** (-(mp.mp.dps // 2 + 10))
    a, b = mp.mpf(a), mp.mpf(b)
    side = 0
    for _ in range(400):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if fc == 0:
            return c
        if (fc > 0) == (fb > 0):
            b, fb = c, fc
            if side == -1:
                fa /= 2
            side = -1
        else:
            a, fa = c, fc
            if side == +1:
                fb /= 2
            side = +1
        if abs(b - a) < tol:
            break
    return (a * fb - b * fa) / (fb - fa)

def scan(hf, t, a, b, dz):
    """returns (zeros, crits) of H_t in [a,b]; crits as list of (c, H(c))."""
    zs, cs = [], []
    n = int(mp.ceil((mp.mpf(b) - a) / dz))
    grid = [mp.mpf(a) + k * (mp.mpf(b) - a) / n for k in range(n + 1)]
    vals = [hf.derivs(z, t, 1) for z in grid]
    H = lambda z: hf.derivs(z, t, 0)[0]
    Hp = lambda z: hf.derivs(z, t, 1)[1]
    for i in range(n):
        (h0, d0), (h1, d1) = vals[i], vals[i + 1]
        if d0 * d1 < 0:
            c = refine_root(Hp, grid[i], grid[i + 1], d0, d1)
            cs.append((c, H(c)))
    # count sign changes of H on merged sorted list of grid points and critical points
    pts = sorted([(g, v[0]) for g, v in zip(grid, vals)] + cs, key=lambda p: p[0])
    for (z0, h0), (z1, h1) in zip(pts, pts[1:]):
        if h0 * h1 < 0:
            zs.append(refine_root(H, z0, z1, h0, h1))
    return zs, cs
