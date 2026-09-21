"""Driver: certified minima (P1a), JS regional sup bounds (P1b), per-box
Taylor certificates (P2)."""
from mpmath import mp, mpf
from fractions import Fraction as Fr
from jets import J2, JS, _js, js_exp_neg_half, js_sqrt, js_Phi_diff, PHI0
import dag
from dag import (station_dag, lam_value_mpf, lam_assemble_J2, DEN, GLS, cser,
                 mat, mmul, mT, build_frame, neumann_inv, DagError, MON9A,
                 MON9B, G9E, C9E)

def det_mpf(M):
    n = len(M)
    A = [row[:] for row in M]
    d = mpf(1)
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(A[r][col]))
        if A[p][col] == 0:
            return mpf(0)
        if p != col:
            A[col], A[p] = A[p], A[col]
            d = -d
        pv = A[col][col]
        d = d * pv
        for r in range(col + 1, n):
            f = A[r][col] / pv
            for j in range(col, n):
                A[r][j] = A[r][j] - f * A[col][j]
    return d

def vh9_entries(MON, y1, y2):
    """Vh9 matrix for FUN9 at numeric point (y1,y2) (mpf)."""
    FUN = dag.FUN6 + [((0, 0), None), ((1, 0), None), ((0, 1), None)]
    n = len(FUN)
    Vh = [[None] * n for _ in range(n)]
    for i, (ai, pi) in enumerate(FUN):
        x, y = (y1, y2) if pi is None else (mpf(pi[0]), mpf(pi[1]))
        for a_idx, (px, py) in enumerate(MON):
            if ai == (0, 0): v = x ** px * y ** py
            elif ai == (1, 0): v = px * x ** (px - 1) * y ** py if px > 0 else mpf(0)
            elif ai == (0, 1): v = py * x ** px * y ** (py - 1) if py > 0 else mpf(0)
            Vh[i][a_idx] = v
    return Vh

def detvh9(MON, y1, y2):
    return det_mpf(vh9_entries(MON, y1, y2))

def dvh9_bound(MON, x0, x1, y0, y1):
    """crude sup of |grad detVh9| on the box [x0,x1]x[y0,y1] via monomial bounds."""
    FUN = dag.FUN6 + [((0, 0), None), ((1, 0), None), ((0, 1), None)]
    xa = max(abs(x0), abs(x1)); ya = max(abs(y0), abs(y1))
    n = len(FUN)
    # entry sups
    es = []
    for i, (ai, pi) in enumerate(FUN):
        xx = xa if pi is None else abs(pi[0]); yy = ya if pi is None else abs(pi[1])
        row = []
        for (px, py) in MON:
            if ai == (0, 0): v = xx ** px * yy ** py
            elif ai == (1, 0): v = px * xx ** max(px - 1, 0) * yy ** py if px > 0 else mpf(0)
            elif ai == (0, 1): v = py * xx ** px * yy ** max(py - 1, 0) if py > 0 else mpf(0)
            row.append(v)
        es.append(row)
    # |dVh/dy|: only the three y-rows (indices 6..8), monomial derivatives
    emax = max(max(r) for r in es)
    import math
    adj = mpf(math.factorial(n - 1)) * emax ** (n - 1)
    # sup of each y-row entry derivative wrt y1,y2
    gmax = mpf(0)
    for i in range(6, 9):
        ai = FUN[i][0]
        for (px, py) in MON:
            if ai == (0, 0):
                gx = px * xa ** max(px - 1, 0) * ya ** py
                gy = py * xa ** px * ya ** max(py - 1, 0)
            elif ai == (1, 0):
                gx = px * (px - 1) * xa ** max(px - 2, 0) * ya ** py if px > 1 else mpf(0)
                gy = px * py * xa ** max(px - 1, 0) * ya ** max(py - 1, 0)
            else:
                gx = px * py * xa ** max(px - 1, 0) * ya ** max(py - 1, 0)
                gy = py * (py - 1) * xa ** px * ya ** max(py - 2, 0) if py > 1 else mpf(0)
            gmax = max(gmax, abs(gx) + abs(gy))
    return adj * gmax * 3  # 3 y-rows

def net_min_max(f, box, h):
    """min/max of f over a rational net of spacing h covering box (mpf evals)."""
    x0, x1, y0, y1 = box
    lo = None; hi = None
    nx = int(round((x1 - x0) / h)); ny = int(round((y1 - y0) / h))
    for i in range(nx + 1):
        for j in range(ny + 1):
            v = f(mpf(x0 + i * h), mpf(y0 + j * h))
            lo = v if lo is None or v < lo else lo
            hi = v if hi is None or v > hi else hi
    return lo, hi

def lam_assemble_JS(res, minima):
    """Assemble lambda JS (sup bounds B0..B3) from JS ingredients.
    Valid on domains where the typing signs can go either way: the product
    A*Pw*N bounds |lambda| on the typed part and lambda = 0 elsewhere, so the
    JS of the product bounds lambda's derivative sups wherever lambda is C2
    (off the typing boundary and clip kink; those curves are handled by box
    subdivision / C0 fallback in P2)."""
    c6v, qv, st2, m = res['c6v'], res['qv'], res['st2'], res['m']
    cM, cS, cY = res['cM'], res['cS'], res['cY']
    s_min = mp.sqrt(minima['detS6'])
    A = js_exp_neg_half(qv).div(js_sqrt(c6v, minima['detS6']), s_min) / (2 * mp.pi)
    st = js_sqrt(st2, minima['st2'])
    st_min = mp.sqrt(minima['st2'])
    z1 = (-m).div(st, st_min)
    z2 = (-1 - m).div(st, st_min)
    Pw = js_Phi_diff(z1, z2)
    N = cM * cS * cY
    lam = A * Pw * N / DEN
    out = dict(res)
    out.update(A=A, Pw=Pw, N=N, lam=lam, st=st, z1=z1, z2=z2)
    return out
