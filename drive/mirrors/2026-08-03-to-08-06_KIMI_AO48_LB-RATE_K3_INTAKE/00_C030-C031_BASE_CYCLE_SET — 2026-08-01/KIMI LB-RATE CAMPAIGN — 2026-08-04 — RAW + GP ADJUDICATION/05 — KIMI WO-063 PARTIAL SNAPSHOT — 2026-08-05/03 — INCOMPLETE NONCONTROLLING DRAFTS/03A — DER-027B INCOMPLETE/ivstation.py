"""Interval limit-intensity evaluator: port of c027 station.py to ILS/iv.
Enclosure semantics: returns lam = [lo,hi] containing the exact limit functional
lambda(y) for every y in the input box, under the universal hull rule:
  - fully generic + typed-sign-definite box  -> two-sided [lo,hi]
  - proven-degenerate box (q/window/variance escape, or untyped) -> exact [0,0]
  - ambiguous (sign straddle)                -> hull [0, hi]  (sound: true value
    is either 0 or the formula value, both <= hi; and >= 0)
Fatal straddle (division by interval containing 0, pivot straddle, c6/st2
straddle) raises Split -> caller subdivides.
"""
from fractions import Fraction as Fr
from mpmath import iv, mp

def _mid(t):
    return (float(t.a) + float(t.b)) / 2
from ivlam import (ILS, ISc, V, IVT, NM, Split, cser, mat, mmul, mT, frmat_inv,
                   const_congr, build_frame, neumann_inv, is_exact_zero)
import ivlam

B = Fr(6, 5)
Mt = (Fr(0), Fr(0)); St = (Fr(1), Fr(0))
FUN6 = [((0, 0), Mt), ((1, 0), Mt), ((0, 1), Mt), ((0, 0), St), ((1, 0), St), ((0, 1), St)]
MON6 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0)]
G6E = [0, 1, 1, 0, 1, 1]; C6E = [0, 1, 1, 2, 2, 3]
MON9A = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2)]
MON9B = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (0, 3)]
G9E = [0, 1, 1, 0, 1, 1, 0, 1, 1]; C9E = [0, 1, 1, 2, 2, 2, 3, 3, 3]
OUTY = [((0, 0), None), ((1, 0), None), ((0, 1), None)]   # points filled with Yt
H9 = [((2, 0), Mt), ((1, 1), Mt), ((0, 2), Mt),
      ((2, 0), St), ((1, 1), St), ((0, 2), St),
      ((2, 0), None), ((1, 1), None), ((0, 2), None)]

PHI0 = 1 / iv.sqrt(2 * iv.pi)   # phi(0)

def _phi_tail_hi(xmag):
    """upper bound of Phi(-x) for x>0 via Mills ratio: phi(x)/x (rigorous)."""
    return iv.exp(-xmag * xmag / 2) / (iv.sqrt(2 * iv.pi) * xmag)

def Phi_iv(z, K=64):
    """certified interval enclosure of the standard normal CDF at interval z."""
    a, b = z.a, z.b
    lo = _Phi_point(a, K)[0]; hi = _Phi_point(b, K)[1]
    return iv.mpf((lo, hi))

def _Phi_point(x, K):
    """certified [lo,hi] for Phi at a single mpf endpoint x."""
    if x <= -6:
        hi = _phi_tail_hi(-x)
        return [iv.mpf(0), hi]
    if x >= 6:
        t = _phi_tail_hi(x)
        return [1 - t, iv.mpf(1)]
    # power series Phi(x) = 1/2 + phi0 * sum_k (-1)^k x^{2k+1}/(2^k k! (2k+1))
    x2 = x * x
    s = iv.mpf(0)
    term = iv.mpf(x)          # x^{2k+1}/(2^k k!) at k=0 ... build iteratively
    fac = iv.mpf(1)
    xp = iv.mpf(x)
    for k in range(K):
        t = xp / (fac * (2 * k + 1))
        s = s + ((-1) ** k) * t
        xp = xp * x2 / 2      # x^{2(k+1)+1}/2^{k+1}
        fac = fac * (k + 1)
    # tail: |R_K| <= phi0 * |t_K| / (1 - q), q = x2(2K+1)/(2(K+1)(2K+3))
    q = x2 * (2 * K + 1) / (2 * (K + 1) * (2 * K + 3))
    if not (q.b < 1):
        raise Split('Phi series ratio')
    tK = abs(xp) / (fac * (2 * K + 1))
    R = PHI0 * tK / (1 - q)
    mid = iv.mpf('0.5') + PHI0 * s + iv.mpf((-R.b, R.b))
    return [mid.a, mid.b]

DEN = (V(B * B + 2) * Phi_iv(V(B) / iv.sqrt(2))
       + iv.sqrt(2) * V(B) * iv.exp(-V(B) * V(B) / 4) / iv.sqrt(2 * iv.pi))

def _Trows(W, ge, ce, vecs):
    n = len(W); m = len(vecs)
    rows = []
    for j in range(n):
        s = ILS(0, [])
        for p in range(m):
            if not is_exact_zero(W[j][p]):
                s = s + vecs[p].shift(ge[p]) * W[j][p]
        rows.append(s.shift(-ce[j]))
    return rows

def _Krows(OUT, FUN, W, ge, ce, Yt):
    K = []
    for (ao, po) in OUT:
        p_y = Yt if po is None else (V(po[0]), V(po[1]))
        Sop = [cser(ao, FUN[p][0], (p_y[0] - V(FUN[p][1][0]), p_y[1] - V(FUN[p][1][1])))
               for p in range(len(FUN))]
        K.append(_Trows(W, ge, ce, Sop))
    return K

_CACHE = {}

def frame6():
    """6-pin frame, computed once (exact points -> zero-width intervals)."""
    if 'f6' not in _CACHE:
        ivlam.NM = 17
        Spp6, Vh6, W6, G6, G06 = build_frame(FUN6, MON6, G6E, C6E)
        X6 = neumann_inv(G6, G06, K=13)
        vals6 = [ISc(B), ILS(0, []), ILS(0, []),
                 ISc(B) - ILS(3, [V(Fr(1, 6))]), ILS(0, []), ILS(0, [])]
        u6 = _Trows(W6, G6E, C6E, vals6)
        _CACHE['f6'] = (W6, X6, u6)
    return _CACHE['f6']

def sdiv(A, Bs, border):
    """series division A/Bs; Bs leading coefficient at absolute order `border`;
    result offset A.off-border. Mirrors c027 sdiv."""
    b0 = Bs.coef(border)
    if b0.a <= 0 <= b0.b:
        raise Split('sdiv leading-coefficient straddle')
    off = A.off - border
    a = A.c
    c = []
    for k in range(len(a)):
        x = a[k]
        for j in range(min(k, len(c))):
            bi = k - j
            t = Bs.coef(border + bi)
            x = x - c[j] * t
        c.append(x / b0)
    return ILS(off, c)

def _hull_series(s1, s2):
    off = min(s1.off, s2.off)
    end = max(s1.off + len(s1.c), s2.off + len(s2.c))
    c = []
    for K in range(off, end):
        t1 = s1.coef(K); t2 = s2.coef(K)
        c.append(iv.mpf(min(t1.a, t2.a), max(t1.b, t2.b)))
    return ILS(off, c)

def station_iv(Y1, Y2, want_diag=False):
    """Yt = (Y1,Y2) intervals. Returns dict with lam enclosure + diagnostics."""
    W6, X6, u6 = frame6()
    ivlam.NM = 17
    Yt = (Y1, Y2)
    OUT = [((0, 0), None), ((1, 0), None), ((0, 1), None)]
    KYr = _Krows(OUT, FUN6, W6, G6E, C6E, Yt)
    KY = mmul(KYr, X6)
    mo = [sum((KY[i][j] * u6[j] for j in range(6)), ILS(0, [])) for i in range(3)]
    Soo = mat(3)
    for i, (ai, pi) in enumerate(OUT):
        for j, (aj, pj) in enumerate(OUT):
            Soo[i][j] = cser(ai, aj, (iv.mpf(0), iv.mpf(0)))
    KKt = mmul(KY, mT(KYr))
    So = [[Soo[i][j] - KKt[i][j] for j in range(3)] for i in range(3)]
    detS = So[1][1] * So[2][2] - So[1][2] * So[1][2]
    c6v = detS.coef(6)
    if c6v.a <= 0 <= c6v.b:
        raise Split('c6 straddle')
    if c6v.b < 0:
        raise Split('c6 negative (impossible for a determinant): engine bug')
    adj = [[So[2][2], -So[1][2]], [-So[1][2], So[1][1]]]
    mg = [mo[1], mo[2]]; Sfg = [So[0][1], So[0][2]]
    qn = (mg[0] * (adj[0][0] * mg[0] + adj[0][1] * mg[1])
          + mg[1] * (adj[1][0] * mg[0] + adj[1][1] * mg[1]))
    # genericity: coefficients below order 6 must vanish (structurally)
    q_esc = _order_class(qn, 6)
    if q_esc == 'straddle':
        raise Split('qn order straddle')
    qv = qn.coef(6) / c6v
    cross = Sfg[0] * (adj[0][0] * mg[0] + adj[0][1] * mg[1]) \
          + Sfg[1] * (adj[1][0] * mg[0] + adj[1][1] * mg[1])
    muN = mo[0] * detS - cross - ISc(B) * detS
    w_esc = _order_class(muN, 9)
    if w_esc == 'straddle':
        raise Split('muN order straddle')
    s2N = So[0][0] * detS - (Sfg[0] * (adj[0][0] * Sfg[0] + adj[0][1] * Sfg[1])
                             + Sfg[1] * (adj[1][0] * Sfg[0] + adj[1][1] * Sfg[1]))
    v_esc = _order_class(s2N, 12)
    if v_esc == 'straddle':
        raise Split('s2N order straddle')
    st2 = 36 * s2N.coef(12) / c6v
    if not (st2.a > 0):
        raise Split('st2 straddle')
    st = iv.sqrt(st2)
    m = 6 * muN.coef(9) / c6v
    if q_esc == 'dead' or w_esc == 'dead' or v_esc == 'dead':
        return dict(lam=iv.mpf(0), osc=iv.mpf(0), mode='dead',
                    q=float(_mid(qv)), m=float(_mid(m))) if want_diag else dict(lam=iv.mpf(0), osc=iv.mpf(0), mode='dead')
    Pw = Phi_iv(-m / st) - Phi_iv((-1 - m) / st)
    # clip branch for v*
    if m.a > -1 and m.b < 0:
        vser = sdiv(muN, detS, 6)
    elif m.b <= -1:
        vser = ILS(3, [V(Fr(-1, 6))])
    elif m.a >= 0:
        vser = ILS(3, [iv.mpf(0)])
    else:
        vser = _hull_series(sdiv(muN, detS, 6),
                            ILS(3, [iv.mpf(max(m.a, -1), min(max(m.b, -1), 0)) / 6]))
    # 9-pin frame
    ivlam.NM = 12
    FUN9 = FUN6 + [((0, 0), Yt), ((1, 0), Yt), ((0, 1), Yt)]
    built = None
    for MB, bname in [(MON9A, 'A'), (MON9B, 'B')]:
        try:
            Spp9, Vh9, W9, G9, G09 = build_frame(FUN9, MB, G9E, C9E)
            X9 = neumann_inv(G9, G09, K=6)
            built = (W9, X9, bname)
            break
        except Split:
            continue
    if built is None:
        raise Split('9-frame degenerate both bases')
    W9, X9, bname = built
    H9r = [((2, 0), Mt), ((1, 1), Mt), ((0, 2), Mt),
           ((2, 0), St), ((1, 1), St), ((0, 2), St),
           ((2, 0), None), ((1, 1), None), ((0, 2), None)]
    K9r = mmul(_Krows(H9r, FUN9, W9, G9E, C9E, Yt), X9)
    vals9 = [ISc(B), ILS(0, []), ILS(0, []),
             ISc(B) - ILS(3, [V(Fr(1, 6))]), ILS(0, []), ILS(0, []),
             ISc(B) + vser, ILS(0, []), ILS(0, [])]
    u9 = _Trows(W9, G9E, C9E, vals9)
    mh = [sum((K9r[i][j] * u9[j] for j in range(9)), ILS(0, [])) for i in range(9)]
    dM = mh[0] * mh[2] - mh[1] * mh[1]
    dS = mh[3] * mh[5] - mh[4] * mh[4]
    dY = mh[6] * mh[8] - mh[7] * mh[7]
    tr = mh[0] + mh[2]
    cM = dM.coef(2); cS = dS.coef(2); cY = dY.coef(2); trc = tr.coef(1)
    A = iv.exp(-qv / 2) / (2 * iv.pi * iv.sqrt(c6v))
    Nabs = abs(cM) * abs(cS) * abs(cY)
    hi = A.b * Pw.b * Nabs.b / DEN.a
    signs = [cM.a > 0, cS.b < 0, cY.b < 0, trc.b < 0]
    antis = [cM.b <= 0, cS.a >= 0, cY.a >= 0, trc.a >= 0]
    if any(antis):
        lam = iv.mpf(0); mode = 'untyped'
    elif all(signs):
        N = (-1 if cS.b < 0 else 1) * 1
        # cM>0, cS<0, cY<0 -> |cM cS cY| = cM*(-cS)*(-cY) = cM*cS*cY (positive)
        Nv = cM * (-cS) * (-cY)
        lo = A.a * Pw.a * Nv.a / DEN.b
        lam = iv.mpf((max(lo, mp.mpf(0)), hi)); mode = 'typed'
    else:
        lam = iv.mpf((mp.mpf(0), hi)); mode = 'hull'
    out = dict(lam=lam, osc=lam.b - lam.a, mode=mode, basis=bname,
               q=float(_mid(qv)), m=float(_mid(m)),
               st=float(_mid(st)), Pw=float(_mid(Pw)),
               cM=float(_mid(cM)), cS=float(_mid(cS)),
               cY=float(_mid(cY)), tr=float(_mid(trc)),
               c6=float(_mid(c6v)))
    return out

def _order_class(s, K):
    """classify series s against structural order K:
    'generic' if all coefficients below K are exact zero,
    'dead' if some coefficient below K is definitely nonzero,
    'straddle' otherwise."""
    dead = False
    for j in range(s.off, K):
        t = s.coef(j)
        if is_exact_zero(t):
            continue
        if t.a == 0 and t.b == 0:
            continue
        if t.a <= 0 <= t.b:
            return 'straddle'
        dead = True
    return 'dead' if dead else 'generic'
