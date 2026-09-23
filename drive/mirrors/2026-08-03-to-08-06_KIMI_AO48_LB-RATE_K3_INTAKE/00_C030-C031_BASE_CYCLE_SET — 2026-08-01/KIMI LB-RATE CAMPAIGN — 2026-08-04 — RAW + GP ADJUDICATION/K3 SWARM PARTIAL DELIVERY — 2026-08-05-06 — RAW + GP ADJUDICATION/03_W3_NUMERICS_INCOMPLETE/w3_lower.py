# Certified lower bound of int_box rho dy per box.
import math
from mpmath import iv, mpi
import mpmath as mp

import importlib.util
_spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3c)

ivm = iv.mpf
lo, hi, mid, width = w3c.lo, w3c.hi, w3c.mid, w3c.width
ck = w3c.ck
phi_pdf, Phi_cdf = w3c.phi_pdf, w3c.Phi_cdf


def _fup(x):
    return float(hi(iv.fabs(x))) * (1 + 1e-12)


def g_sensitivities(m, V, tag):
    """certified (L_m[3], L_V[3][3]) bounds on |dg/dtheta| at (m,V);
    g = E[|detH| 1{det<0}], H ~ N(m, V)."""
    VQ = w3c._vq_entries(V)
    trA = VQ[0][0] + VQ[1][1] + VQ[2][2]
    P2 = w3c._mat3_mul(VQ, VQ)
    S2 = P2[0][0] + P2[1][1] + P2[2][2]
    lamu = math.sqrt(_fup(S2))
    m0u = math.sqrt(_fup(m[0] ** 2 + m[1] ** 2 + m[2] ** 2))
    Vi, detV = w3c.inv3x3_spd(V, tag + " Vsens")
    KMAX = 10
    NK = 2 * KMAX
    # kappa derivative bounds
    dk_m = [0.0] * (NK + 1)
    dk_V = [0.0] * (NK + 1)
    for j in range(1, NK + 1):
        fac = 2.0 ** (j - 1) * math.factorial(j - 1)
        dk_m[j] = fac * (2 * j * m0u * lamu ** (j - 1)) * (1 + 1e-12)
        dk_V[j] = fac * (j * lamu ** (j - 1) + j * (j - 1) * m0u * m0u * lamu ** max(0, j - 2)) * (1 + 1e-12)
    kappas = [None]
    for j in range(1, NK + 1):
        Pj = VQ
        for _ in range(j - 1):
            Pj = w3c._mat3_mul(Pj, VQ)
        trj = Pj[0][0] + Pj[1][1] + Pj[2][2]
        mj = w3c._mQXm(m, Pj)
        kappas.append(ivm(2 ** (j - 1) * math.factorial(j - 1)) * (trj + j * mj))
    kapu = [0.0] + [_fup(kappas[j]) for j in range(1, NK + 1)]
    moms = [ivm(1), kappas[1]]
    for n in range(2, NK + 1):
        s = ivm(0)
        for k in range(1, n + 1):
            s += ivm(math.comb(n - 1, k - 1)) * kappas[k] * moms[n - k]
        moms.append(s)
    momu = [_fup(x) for x in moms]
    dm_m = [0.0] * (NK + 1)
    dm_V = [0.0] * (NK + 1)
    for n in range(1, NK + 1):
        sm = sv = 0.0
        for k in range(1, n + 1):
            cb = math.comb(n - 1, k - 1)
            sm += cb * (dk_m[k] * momu[n - k] + kapu[k] * dm_m[n - k])
            sv += cb * (dk_V[k] * momu[n - k] + kapu[k] * dm_V[n - k])
        dm_m[n] = sm * (1 + 1e-12)
        dm_V[n] = sv * (1 + 1e-12)
    Fi_m = [_fup(Vi[i][i]) for i in range(3)]
    Fi_V = [[_fup(Vi[i][i] * Vi[j][j] + Vi[i][j] ** 2) for j in range(3)] for i in range(3)]
    Qm = [m[2] / 2, -m[1], m[0] / 2]
    W_m = [2 * math.sqrt(_fup(VQ[a][a]) + _fup(Qm[a]) ** 2) * (1 + 1e-12) for a in range(3)]
    nu2 = _fup(w3c._mQXm(m, Vi)) / max(1e-300, float(lo(detV)))
    rho = min(0.9 / (2.0 * lamu), 0.4 / (lamu * max(1.0, nu2)), 0.25)
    t0 = rho / 3.0
    c1u = 2 * _fup(trA)
    e2A = ((VQ[0][0] * VQ[1][1] - VQ[0][1] * VQ[1][0])
           + (VQ[0][0] * VQ[2][2] - VQ[0][2] * VQ[2][0])
           + (VQ[1][1] * VQ[2][2] - VQ[1][2] * VQ[2][1]))
    c2u = 4 * _fup(e2A)
    c2r = 4 * e2A
    c3r = 2 * detV
    c3lo = float(lo(c3r)) if lo(c3r) > 0 else 0.0
    c3u = _fup(c3r)
    E1m = tuple(tuple((trA if i == j else ivm(0)) - VQ[i][j] for j in range(3)) for i in range(3))
    adjB = w3c._mat3_mul(VQ, VQ)
    adjB = tuple(tuple(adjB[i][j] - trA * VQ[i][j] + (e2A if i == j else ivm(0)) for j in range(3)) for i in range(3))
    n0u = _fup(w3c._mQXm(m, tuple(tuple(ivm(int(i == j)) for j in range(3)) for i in range(3))))
    n1u = 2 * _fup(w3c._mQXm(m, E1m))
    n2u = 4 * _fup(w3c._mQXm(m, adjB))

    def S1(dm):
        s = 0.0
        KS = 6
        for k in range(1, KS + 1):
            s += dm[2 * k] * t0 ** (2 * k - 1) / (math.factorial(2 * k) * (2 * k - 1))
        rem = dm[2 * KS] * t0 ** (2 * KS - 1) / (math.factorial(2 * KS) * (2 * KS - 1))
        rem /= (1 - (t0 / rho) ** 2)
        return (s + rem) * (1 + 1e-12)

    def dlo_panel(a, b):
        # |D(t)| >= max over the piecewise lower bounds on [a,b]
        v1 = max(0.0, 1.0 - c2u * b * b)
        v2 = max(0.0, c3lo * a ** 3 - c1u * b)
        v3 = 0.0
        if lo(c2r) > 0:
            v3 = max(0.0, float(lo(c2r)) * a * a - 1.0)
        return max(v1, v2, v3, 1e-300)

    def S23(theta, a_idx):
        # returns int_{t0}^inf |dphi/dtheta|/t^2 dt bound (float)
        if theta == 'm':
            Wth = W_m[a_idx]
            Fth = Fi_m[a_idx]
            n0t = 2 * m0u
            n1t = 4 * m0u * (c1u / 2 + lamu)
            n2t = 8 * m0u * lamu * lamu
            dD_over_D = None  # |D_theta| = 0 for mean
        else:
            i, j = a_idx
            Wth = 1e300
            Fth = Fi_V[i][j]
            n0t = 0.0
            n1t = 4 * m0u * m0u
            n2t = 8 * m0u * m0u * lamu
            dD_over_D = 2.0  # |D_theta|/|D| <= 2 t
        T1 = max(4 * t0, 2.0 / max(1e-12, math.sqrt(momu[2])))
        total = 0.0
        a = t0
        ratio = (T1 / t0) ** (1.0 / 40)
        for _ in range(40):
            b = a * ratio
            dl = dlo_panel(a, b)
            Nt = n0t + n1t * b + n2t * b * b
            Nu = n0u + n1u * b + n2u * b * b
            if dD_over_D is None:
                Udec = (1.0 / math.sqrt(dl)) * (b * Nt / dl)
            else:
                Udec = (1.0 / math.sqrt(dl)) * (0.5 * dD_over_D * b + b * (Nt + Nu * dD_over_D * b) / dl)
            U = min(b * Wth, math.sqrt(Fth), Udec) / (a * a)
            total += U * (b - a)
            a = b
        # tail [T1, inf): decay form with |D| >= c3lo t^3/2 for t >= Tc
        Tc = (2.0 / c3lo) ** (1.0 / 3.0) if c3lo > 1e-300 else None
        if Tc is not None:
            TT = max(T1, Tc)
            # add panel-ish bound on [T1, TT]
            if TT > T1:
                dl = dlo_panel(T1, TT)
                Nt = n0t + n1t * TT + n2t * TT * TT
                Nu = n0u + n1u * TT + n2u * TT * TT
                if dD_over_D is None:
                    Udec = (1.0 / math.sqrt(dl)) * (TT * Nt / dl)
                else:
                    Udec = (1.0 / math.sqrt(dl)) * (0.5 * dD_over_D * TT + TT * (Nt + Nu * dD_over_D * TT) / dl)
                U = min(math.sqrt(Fth), Udec) / (T1 * T1)
                total += U * (TT - T1)
            cc = (c3lo / 2.0) ** (-1.5)
            # |D_theta| <= dD t |D| handled via 0.5*dD*t ; |N_theta| <= n0t+n1t t+n2t t^2
            if dD_over_D is None:
                tail = cc * (n0t / 2.5 * TT ** -2.5 + n1t / 1.5 * TT ** -1.5 + n2t / 0.5 * TT ** -0.5)
            else:
                tail = cc * (0.5 * dD_over_D / 0.5 * TT ** -0.5
                             + (n0t / 2.5 * TT ** -2.5 + n1t / 1.5 * TT ** -1.5 + n2t / 0.5 * TT ** -0.5)
                             + (n0u + n1u + n2u) * dD_over_D / 1.5 * TT ** -1.5)
            tail = min(tail, math.sqrt(Fth) / TT)
        else:
            tail = math.sqrt(Fth) / T1
        return (total + tail) * (1 + 1e-12)

    L_m = [0.0] * 3
    L_V = [[0.0] * 3 for _ in range(3)]
    for a in range(3):
        L_m[a] = 0.5 * ((2.0 / math.pi) * (S1(dm_m) + S23('m', a)) + dk_m[1])
    for i in range(3):
        for j in range(3):
            L_V[i][j] = 0.5 * ((2.0 / math.pi) * (S1(dm_V) + S23('V', (i, j))) + dk_V[1])
    return L_m, L_V


# ---------------- convolution for int_box Pwin dy ----------------
def _phi(x):
    return phi_pdf(x)


def _dPhi_sub(y0, y1, depth=0):
    if depth >= 8 or float(w3c.hi((y1 - y0) / 2)) <= 0.75:
        return w3c.dPhi(y0, y1, "convl")
    ym = (y0 + y1) / 2
    return _dPhi_sub(y0, ym, depth + 1) + _dPhi_sub(ym, y1, depth + 1)


def _int_phi_lin(K, sig, x0, x1):
    """int_{x0}^{x1} phi((K + x)/sig) dx  = sig (Phi((K+x1)/sig) - Phi((K+x0)/sig))"""
    return sig * _dPhi_sub((K + x0) / sig, (K + x1) / sig)


def _int_x_phi_lin(K, sig, x0, x1):
    """int_{x0}^{x1} x phi((K+x)/sig) dx = sig^2 (phi((K+x0)/sig)-phi((K+x1)/sig)) - K sig (Phi1-Phi0)"""
    return (sig * sig * (_phi((K + x0) / sig) - _phi((K + x1) / sig))
            - K * sig * _dPhi_sub((K + x0) / sig, (K + x1) / sig))


PHI1 = None


def F_conv(c, Rm, sig_lo, A, B):
    """E_X[phi((|c - X| + Rm)/sig_lo)] with X ~ trapezoid on [-(A+B), A+B], A>=B>=0.
    c >= 0. Interval c allowed."""
    global PHI1
    if PHI1 is None:
        PHI1 = phi_pdf(ivm(1))
    if A < B:
        A, B = B, A
    M = A + B
    D = A - B
    # density: w(x) = 1/(2A) on |x|<=D, (M-|x|)/(4AB) on D<|x|<=M
    # F = int_{-M}^{M} phi((|c-x|+Rm)/sig) w(x) dx
    # split at x = c (and symmetric): substitute u = |c - x|:
    # int_{-M}^{M} f(|c-x|) w(x) dx = int_0^{c+M} f(u) [w(c+u) (if c+u<=M) + w(c-u) (if c-u>=-M)] du
    # with w defined piecewise; we integrate numerically by breaking at the kink points
    # u where c+u or c-u crosses +-D or +-M. Collect breakpoints:
    sig = sig_lo
    bps = set([0.0])
    for xx in (D, M):
        if c + xx >= 0:
            bps.add(c + xx)
        if c - xx >= 0:
            bps.add(c - xx)
    # kink where |c-u|+Rm = sig (phi(max(1,.)) sharpening)
    sigf = float(hi(sig))
    if sigf > Rm:
        bps.add(sigf - Rm)
    pts = sorted(bps)
    total = ivm(0)

    def w(x):
        # trapezoid density (interval-safe)
        ax = iv.fabs(x)
        if float(hi(ax)) <= D:
            return ivm(1) / ivm(2 * A)
        v = (ivm(M) - ax) / ivm(4 * A * B)
        if float(lo(v)) < 0:
            return mpi(0, max(0.0, float(hi(v))))
        return v

    # w may be interval in c; handle by evaluating w with the c-interval throughout.
    for i in range(len(pts)):
        u0 = pts[i]
        u1 = pts[i + 1] if i + 1 < len(pts) else c + M
        if u1 <= u0:
            continue
        # on [u0,u1], both w(c+u) and w(c-u) are linear in u (between kinks):
        # w(c+u) = wu0 + (wu1-wu0)/(u1-u0) (u-u0), similarly w(c-u)
        wp0 = w(c + u0)
        wp1 = w(c + u1)
        wm0 = w(c - u0)
        wm1 = w(c - u1)
        # sharp integrand: phi(max(1, (u+Rm)/sig)) -- const phi(1) where u+Rm <= sig_lo
        const_piece = (u1 + Rm <= float(lo(sig)))
        # integrate f(u)=phi((u+Rm)/sig) times each linear piece
        for (w0, w1) in ((wp0, wp1), (wm0, wm1)):
            if float(hi(w0)) == 0 and float(hi(w1)) == 0:
                continue
            slope = (w1 - w0) / ivm(u1 - u0)
            if const_piece:
                # phi(max(1,.)) = phi(1) on the whole piece; but the two pieces
                # correspond to |c-x| via u=|c-x|: both share the same u range
                base = ivm(u1 - u0)
                xpart = ivm((u1 - u0) ** 2) / 2
                total += PHI1 * (w0 * base + slope * xpart)
                continue
            base = _int_phi_lin(Rm, sig, u0, u1)
            xpart = _int_x_phi_lin(Rm, sig, u0, u1) - ivm(u0) * base
            total += w0 * base + slope * xpart
    return total


def box_lower(mt_c, gmt, Rm, vt, pgrad, g_lo_iv, win, area, tag):
    """certified lower bound of int_box rho dy.
    mt_c: center mt (interval, point); gmt=(dmt/dx, dmt/dy) intervals;
    Rm: quadratic remainder (float-ish interval); vt: interval over box;
    pgrad: interval over box; g_lo_iv: interval lower bound on g over box x window;
    win=(u0,u1) window; area = 4 r1 r2."""
    ck(lo(vt) > 0, "vt>0 %s" % tag)
    sig_lo = iv.sqrt(ivm(lo(vt)))
    sig_hi = iv.sqrt(ivm(hi(vt)))
    ell = ivm(hi(win[1]) - lo(win[0]))
    b_hi = ivm(hi(win[1]))
    b_lo = ivm(lo(win[0]))
    # c(u) = u - mt_c for u in window: c interval
    c_lo = b_lo - mt_c
    c_hi = b_hi - mt_c
    crange = mpi(lo(c_lo), hi(c_hi))
    # |c| max over window
    c_max = iv.fabs(crange)
    # gradient magnitudes: |g1| r1, |g2| r2 as A, B (with r from area? need r1, r2 separately)
    return crange, c_max, sig_lo, sig_hi
