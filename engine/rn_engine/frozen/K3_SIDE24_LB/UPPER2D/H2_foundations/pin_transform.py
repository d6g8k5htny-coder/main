#!/usr/bin/env python3
"""
pin_transform.py -- H2 foundations LIBRARY 2: the canonical exact-pin
transformation library for the 2D upper-theorem campaign (work-order items
176-197).

OBJECTS (all conventions explicit and tested in selftest_pin_transform.py).

Raw pins (P6 order):
    P6_r = (f(M_r), f_x(M_r), f_y(M_r), f(S_r), f_x(S_r), f_y(S_r)),
    M_r = (-r/2, 0), S_r = (r/2, 0).

Corrected pin basis (the program's typed pair-Palm basis):
    V_r = (f(M), f_x(M), f_y(M), (f_x(S)-f_x(M))/r, (f_y(S)-f_y(M))/r,
           [f(S)-f(M)-(r/2)(f_x(S)+f_x(M))]/r^3) = B_r P6_r
    (B_r = matrix_B; det B_r = r^{-5} exactly -- symbolic).

Hermite-normalized pins (LPW candidate's section 5):
    U_r = T_r P6_r,  T_r = D_r^{-1} V^{-1} R_r,
    D_r = diag(1, r, r, r^2, r^2, r^3),  R_r = diag(1, r, r, 1, r, r),
    V = the 6x6 evaluation matrix on the polynomial basis (1, x, y, x^2, xy,
        x^3) at the UNIT separation points m = -1/2, s = +1/2.

ROW-MAJOR CONVENTION (explicit; the formula T_r = D_r^{-1} V^{-1} R_r holds
with THIS V and fails with its transpose):
    V[i][j] = evaluation_i(basis_j),  i.e. row i is one of the six pin
    evaluations (p(m), p_x(m), p_y(m), p(s), p_x(s), p_y(s)) in P6 order,
    column j is one of the basis polynomials (1, x, y, x^2, xy, x^3).
    det V = -1 exactly (sympy).  The column-major (transposed) choice has the
    SAME determinant -1 but does NOT satisfy the formula -- so the convention
    is load-bearing and is pinned by an explicit negative test, not by det.

Verified symbolically (sympy exact rational arithmetic, r and b symbols):
    det D_r = r^9,  det R_r = r^4,  det V = -1,
    T_r == the candidate's displayed 6x6 matrix,  det T_r = -r^{-5},
    T_r (b, 0, 0, b - r^3/6, 0, 0)^T = (b - r^3/12, -r^2/4, 0, 0, 0, 1/3)^T,
    det B_r = r^{-5},  C_r = T_r B_r^{-1} has det -1 and T_r = C_r B_r,
    and T_r recovers polynomial coefficients EXACTLY through the cubic span:
    T_r P6(p) = c for p = sum_j c_j basis_j (Hermite exactness; this is what
    the trapezoidal correction term -(r/2)(f_x(S)+f_x(M)) buys).

U_0 limit: U_r -> (f(0), f_x(0), f_y(0), f_xx(0)/2, f_xy(0), f_xxx(0)/6) in
Gaussian L2; numerically tested against cov_exact endpoint moments.

CERTIFIED INVERSES (theorem-grade).  certified_inverse never ships a bare
inverse: it certifies, by interval Neumann-series arithmetic (E = I - A B,
rho = ||E||_F < 1/2 required), the condition bound
    kappa_cert = ||A||_F (||B||_F + ||B||_F rho/(1-rho)),
and REFUSES (CertificationError) unless  kappa_cert * consumer_eps <= CERT_TOL.
In the normalized frame (A = T_r G T_r^T, eigenfloor O(1)) this passes with
wide margin down to r = 1e-5 and below; in the RAW frame at small r the raw
pin covariance has lambda_min = Theta(r^6), so kappa_cert ~ r^{-6} and the
guard refuses -- treating the raw inverse as numerically safe without
normalization is a defect class and is a named mutation.

MUTATION HOOKS (self-test only; each rejected fail-closed, both modes):
    transpose_T       T_r replaced by its transpose;
    cubic_sign        sign of the cubic Hermite coordinate (row 6) flipped;
    drop_trapezoid    trapezoidal correction removed from B_r row 6;
    raw_inverse_unsafe  the conditioning guard is disabled.

FAIL-CLOSED CONTRACT.  Public entries validate inputs (PinInputError); no
bare asserts; deterministic (sympy exact arithmetic + mpmath at cov_exact's
configured dps); no wall clock.  Numeric entry points come in mpf (fast) and
iv (interval, certified) paths.
"""
import sympy as sp
import mpmath
from mpmath import mp, mpf, iv

import cov_exact as ce   # sister library (same directory); sets dps at import

CERT_TOL = mpf('1e-8')   # shipped-inverse accuracy budget at consumer precision


class PinInputError(ValueError):
    """Raised on any contract violation (fail-closed input validation)."""


class CertificationError(RuntimeError):
    """Raised when a certified bound cannot be established (fail-closed)."""


MUTATIONS = ("transpose_T", "cubic_sign", "drop_trapezoid", "raw_inverse_unsafe")

_r, _b = sp.symbols('r b')
_x, _y = sp.symbols('x y')

P6_ORDER = ("f(M_r)", "f_x(M_r)", "f_y(M_r)", "f(S_r)", "f_x(S_r)", "f_y(S_r)")
BASIS = (sp.Integer(1), _x, _y, _x**2, _x*_y, _x**3)
U0_LIMIT = ("f(0)", "f_x(0)", "f_y(0)", "f_xx(0)/2", "f_xy(0)", "f_xxx(0)/6")

_EVALS = ("p", "p_x", "p_y")   # per point, P6 order


def _evals_of(p, x0):
    pt = {_x: x0, _y: 0}
    return [p.subs(pt), sp.diff(p, _x).subs(pt), sp.diff(p, _y).subs(pt)]


# --------------------------------------------------------------------------
# symbolic matrices (sympy exact rational arithmetic; r a sympy symbol or exact)
# --------------------------------------------------------------------------
def matrix_V(convention='row-major'):
    """The 6x6 evaluation matrix at UNIT separation m=-1/2, s=+1/2.
    convention='row-major'  : V[i][j] = evaluation_i(basis_j)  (canonical);
    convention='column-major': the transpose (basis rows).  det = -1 both
    ways, but only 'row-major' satisfies T_r = D_r^{-1} V^{-1} R_r."""
    rows = []
    for which in ('m', 's'):
        x0 = sp.Rational(-1, 2) if which == 'm' else sp.Rational(1, 2)
        for e in range(3):
            rows.append([_evals_of(basis_j, x0)[e] for basis_j in BASIS])
    V = sp.Matrix(rows)
    if convention == 'row-major':
        return V
    if convention == 'column-major':
        return V.T
    raise PinInputError("convention must be 'row-major' or 'column-major'")


def matrix_D(r):
    """D_r = diag(1, r, r, r^2, r^2, r^3); det = r^9."""
    return sp.diag(1, r, r, r**2, r**2, r**3)


def matrix_R(r):
    """R_r = diag(1, r, r, 1, r, r); det = r^4."""
    return sp.diag(1, r, r, 1, r, r)


def matrix_B(r, drop_trapezoid=False):
    """B_r: raw pins -> corrected typed pair-Palm basis V_r = B_r P6_r.
    Row 6 carries the trapezoidal correction -(1/(2 r^2)) (f_x(M)+f_x(S));
    det B_r = r^{-5} with or without it, so the correction is pinned by the
    Hermite exactness test, not by the determinant."""
    tr = 0 if drop_trapezoid else -1 / (2 * r**2)
    return sp.Matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, -1/r, 0, 0, 1/r, 0],
        [0, 0, -1/r, 0, 0, 1/r],
        [-1/r**3, tr, 0, 1/r**3, tr, 0]])


def matrix_C(r):
    """C_r = T_r B_r^{-1}: corrected basis -> Hermite-normalized U_r.
    det C_r = -1; C_r B_r = T_r."""
    return sp.simplify(matrix_T(_r).subs(_r, r) * matrix_B(r) ** -1)


def displayed_T(r):
    """The LPW candidate's displayed T_r (section 5 of the candidate document;
    columns follow P6_r).  Kept hard-coded as the independent reference the
    derived T_r = D_r^{-1} V^{-1} R_r is verified against."""
    return sp.Matrix([
        [sp.Rational(1, 2), r/8, 0, sp.Rational(1, 2), -r/8, 0],
        [-3/(2*r), sp.Rational(-1, 4), 0, 3/(2*r), sp.Rational(-1, 4), 0],
        [0, 0, sp.Rational(1, 2), 0, 0, sp.Rational(1, 2)],
        [0, -1/(2*r), 0, 0, 1/(2*r), 0],
        [0, 0, -1/r, 0, 0, 1/r],
        [2/r**3, 1/r**2, 0, -2/r**3, 1/r**2, 0]])


_T_CACHE = {}

def matrix_T(r, mutation=None):
    """The canonical Hermite pin transform T_r = D_r^{-1} V^{-1} R_r
    (row-major V).  mutation in MUTATIONS applies the named defect
    (self-test only)."""
    if mutation is not None and mutation not in MUTATIONS:
        raise PinInputError("unknown mutation %r" % (mutation,))
    if mutation == 'raw_inverse_unsafe':
        return matrix_T(r)          # does not touch T; disables the inverse guard
    key = ('T', mutation)
    if key not in _T_CACHE:
        if mutation == 'transpose_T':
            T = displayed_T(_r).T
        elif mutation == 'cubic_sign':
            T = sp.Matrix(matrix_T(_r).tolist())   # mutable copy
            T[5, :] = -T[5, :]
        elif mutation == 'drop_trapezoid':
            T = sp.simplify(matrix_C(_r) * matrix_B(_r, drop_trapezoid=True))
        else:
            T = sp.simplify(matrix_D(_r) ** -1 * matrix_V('row-major') ** -1 * matrix_R(_r))
        _T_CACHE[key] = T
    return _T_CACHE[key].subs(_r, r)


def pin_values(r, b):
    """The prescribed raw pin values (b, 0, 0, b - r^3/6, 0, 0)^T."""
    return sp.Matrix([b, 0, 0, b - r**3/6, 0, 0])


def u_symbolic(r, b, mutation=None):
    """u_r = T_r (b, 0, 0, b - r^3/6, 0, 0)^T
    = (b - r^3/12, -r^2/4, 0, 0, 0, 1/3)^T  (verified in the self-test)."""
    return sp.simplify(matrix_T(r, mutation) * pin_values(r, b))


# --------------------------------------------------------------------------
# numeric evaluation (mpf fast path; iv certified path)
# --------------------------------------------------------------------------
def _check_r(r):
    rv = mpf(r)
    if not rv > 0:
        raise PinInputError("r must be positive (T_r is singular at r = 0)")
    return rv


def _t_entries(mode, mutation):
    """Return the 6x6 entries of P(r) = r^3 T_r as exact-rational-coefficient
    polynomials (lists of (Fraction-free) sympy rationals, low degree first),
    derived from the symbolic matrix so the numeric path IS the verified one."""
    key = ('poly', mutation)
    if key not in _T_CACHE:
        T = matrix_T(_r, mutation)
        polys = []
        for i in range(6):
            row = []
            for j in range(6):
                p = sp.Poly(sp.expand(T[i, j] * _r**3), _r)
                row.append([sp.Rational(c) for c in p.all_coeffs()[::-1]])
            polys.append(row)
        _T_CACHE[key] = polys
    return _T_CACHE[key]


def _eval_poly(coeffs, x):
    """Horner with exact rational coefficients (mpf arithmetic)."""
    acc = None
    for c in reversed(coeffs):
        term = mpf(int(c.p)) / int(c.q)
        acc = term if acc is None else acc * x + term
    return acc if acc is not None else mpf(0)


def matrix_T_num(r, mode='mpf', mutation=None):
    """T_r as a 6x6 list-of-lists of mpf (mode='mpf') or iv.mpf (mode='iv').
    Entries are Laurent polynomials in r with rational coefficients;
    r^3 T_r is polynomial (degree <= 4) and is Horner-evaluated exactly."""
    rv = _check_r(r)
    polys = _t_entries(mode, mutation)
    if mode == 'mpf':
        x = rv
        out = [[_eval_poly(polys[i][j], x) / rv**3 for j in range(6)] for i in range(6)]
    elif mode == 'iv':
        x = iv.mpf(rv)
        out = []
        for i in range(6):
            row = []
            for j in range(6):
                acc = None
                for c in reversed(polys[i][j]):
                    term = iv.mpf(int(c.p)) / int(c.q)
                    acc = term if acc is None else acc * x + term
                row.append((acc if acc is not None else iv.mpf(0)) / x**3)
            out.append(row)
    else:
        raise PinInputError("mode must be 'mpf' or 'iv'")
    return out


def apply_T(p6, r, mode='mpf', mutation=None):
    """U_r = T_r P6_r for a numeric 6-vector P6_r (list, P6 order)."""
    if len(p6) != 6:
        raise PinInputError("P6_r must have 6 components, order " + str(P6_ORDER))
    T = matrix_T_num(r, mode, mutation)
    if mode == 'mpf':
        v = [mpf(x) for x in p6]
    elif mode == 'iv':
        v = [iv.mpf(mpf(x)) for x in p6]
    else:
        raise PinInputError("mode must be 'mpf' or 'iv'")
    return [sum((T[i][j] * v[j] for j in range(6)), v[0] * 0) for i in range(6)]


def transform_cov(G, r, mode='mpf', mutation=None):
    """Gamma_U = T_r Gamma_P T_r^T for a 6x6 raw pin covariance (list of
    lists or mp.matrix)."""
    T = matrix_T_num(r, mode, mutation)
    zero = mpf(0) if mode == 'mpf' else iv.mpf(0)
    Gm = _as_rows(G)
    TG = [[sum((T[i][k] * Gm[k][j] for k in range(6)), zero) for j in range(6)] for i in range(6)]
    return [[sum((TG[i][k] * T[j][k] for k in range(6)), zero) for j in range(6)] for i in range(6)]


def pin_covariance(r, impl='spectral', mode='mpf'):
    """The exact raw 6-pin covariance of P6_r under the periodized law
    (cov_exact), pins at M_r = (-r/2, 0), S_r = (r/2, 0)."""
    rv = _check_r(r)
    pos = [(-rv/2, mpf(0)), (rv/2, mpf(0))]
    jet = [(0, 0), (1, 0), (0, 1)]
    G = []
    for p in range(2):
        for a in jet:
            row = []
            for q in range(2):
                for c in jet:
                    row.append(ce.cov(a, c, pos[p][0] - pos[q][0],
                                      pos[p][1] - pos[q][1], impl, mode))
            G.append(row)
    return G


# --------------------------------------------------------------------------
# certified inversion (Neumann-series certificate; guard fail-closed)
# --------------------------------------------------------------------------
_IV_TYPE = type(iv.mpf(0))


def _as_rows(G):
    """Accept a 6x6 list-of-lists or mp.matrix; return list-of-lists."""
    try:
        return [[G[i][j] for j in range(6)] for i in range(6)]
    except (TypeError, KeyError):
        return [[G[i, j] for j in range(6)] for i in range(6)]


def _to_iv_matrix(G):
    rows = _as_rows(G)
    return [[g if isinstance(g, _IV_TYPE) else iv.mpf(mpf(g)) for g in row] for row in rows]


def _mid_matrix(G_iv):
    M = mp.matrix(6, 6)
    for i in range(6):
        for j in range(6):
            lo, hi = mpf(G_iv[i][j].a), mpf(G_iv[i][j].b)  # endpoints first
            M[i, j] = (lo + hi) / 2
    return M


def _frob_bound_iv(A):
    """Rigorous upper bound on the Frobenius norm of an interval matrix:
    every contained real matrix M has ||M||_F <= sqrt(sum maxabs(A_ij)^2).
    (Squared intervals straddling zero would give negative lower ends; the
    max-abs form avoids that and is still rigorous.)"""
    tot = mpf(0)
    for row in A:
        for x in row:
            m = max(abs(mpf(x.a)), abs(mpf(x.b)))
            tot += m * m
    return mp.sqrt(tot)


def certified_inverse(G, r, frame='normalized', consumer_eps=None, guard=True,
                      mutation=None):
    """Certified inverse of the 6x6 raw pin covariance G.

    frame='normalized' (canonical): invert A = T_r G T_r^T and map back
        G^{-1} = T_r^T A^{-1} T_r   (T_r exact, so the map-back is lossless).
    frame='raw': invert G directly (refused by the guard at small r).

    Certificate (interval Neumann series, B an approximate inverse):
        E = I - A B (interval), rho = ||E||_F;  require rho < 1/2;
        ||A^{-1} - B||_F <= ||B||_F rho/(1 - rho);
        kappa_cert = ||A||_F (||B||_F + err),  lambda_min >= 1/(||B||_F + err).
    The guard REFUSES (CertificationError) unless
        kappa_cert * consumer_eps <= CERT_TOL (= 1e-8),
    i.e. a consumer at unit roundoff consumer_eps inherits entry accuracy at
    most CERT_TOL.  consumer_eps defaults to 1e-16 (float64-grade consumer).

    Returns (A_or_G_inverse_as_interval_6x6, certificate dict).  With
    mutation='raw_inverse_unsafe' the guard is disabled -- the self-test
    proves the certificate itself condemns the result."""
    if mutation is not None and mutation not in MUTATIONS:
        raise PinInputError("unknown mutation %r" % (mutation,))
    if consumer_eps is None:
        consumer_eps = mpf('1e-16')
    consumer_eps = mpf(consumer_eps)
    if not consumer_eps > 0:
        raise PinInputError("consumer_eps must be positive")
    if frame not in ('normalized', 'raw'):
        raise PinInputError("frame must be 'normalized' or 'raw'")
    rv = _check_r(r)
    G_iv = _to_iv_matrix(G)
    if frame == 'normalized':
        T = matrix_T_num(rv, 'iv', mutation)
        zero = iv.mpf(0)
        TG = [[sum((T[i][k] * G_iv[k][j] for k in range(6)), zero) for j in range(6)]
              for i in range(6)]
        A_iv = [[sum((TG[i][k] * T[j][k] for k in range(6)), zero) for j in range(6)]
                for i in range(6)]
    else:
        A_iv = G_iv
    B = mp.inverse(_mid_matrix(A_iv))
    B_iv = [[iv.mpf(B[i, j]) for j in range(6)] for i in range(6)]
    # E = I - A B in interval arithmetic
    E = []
    for i in range(6):
        row = []
        for j in range(6):
            s = iv.mpf(0)
            for k in range(6):
                s = s + A_iv[i][k] * B_iv[k][j]
            row.append((iv.mpf(1) if i == j else iv.mpf(0)) - s)
        E.append(row)
    rho = _frob_bound_iv(E)
    if not rho < mpf('0.5'):
        raise CertificationError(
            "Neumann residual rho = %s >= 1/2; inverse not certifiable" % mp.nstr(rho, 3))
    Bnorm = _frob_bound_iv(B_iv)
    Anorm = _frob_bound_iv(A_iv)
    err = Bnorm * rho / (1 - rho)
    Ainv_norm = Bnorm + err
    kappa = Anorm * Ainv_norm
    cert = {
        'frame': frame, 'r': mp.nstr(rv, 6), 'rho': rho, 'err_F': err,
        'kappa_cert': kappa, 'lambda_min_lower': 1 / Ainv_norm,
        'consumer_eps': consumer_eps, 'kappa_eps': kappa * consumer_eps,
        'cert_tol': CERT_TOL,
    }
    eff_guard = guard and mutation != 'raw_inverse_unsafe'
    if eff_guard and not kappa * consumer_eps <= CERT_TOL:
        raise CertificationError(
            "uncertified inverse: kappa_cert*consumer_eps = %s > CERT_TOL = 1e-8 "
            "(frame=%s, r=%s); the normalized frame is the only safe route at small r"
            % (mp.nstr(kappa * consumer_eps, 3), frame, mp.nstr(rv, 4)))
    Ainv_iv = [[B_iv[i][j] + iv.mpf([-err, err]) for j in range(6)] for i in range(6)]
    if frame == 'normalized':
        T = matrix_T_num(rv, 'iv', mutation)
        zero = iv.mpf(0)
        Tt = [[T[j][i] for j in range(6)] for i in range(6)]
        TtA = [[sum((Tt[i][k] * Ainv_iv[k][j] for k in range(6)), zero) for j in range(6)]
               for i in range(6)]
        Ginv = [[sum((TtA[i][k] * T[k][j] for k in range(6)), zero) for j in range(6)]
                for i in range(6)]
        return Ginv, cert
    return Ainv_iv, cert
