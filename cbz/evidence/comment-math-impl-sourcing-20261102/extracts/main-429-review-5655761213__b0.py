"""Independent jet-covariance builder for the ideal finite Gaussian field (FG eq. (1)).

F(x) = sqrt(q_0) C_0 + sum_{n in P} sqrt(2 q_n) [C_n cos(omega n.x) + S_n sin(omega n.x)]

Rows are built directly from the real coefficient representation J = L xi, Gamma = L L^T,
NOT from the moment formula used by the author controls.
Contact rows (FG Sec. 2, J_0): (F, F_u, F_uu, F_uuu, F_v, F_uv, F_vv) with orders (a,c)
in (u,v): (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2).
"""
import numpy as np

ROWS = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (0, 2))


def modes(K):
    n1, n2 = np.meshgrid(np.arange(-K, K + 1), np.arange(-K, K + 1), indexing="ij")
    n1 = n1.ravel(); n2 = n2.ravel()
    rep = (n1 > 0) | ((n1 == 0) & (n2 > 0))  # one member of each nonzero pair
    return n1, n2, rep


def row_matrix(weights, K, omega, x, theta, reflect=False):
    """weights: array over the (2K+1)^2 grid in modes(K) order (symmetric, positive).
    Returns L of shape (7, 1 + 2*#pairs): row coefficients of the seven contact rows at x."""
    n1, n2, rep = modes(K)
    zero = (n1 == 0) & (n2 == 0)
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    if reflect:
        v = -v
    xi1 = omega * n1[rep]; xi2 = omega * n2[rep]
    au = xi1 * u[0] + xi2 * u[1]
    cv = xi1 * v[0] + xi2 * v[1]
    ph = xi1 * x[0] + xi2 * x[1]
    amp = np.sqrt(2 * weights[rep])
    e = np.exp(1j * ph)
    L = np.zeros((7, 1 + 2 * rep.sum()))
    for r, (a, c) in enumerate(ROWS):
        P = (1j) ** (a + c) * au ** a * cv ** c * e
        if a == 0 and c == 0:
            L[r, 0] = np.sqrt(weights[zero][0])
        L[r, 1::2] = amp * P.real  # cos coefficient
        L[r, 2::2] = amp * P.imag  # sin coefficient
    return L


def gamma(weights, K, omega, theta, x=(0.0, 0.0), reflect=False):
    L = row_matrix(weights, K, omega, np.array(x), theta, reflect)
    return L @ L.T


def lam_min_over_frames(weights, K, omega, nth=721):
    ths = np.linspace(0, 2 * np.pi, nth)
    vals = [np.linalg.eigvalsh(gamma(weights, K, omega, t))[0] for t in ths]
    i = int(np.argmin(vals))
    return vals[i], ths[i], vals
