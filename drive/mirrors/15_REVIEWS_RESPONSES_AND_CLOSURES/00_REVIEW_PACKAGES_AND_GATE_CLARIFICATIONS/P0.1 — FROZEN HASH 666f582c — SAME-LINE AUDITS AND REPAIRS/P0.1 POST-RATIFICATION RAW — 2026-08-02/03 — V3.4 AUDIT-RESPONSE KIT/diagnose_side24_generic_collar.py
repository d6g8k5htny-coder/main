#!/usr/bin/env python3
"""Numerical diagnostic for the generic transverse collar face.

This is not the proof.  It evaluates the exact Fourier-series covariance of
the side-24 periodized Bargmann--Fock field after a spherical spectral
truncation, using the corrected pair-pin basis and the anisotropically
normalized witness-gradient residuals.  It checks convergence to the local
contact Schur complement and records the ratio

    det Cov(grad f(y) | pair pins) /
        [varrho**6 * (4*X**2 + varrho**2)].

The analytic proof uses full Fourier support plus compactness; this script is
only a regression/counterexample search on the generic face.
"""

from __future__ import annotations

import math
import numpy as np


SIDE = 24.0
FREQ = 2.0 * math.pi / SIDE
CUTOFF = 6.0


def ck(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit("CHECK FAILED: " + message)


def lattice() -> tuple[np.ndarray, np.ndarray]:
    n = int(math.ceil(CUTOFF / FREQ)) + 1
    a = np.arange(-n, n + 1, dtype=float) * FREQ
    kx, ky, kz = np.meshgrid(a, a, a, indexing="ij")
    k = np.stack((kx.ravel(), ky.ravel(), kz.ravel()), axis=1)
    norm2 = np.einsum("ij,ij->i", k, k)
    keep = norm2 <= CUTOFF**2
    k = k[keep]
    w = np.exp(-norm2[keep] / 2.0)
    # K_24(0)=1 normalization.  A common scalar would not affect positivity,
    # but it does make the planar comparison ratio close to one.
    w /= w.sum()
    return k, np.sqrt(w)


K, SQRT_W = lattice()


def frame_from_t(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    t /= np.linalg.norm(t)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(t @ seed)) > 0.85:
        seed = np.array([0.0, 1.0, 0.0])
    e2 = seed - (seed @ t) * t
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(t, e2)
    return np.stack((t, e2, e3), axis=0)


def value_coeff(point: np.ndarray) -> np.ndarray:
    return SQRT_W * np.exp(1j * (K @ point))


def derivative_coeff(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return 1j * (K @ direction) * value_coeff(point)


def gram(rows: list[np.ndarray]) -> np.ndarray:
    a = np.stack(rows, axis=0)
    g = a @ a.conj().T
    return np.real_if_close(g, tol=1000).real


def schur(rows_pin: list[np.ndarray], rows_witness: list[np.ndarray]) -> np.ndarray:
    g = gram(rows_pin + rows_witness)
    n = len(rows_pin)
    pp = g[:n, :n]
    pw = g[:n, n:]
    return g[n:, n:] - pw.T @ np.linalg.solve(pp, pw)


def finite_rows(r: float, frame: np.ndarray, xi: np.ndarray):
    """Corrected pins V_r and normalized residuals R_r.

    Coordinates are centered at x=0, M=-rt/2, S=rt/2, y=r*xi.
    xi is expressed in the orthonormal frame (t,e2,e3).
    """
    t, e2, e3 = frame
    dirs = [t, e2, e3]
    xi_vec = xi @ frame
    m = -0.5 * r * t
    s = 0.5 * r * t
    y = r * xi_vec

    vm = value_coeff(m)
    vs = value_coeff(s)
    gm = [derivative_coeff(m, d) for d in dirs]
    gs = [derivative_coeff(s, d) for d in dirs]
    dg = [(gs[j] - gm[j]) / r for j in range(3)]
    dv = (vs - vm - 0.5 * r * (gs[0] + gm[0])) / r**3
    pins = [vm, *gm, *dg, dv]

    gy = [derivative_coeff(y, d) for d in dirs]
    # Transverse residuals: subtract the endpoint value and the pinned axial
    # first variation.  Conditional covariance is unchanged by these pin
    # combinations.
    a_t = float(xi[0] + 0.5)
    r2 = (gy[1] - gm[1] - r * a_t * dg[1]) / r
    r3 = (gy[2] - gm[2] - r * a_t * dg[2]) / r

    u = xi.copy()
    u[0] += 0.5
    axial_linear = u[0] * dg[0] + u[1] * dg[1] + u[2] * dg[2]
    cubic_cancel = 6.0 * xi[0] ** 2 - 1.5
    r1 = (gy[0] - gm[0] - r * axial_linear + r**2 * cubic_cancel * dv) / r**2
    return pins, [r1, r2, r3]


def limit_rows(frame: np.ndarray, xi: np.ndarray):
    """Contact-jet pins and leading generic-face residual operators."""
    t, e2, e3 = frame
    kt, k2, k3 = K @ t, K @ e2, K @ e3
    it, i2, i3 = 1j * kt, 1j * k2, 1j * k3
    one = SQRT_W.astype(complex)
    pins = [
        one,
        it * one,
        i2 * one,
        i3 * one,
        it**2 * one,
        it * i2 * one,
        it * i3 * one,
        -(it**3) * one / 12.0,
    ]
    vop = xi[1] * i2 + xi[2] * i3
    longitudinal = (xi[0] * it**2 * vop + 0.5 * it * vop**2) * one
    transverse2 = vop * i2 * one
    transverse3 = vop * i3 * one
    return pins, [longitudinal, transverse2, transverse3]


def q0(xi: np.ndarray) -> float:
    rho2 = float(xi[1] ** 2 + xi[2] ** 2)
    return rho2**3 * (4.0 * float(xi[0] ** 2) + rho2)


def ratio(r: float, frame: np.ndarray, xi: np.ndarray) -> tuple[float, float]:
    if r == 0.0:
        p, w = limit_rows(frame, xi)
    else:
        p, w = finite_rows(r, frame, xi)
    c = schur(p, w)
    ev = np.linalg.eigvalsh(c)
    return float(np.linalg.det(c) / q0(xi)), float(ev.min())


def main() -> None:
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.31, -0.77, 0.56]),
    ]
    xis = [
        np.array([-1.20, 0.35, 0.20]),
        np.array([-0.75, 0.40, -0.55]),
        np.array([-0.22, 0.28, 0.44]),
        np.array([0.23, -0.36, 0.31]),
        np.array([0.78, 0.62, 0.18]),
        np.array([1.25, -0.48, -0.39]),
    ]
    radii = [0.4, 0.2, 0.1, 0.05, 0.02, 0.0]
    all_ratios = []
    all_mins = []
    print(f"modes={len(K)}, cutoff={CUTOFF}, side={SIDE}")
    for ti, t in enumerate(directions):
        frame = frame_from_t(t)
        for xi in xis:
            vals = [ratio(r, frame, xi) for r in radii]
            ratios = [v[0] for v in vals]
            mins = [v[1] for v in vals]
            all_ratios.extend(ratios)
            all_mins.extend(mins)
            rel = abs(ratios[-2] / ratios[-1] - 1.0)
            print(
                f"t{ti} xi={xi.tolist()} ratios="
                + " ".join(f"{v:.6g}" for v in ratios)
                + f" rel(r=.02,limit)={rel:.3e}"
            )
            ck(min(mins) > 0.0, "nonpositive normalized Schur eigenvalue")
            ck(rel < 0.08, "finite-r normalized covariance not approaching contact limit")
    print(f"ratio range over diagnostic grid: [{min(all_ratios):.6g}, {max(all_ratios):.6g}]")
    print(f"minimum normalized Schur eigenvalue: {min(all_mins):.6g}")
    ck(min(all_ratios) > 0.0, "nonpositive determinant ratio")
    print("SCOPE LIMIT: truncated-Fourier diagnostic on a generic compact set;")
    print("the two-sided finite-r theorem uses exact full support and compactness.")
    print("ALL_CHECKS_PASS")


if __name__ == "__main__":
    main()
