"""Section 4 counterfamily 2: repeated base G (uniform 1/25 on {-2..2}^2, L=2pi) + eps * uniform fill,
built directly on the large torus L_N = 2 pi N with the q^(N) weights of eq. (1)."""
import sys, json
import numpy as np
sys.path.insert(0, sys.argv[1])
from jetcov import modes, gamma, row_matrix

out = {}
# ---- base field one-contact floor m_G over all frames (and reflected frames)
wb = np.full(25, 1 / 25)
ths = np.linspace(0, np.pi / 2, 2001)  # square symmetry: period pi/2 suffices; also check full circle coarse
lb = np.array([np.linalg.eigvalsh(gamma(wb, 2, 1.0, t))[0] for t in ths])
ths_full = np.linspace(0, 2 * np.pi, 721)
lb_full = [np.linalg.eigvalsh(gamma(wb, 2, 1.0, t))[0] for t in ths_full]
lb_ref = [np.linalg.eigvalsh(gamma(wb, 2, 1.0, t, reflect=True))[0] for t in ths_full]
mG = float(lb.min())
out["base"] = dict(m_G=mG, argmin_theta=float(ths[lb.argmin()]), lam_min_theta0=float(lb[0]),
                   lam_min_theta_pi4=float(lb[1000]), full_circle_min=float(min(lb_full)),
                   reflected_min=float(min(lb_ref)),
                   review_lower_bound=361267200 / 319913861015774089)
# axial LDL pivots (compare with author JSON 1,2,14/5,72/25,2,4,14/5)
G0 = gamma(wb, 2, 1.0, 0.0)
Lc = np.linalg.cholesky(G0); out["base"]["axial_pivots"] = list(np.round(np.diag(Lc) ** 2, 12))
out["base"]["axial_det"] = float(np.linalg.det(G0)); out["base"]["axial_trace"] = float(np.trace(G0))
out["base"]["pi4_det"] = float(np.linalg.det(gamma(wb, 2, 1.0, np.pi / 4)))

rows = []
for N in (4, 5, 8, 16, 32, 64):
    K = 2 * N; om = 1.0 / N
    n1, n2, rep = modes(K)
    size = (4 * N + 1) ** 2
    for eps in (0.5, 1e-2, 1e-4, float(N) ** -12):
        w = np.full(n1.shape, eps / size)
        lat = (n1 % N == 0) & (n2 % N == 0)
        w[lat] += (1 - eps) / 25  # n = N m, m in {-2..2}^2
        assert lat.sum() == 25 and np.all(w > 0) and abs(w.sum() - 1) < 1e-12
        m12 = float(np.sum(w * ((n1 ** 2 + n2 ** 2) / N ** 2) ** 6))
        # one-contact floor over frames, direct from q^(N)
        tt = np.linspace(0, np.pi / 2, 91)
        lN = min(np.linalg.eigvalsh(gamma(w, K, om, t, x=(0.37, -1.1)))[0] for t in tt)
        # two-contact at h=(2 pi j,0), j=floor(N/2): fourteen-row covariance, same frame
        j = N // 2; h = np.array([2 * np.pi * j, 0.0]); x = np.array([0.37, -1.1])
        res = {}
        for t in (0.0, 0.3, np.pi / 4):
            L1 = row_matrix(w, K, om, x, t); L2 = row_matrix(w, K, om, x + h, t)
            S = np.vstack([L1, L2]); Sig = S @ S.T
            d = (L1[0] - L2[0]) / np.sqrt(2)  # unit vector selecting (F(x)-F(x+h))/sqrt2
            res[round(t, 3)] = dict(lam14=float(np.linalg.eigvalsh(Sig)[0]), diffvar=float(d @ d))
        # fill-only correlation at h (no cancellation issue)
        wf = np.full(n1.shape, 1.0 / size)
        Lf1 = row_matrix(wf, K, om, x, 0.0); Lf2 = row_matrix(wf, K, om, x + h, 0.0)
        RH = float(Lf1[0] @ Lf2[0])
        Rn = float(row_matrix(w, K, om, x, 0.0)[0] @ row_matrix(w, K, om, x + h, 0.0)[0])
        # a NON-lattice macroscopic separation for contrast
        h2 = np.array([2 * np.pi * j + np.pi / 2, 0.0])
        Rnon = float(row_matrix(w, K, om, x, 0.0)[0] @ row_matrix(w, K, om, x + h2, 0.0)[0])
        rows.append(dict(N=N, eps=eps, M12=m12, one_contact_floor=lN, claimed_floor=(1 - eps) * mG,
                         floor_ok=bool(lN >= (1 - eps) * mG * (1 - 1e-9)),
                         R_H=RH, claimed_R_H=1 / (4 * N + 1), R_N=Rn, claimed_R_N=1 - eps + eps / (4 * N + 1),
                         claimed_small=eps * (1 - 1 / (4 * N + 1)), two_contact=res,
                         torus_dist_over_L=min(j, N - j) / N, R_nonlattice=Rnon))
out["family"] = rows
print(json.dumps(out, indent=1, default=float))
