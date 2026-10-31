# C022 bookkeeping closures: R0-UNIF addendum · Lemma LB restatement note

**Cycle:** C022 · **Date:** 2026-07-09 · **Freeze:** sha256 `c8fe1ed1…f4a`

---

## Part A — R0-UNIF addendum (supplement to MS_Shift_R0_Closure §6)

**Obligation:** OBL-R0-UNIF — uniformity, over the compact parameter window, of the nondegeneracy
input feeding R0 (a.s. Morse–Smale structure / the 18×18 extended-jet system).

**Statement.** Let $W = [r_-, r_+] \times [b_-, b_+]$ be a compact parameter window with
$0 < r_- \le r_+$, and let $G_{18}(r, b; y)$ denote the 18×18 Gram of the full 2-jet functionals at
$(M_r, S_r, y)$ for $y$ in a compact station set $Y$ avoiding the analytic coincidence set
$\mathcal{N}$ (pin collisions / exact-degeneracy loci, a finite union of analytic varieties for the
BF kernel). Then $\lambda_{\min}(G_{18})$ is a continuous, strictly positive function on the compact
$(W \times Y) \setminus \mathcal{N}$-closure of the working region, hence **uniformly bounded
below**: $\lambda_{\min} \ge \lambda_*(W, Y) > 0$.

**Proof.** Entries of $G_{18}$ are finite compositions of $e^{-|\cdot|^2/2}$ and polynomials —
jointly analytic in $(r, b, y)$; $\lambda_{\min}$ of a symmetric matrix is continuous in its
entries; positivity at every point of the region holds because a vanishing $\lambda_{\min}$ would
give a nontrivial exact linear dependency among the 18 jet functionals, i.e. membership in
$\mathcal{N}$ (for the BF kernel the RKHS contains $p(x)e^{-|x|^2/2}$ for all polynomials $p$,
which separates any finite set of distinct-point jet functionals — the standard
universal-interpolation property of kernels with full spectral support). Continuity + compactness
gives the uniform bound. ∎ (Derived; standard.)

**Certified anchors.** $\lambda_{\min}(G_{18})$ at the two working stations:
$9.7267\times10^{-21}$ (r = 0.05, margin $10^{40.1}$) and $3.7283\times10^{-23}$ (r = 0.025,
margin $10^{37.7}$) — mp dps 60, eigsy, residual-verified (`c022_cert18_*.json`). The
constant-chase for an explicit closed-form $\lambda_*(W, Y)$ is possible but unnecessary for any
current consumer (R0 needs positivity, not a value); **OBL-R0-UNIF → discharged (bookkeeping
grade)**, with the closed-form chase noted as optional.

---

## Part B — Lemma LB restatement note (OBL-LB-RESTATE)

**Context.** The original Lemma LB (Lemma_LB_Package.md, C010-era) established a lower-bound
mechanism through the $p_\infty$ / probe-rigidity route calibrated at moderate $r$
($r \in \{0.45, 0.70\}$ instruments). Subsequent cycles established that the moderate-$r$ and
asymptotic regimes are qualitatively different (C014/C015 branch adjudications; the C021 second
rung), and C022's Lemma LB-ARCH now carries the asymptotic lower side by a different, certified
route (finite-dimensional certificates + explicit barrier + CM support).

**Restatement (supersede-in-scope).** Lemma LB is henceforth scoped as: *a moderate-$r$
phenomenological lower-bound mechanism*, valid as measured at its calibration rungs, feeding the
moderate-$r$ brackets of the constants table (c_eff(0.7) = 1.31, c_eff(0.4) ≈ 1.7). It is **not**
part of the asymptotic (Theorem A) chain of custody, which runs LB0 → R0 → Lemma LB-ARCH → FD.
No content of Lemma LB is overwritten; its claims stand within the restated scope. The registry
entry OBL-LB-RESTATE → **discharged** by this note.

**Falsifier hygiene.** Any future use of Lemma LB's $p_\infty$ numerics in an asymptotic-grade
derivation is a scope violation and should be flagged in review.
