# Zeta zeros, heat-flow folds and short-bar statistics

**Exploratory packet. Scientific effect: NONE. No status change. No register change.**
This is not a theorem-acceptance entry, a confirmation run or a held-out experiment.
It makes no claim of progress toward the Riemann Hypothesis (RH), and no bound on the
de Bruijn–Newman constant Λ beyond known results. None should be read into it.

- Author: Anthropic Claude (delegated AI work for the project owner, Dylan Roy).
- Nonauthor review from a different provider: slices A and D read, verdict **AMEND** (review
  5486843735); slice B (citations) read, verdict **AMEND** (review 5486879914); both applied
  here (§10). Delta readbacks at `0969b327`: slices A/D **PASS (scoped)** (review 5487707101);
  slice B **PASS (scoped)** (review 5487692150). Integrated into main as `3e589ae0` on
  12 October 2026 by a third OpenAI lane, distinct from both reviewers (§9). Automatic Codex
  review: **NOT_RUN** (quota). Slice C (independent numerical reproduction): different-provider
  replay of the packet's 59 commands **PERFORMED; record PASS (scoped)**. Additional fresh-seed
  evidence is **REPORTED; record applicability PASS (scoped)**, and its tables were
  **independently re-executed: PASS (scoped)** (§9).
  The Riemann–Siegel coefficients now cite Siegel 1932, the source actually checked, and agree
  with it as translated (**PASS (scoped)**); Edwards 1974 stays unverified background reading
  (owner decision). Slice C is not fully closed (§9). Dylan's personal reading: **PENDING**.
- Object: `ZETA-FOLD-EXPLORATION-20261012`, draft v2 as read back at `0969b327` and landed at
  `3e589ae0` (README blob `b367aecd`; v1 carried the same-provider passes, v2 adds the slice A,
  B and D amendments, §10), plus post-integration edits that neither readback covers: the §6.2
  coefficient wording (A/D delta note, §10.1), review-status currency (this block, §6.3, §9)
  and Markdown escapes that fix GitHub rendering in §1.3 and §4.
- Project sources read at main `8977619e` (the cited main files are unchanged from `03acb7be`
  through `7588716d`, the base recorded in `MANIFEST.json`, and through `3e589ae0`) and Math-
  `79cb314d`.

## In brief

An external note forwarded by the project owner on 12 October 2026 proposed three links
between the zeros of the Riemann zeta function and this project's short-bar law. This
packet checks each link, says what is classical, what is wrong, and what is useful to the
project, and keeps the scripts and results so that anyone can re-run them.

**What we found, in plain terms.**

1. **Folds.** When two zeros of the de Bruijn–Newman heat flow H_t merge at a nondegenerate
   double zero (H_zz ≠ 0), the antiderivative of H_t, on a fixed surrounding interval and for
   times close to the merger, gains or loses exactly one persistence bar. That bar follows the
   same cubic fold law as in this project (bar length ∝ gap³, with gap² = 8τ(1 + O(τ)) at
   time τ after the merger). This is ordinary fold geometry, not something special to zeta.
   We computed two actual collisions to about 40 digits. Both are at negative time
   (t ≈ −1.878 and t ≈ −0.385), and the fold law is accurate only close to the collision
   time. None of this says anything about RH.
2. **Theta functions.** The project's periodized Gaussian field is normalized by a Jacobi theta
   value, and zeta can be built from the same theta function. But zeta appears only after
   integrating over all torus sizes, while
   the project holds the torus size fixed. At the project's size the periodization is
   invisible locally (about 10⁻¹²⁵). This link is a footnote.
3. **Statistics.** This is the useful part, and it is about the project's own question, not
   about RH. Eigenvalues of random unitary matrices (CUE) repel quadratically at short range,
   whereas the zeros in the project's Gaussian models repel linearly. Under the hypotheses of
   the project's transfer theorem (Math- equation (D1), §2), the same cubic fold with
   quadratic repulsion gives short-bar density exponent 0 instead of −1/3. These two values
   are theoretical expectations under stated conditions, not calibrated answers: for fixed-N
   CUE the full bar law is the unproved target T-CUE (§6.1), and the Gaussian theorem's
   hypotheses were not checked for our Gaussian primitive. Zeta zeros are conjectured to
   behave like CUE (Montgomery; Odlyzko's numerics; our gap slopes ≈ 3), but pair correlation
   is neither a nearest-neighbour law nor a law of persistence pairs (Montgomery states the
   two-point and consecutive-spacing conjectures separately), so reaching bars needs further
   assumptions. (D1) is a theorem about random models. Among its hypotheses 1–4 it needs a
   finite negative mark moment under the limiting collision-mark measure (a finite ordinary
   marginal moment is not enough; `CLAIMS.md` 5.4). For the deterministic ζ none of these
   objects is defined or verified.
   **Numerically**, for the specified *Hardy-Z primitive* (the antiderivative of
   Z(t)/√(log(t/2π)) in the unfolded variable, endpoint-born bars dropped; §5.3), windows at
   heights 10⁴–10⁶ and Hill estimator, the observed finite-tail estimates are substantially
   above −1/3 and near 0. Sampling-only block-bootstrap intervals exclude −1/3 at the reported
   cutoffs. This is exploratory evidence about that finite-window statistic. It does not
   establish a limiting density exponent or a calibrated rejection accounting for all errors,
   and it says nothing about the ξ-weighted primitive or the de Bruijn–Newman barcode. The data
   were inspected before any prediction was frozen, so this is exploration, not confirmation.

**What is worth doing next** (§6): a theorem for fixed-size CUE matrices (T-CUE), and a
frozen, pre-registered comparison on fresh zeta windows. **What this packet does not claim or
pursue** is listed in §7.

| Link | Status | What it gives this project |
|---|---|---|
| Fold ↔ de Bruijn–Newman zero collisions | Exact, but only near the collision time. Generic fold geometry; the persistence wording is standard Cerf birth–death and vineyards. | A clean deterministic fold family whose time-transversality is forced by a PDE (Proposition 1). Nothing about RH. |
| Periodized-Gaussian normalizer ↔ Jacobi Θ ↔ ζ | Classical identities (Jacobi; Riemann 1859), exact. ζ appears only after a Mellin transform over the torus side L, a parameter the project holds fixed. | A footnote. At L = 24 the periodization changes local jets by about e^(−288). |
| GUE/CUE pair statistics ↔ contact law | For the specified Hardy-Z primitive, windows and estimator: finite-tail estimates substantially above −1/3 and near 0; sampling-only block-bootstrap intervals exclude −1/3 at the reported cutoffs. Exploratory evidence about that finite-window statistic, not a limiting exponent or a calibrated total-error rejection; not the ξ-weighted primitive or a de Bruijn–Newman barcode. GUE behaviour of zeta zeros is itself a conjecture, and pair correlation is not a nearest-neighbour or persistence-pair law. | A natural, non-Gaussian **altered pair-count tilt**: under the (D1) hypotheses, the same cubic fold with r² repulsion predicts density exponent 0, not −1/3 (a theorem target for CUE, T-CUE). |

**Files.**

| Path | Contents |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | Claim-by-claim check of the note (paraphrased), with verdicts, corrected statements and citation provenance. |
| `dbn/` | de Bruijn–Newman flow: three independent H_t implementations (`dbn_common.py`, `hflow.py`, `indep_ht.py`), collision search and fold checks, κ checks. |
| `theta/` | Theta, Mellin, Poisson and Epstein checks, and the project-normalizer checks. |
| `stats/` | Riemann–Siegel zeta zeros, Gaussian and CUE samples, barcodes, exponent estimates and review cross-checks. |
| `results/` | Small committed summaries (JSON, CSV, SVG, `.out` logs). `results/raw/` holds regenerable `.npz` samples and is not committed. |
| [`MANIFEST.json`](MANIFEST.json) | SHA-256 and size of every committed file, and SHA-256 of the raw samples from the exploratory run (written and checked by `make_manifest.py`). |
| `test_exact.py` | Exact rational checks (standard library only). |
| `test_numeric.py` | Numeric smoke checks (skipped unless numpy and mpmath are installed). |

## 1. Classical background

Every item below is classical. Items marked **[num]** were re-checked numerically here with
mpmath 1.4.1 (high precision, not interval-certified). Citation tags: **[snippet]** means the
bibliographic details were checked only against search-engine snippets (arXiv, journal and
encyclopedia pages could not be fetched from this environment: proxy 403 / DNS);
**[memory]** means recalled from memory and unchecked. Where a nonauthor reviewer checked a primary
passage or an author statement, or found only second-hand corroboration, one of three terms
from review 5486879914 replaces the tag (originals the reviewer did not inspect keep theirs): **primary passage checked**
[locator], written inline as **[primary, via review N]** (N = 5486843735, slices A/D, §10.1;
N = 5486879914, slice B, §10.2); **author statement checked; full proof not audited**;
**corroborated second-hand; original unchecked**. "Checked" means the passage supports the
statement as used, within its hypotheses; no proof was reconstructed and no packet calculation
was reproduced by these reviews. A summary is at the end of this section; locators and links
are in `CLAIMS.md` (Citation provenance).

### 1.1 ξ, Ξ and the critical line

ξ(s) = ½ s(s−1) π^(−s/2) Γ(s/2) ζ(s) is entire and satisfies ξ(s) = ξ(1−s). Define
Ξ(x) = ξ(½ + ix), which is real and even for real x. RH says every zero of Ξ is real. The
nontrivial zeros of ζ lie in 0 < Re s < 1; this follows from ζ(1+it) ≠ 0 (Hadamard;
de la Vallée Poussin, 1896) together with the functional equation [Hadamard 1896, opening
sections; Titchmarsh, 2nd ed., ch. III, §3.1 (zero-free line) and ch. II (continuation);
Riemann 1859, pp. 2–3; primary, via review 5486879914; de la Vallée Poussin's original not
inspected].

### 1.2 The de Bruijn–Newman flow

    Φ(u) = Σ_{n≥1} (2π² n⁴ e^{9u} − 3π n² e^{5u}) exp(−π n² e^{4u}),
    H_t(z) = ∫_0^∞ e^{t u²} Φ(u) cos(z u) du.

These are the conventions of Rodgers–Tao (eqs. (1)–(4) [primary, via review 5486879914]) and
Polymath15 [snippet].

- **Normalization.** H_0(z) = Ξ(z/2)/8, so the zero ρ = ½ + iγ corresponds to z = 2γ.
  **[num]** Three independent implementations agree (§5.1): relative difference ≤ 7·10⁻⁴¹
  at z ∈ {0, 10, 28.27, 40, 75, 150} (Gauss–Legendre), ≤ 2·10⁻⁴⁴ for z ≤ 250 (trapezoid rule;
  `results/dbn/smoke_hflow.out`) and ≤ 1.3·10⁻⁵⁷ at z ∈ {0, 10, 28.27, 75} (tanh-sinh, 60 digits;
  `results/dbn/indep_ht_xi.out`). The zeros of H_0 match 2·Im ρ_k for k = 1…41 to 30 digits.
- **Kernel.** Φ is even. It is positive for u ≥ 0 termwise, because each term equals
  πn²e^{5u}(2πn²e^{4u} − 3) e^{−πn²e^{4u}} and 2π > 3; hence it is positive everywhere by
  evenness. With ψ = (Θ−1)/2 and x = e^{4u}, Φ(u) = e^u [2x² ψ''(x) + 3x ψ'(x)], so Φ is a
  derivative of the Jacobi theta function, and its evenness is the theta transformation.
  **[num]** Both checked to 20 digits.
- **Backward heat equation.** ∂_t H = −∂_z² H, by differentiating under the integral.
  **[num]** Finite-difference residual ≤ 4·10⁻²⁷ at three points, against terms of size
  10⁻⁴ to 10⁻¹⁰.
- **Zero dynamics.** At any simple zero, H(t, z_j(t)) = 0 and the heat equation give the exact
  velocity ż_j = −H_t/H_z = H_zz/H_z. In the all-real regime (t > Λ: all zeros real and
  simple), the Hadamard product turns this into dz_j/dt = 2 Σ_{k≠j} 1/(z_j − z_k)
  [Csordas–Smith–Varga 1994, whose Lemma 2.4, p. 115, works in this regime, primary, via review
  5486879914; Rodgers–Tao pp. 4–5, eq. (5), and Polymath15 §9, which separates the general
  simple-root velocity from this regime; primary, via review 5486843735].
  That sum is not absolutely convergent and must be summed symmetrically, for example over
  positive zeros as ż_j = 2[1/(2z_j) + Σ_{k≠j} 2z_j/(z_j² − z_k²)]. At t = 0, at a simple
  zero z_j, a sum over real zeros only (counted with multiplicity) is valid under RH (all zeros
  of H_0 real); unconditionally the nonreal zeros would also enter. Simplicity of the zeros is
  a separate conjecture, not part of RH. In the all-real regime zeros **repel as t
  increases**. As t decreases, neighbours attract, can merge into a real double zero, and then
  become a complex-conjugate pair (Proposition 1, locally).
  **[num]** At t = 0 the exact velocities H_zz/H_z are −0.5796306, −0.7549955 and −0.4654455
  for zeros #1, #2 and #5 (`dbn/zero_velocity.py`), matching finite differences. The
  symmetric real-zero sum over the first 300 zeros plus a tail estimate gives −0.579717,
  −0.755108 and −0.465554, agreeing to the truncation level of about 10⁻⁴. This is a
  consistency and truncation check of the all-real formula, not evidence that omitted
  nonreal contributions vanish.
- **The constant Λ.** There is a Λ such that H_t has only real zeros exactly when t ≥ Λ.
  De Bruijn's strip contraction (1950, Theorem 13, p. 205) gives real zeros for t ≥ 1/2 and
  keeps them real as t increases; Newman (1976, Theorem 3, whose exponential parameter has the
  opposite sign) shows that the threshold is finite [primary, via review 5486879914]. RH ⇔
  Λ ≤ 0. Rodgers–Tao (2020, Theorem 1) proved Λ ≥ 0, so RH ⇔ Λ = 0 [primary, via review
  5486879914].
  - Upper bounds: Λ ≤ 0.22 [Polymath15 2019, Theorem 1.1]; Λ ≤ 0.2 [Platt–Trudgian 2021,
    Corollary 2]; [primary, via review 5486879914].
  - Lower bounds before Rodgers–Tao came from Lehmer pairs [CSV 1994, Theorem 1;
    Csordas–Odlyzko–Smith–Varga 1993, Theorem 2.2; Odlyzko 2000; primary, via review
    5486879914]. Saouter–Gourdon–Demichel 2011 and the 1988–1992 sequence of bounds are
    corroborated second-hand only (Rodgers–Tao's table; CSV p. 108, eq. (1.5)); originals
    unchecked.
  - Λ < 1/2, and at each fixed t > 0 all but finitely many zeros of H_t are real and simple
    [Ki–Kim–Lee 2009; author statement checked; full proof not audited].
  - De Bruijn's strip lemma: if all zeros of H_{t0} satisfy |Im z| ≤ y0, then Λ ≤ t0 + y0²/2
    [de Bruijn 1950, Theorem 13, in this normalization; primary, via review 5486879914].
  - Survey: Newman–Wu 2020 [a survey, not an additional original proof; primary, via review
    5486879914].
  - The time variable rescales by 4 between the z and Ξ(x) conventions.
  - **What the cited results use.** De Bruijn's strip contraction and Newman's finite threshold
    hold for kernels meeting their stated symmetry and decay conditions. Ki's 2022 report
    presents the Ki–Kim–Lee theorems together with extensions to broader Dirichlet-series and
    functional-equation settings [author statement checked; full proof not audited]. Rodgers–Tao
    argue by contradiction: if Λ < 0, the zero dynamics would put the zeros of H_0 into local
    equilibrium, contradicting known statistics of the zeros [introduction; primary, via review
    5486879914]. This packet does not classify which of these inputs are arithmetic. The
    Rodgers–Tao energy argument is a technique for proving the lower bound; it does not forbid
    any particular arrangement of zeros for t > 0.

### 1.3 Theta, Mellin and Epstein

Let Θ(u) = Σ_{n∈ℤ} e^{−π u n²}.

- **Theta transformation.** Θ(u) = u^{−1/2} Θ(1/u) is classical (Jacobi; Poisson summation)
  [Jacobi: corroborated second-hand, Riemann credits him; original unchecked]. Its use to prove
  ξ(s) = ξ(1−s) is Riemann 1859, the second proof [pp. 2–3, Clay translation; primary, via
  review 5486879914]. Mumford's *Tata Lectures on Theta I* is an expository source [memory].
- **Mellin formula.** π^{−s/2} Γ(s/2) ζ(s) = ½ ∫_0^∞ (Θ(u) − 1) u^{s/2−1} du for Re s > 1, an
  ordinary improper integral (Riemann writes it with ψ = (Θ−1)/2) [primary, via review
  5486879914]. **[num]** Checked at s = 2, 3, 4.5 and 2+3i to 18 digits. As u → 0 the
  integrand behaves like u^{(s−3)/2}, so the integral diverges for Re s ≤ 1; continuation needs
  Riemann's split at u = 1.
- **Higher dimensions.** For d ≥ 2 and Re w > d/2,
  ∫_0^∞ (Θ^d − 1) u^{w−1} du = π^{−w} Γ(w) Z_d(w), where Z_d(w) = Σ'\_{n∈ℤ^d} |n|^{−2w} is the
  Epstein zeta function of ℤ^d, with Λ_d(w) = Λ_d(d/2 − w). Two cases:
  Z_2 = 4 ζ(w) L(w, χ_{−4}) and Z_4 = 8 (1 − 4^{1−w}) ζ(w) ζ(w−1).
- **Theta structure plus a symmetry does not place zeros on the symmetry line.** Z_4 has zeros
  at ½ + iγ and 3/2 + iγ, off its symmetry line Re w = 1. Z_4 does have an Euler product: it is
  an imprimitive, shifted product, and it shows that, for such a product, a theta functional
  equation is compatible with zeros off the symmetry line. Separately, Epstein zeta functions of
  positive-definite integral **binary** quadratic forms with class number > 1 have zeros in
  Re s > 1 [Davenport–Heilbronn 1936; corroborated second-hand by Titchmarsh §10.27 for binary
  forms only; original unchecked]; that result does not apply directly to the
  project's ℤ^d forms. **[num]** The factorizations, the functional equations and the
  off-line zeros of Z_4 at γ_1 were checked.

### 1.4 Pair correlation

- **Montgomery's theorem (assuming RH)** gives pair correlation only for test functions whose
  Fourier support lies in (−1, 1); full pair correlation and the stronger spacing and
  higher-correlation statements are conjectural [Montgomery 1973, pp. 181–185; primary, via
  review 5486879914].
- **The conjecture.** R_2(r) = 1 − (sin πr / πr)² = (π²/3) r² − (2π⁴/45) r⁴ + … is the
  unit-density bulk sine-process limit (GUE/CUE as N → ∞), not an exact finite-N GUE formula.
  That ζ's zeros follow it is Montgomery's pair-correlation conjecture. Rudnick–Sarnak 1996,
  Theorem 1.2, proves n-level correlations under RH with restricted support (Σ|ξ_j| < 2 for ζ);
  not every result in that paper assumes RH [primary, via review 5486879914].
- **Evidence.** Odlyzko's computations support it numerically; that is evidence, not a theorem
  about the full statistics [Odlyzko 1987, primary, via review 5486879914; later computations,
  snippet].
- **Small r.** The r → 0 law is not proved. The nearest-neighbour spacing density of the sine
  process has the same leading term (π²/3) s² but is a different observable [Odlyzko 1987,
  eq. (2.23) and §6; primary, via review 5486879914]. The coefficient 32/π² belongs to the
  Wigner surmise, not to the exact bulk law.
- **Three different objects.** Pair correlation is a two-point law. It is not a
  nearest-neighbour (consecutive-spacing) law, and Montgomery states the two conjectures
  separately [Montgomery 1973, pp. 184–185; primary, via review 5486843735]. Neither is a law
  of persistence pairs, which the elder rule selects (§3). The sine process has the stated
  leading nearest-gap behaviour, but carrying it over to ζ, and then to bars, needs further
  assumptions.

**Citation provenance** (locators and links: `CLAIMS.md`, Citation provenance).

- *Primary passage checked by a nonauthor reviewer, not by the author.* Review 5486843735:
  Rodgers–Tao eqs. (1)–(5), pp. 4–5; Montgomery 1973, pp. 184–185; Polymath15 Theorem 1.5
  (pp. 9–10) and §9. Review 5486879914: Riemann 1859 (Clay translation, pp. 2–3); Hadamard
  1896; Titchmarsh, 2nd ed., ch. II and ch. III §3.1; de Bruijn 1950, Theorem 13; Newman 1976,
  Theorem 3; Csordas–Smith–Varga 1994, Theorem 1, Corollary 2, Lemma 2.4;
  Csordas–Odlyzko–Smith–Varga 1993, Theorem 2.2; Odlyzko 2000; Rodgers–Tao eqs. (1)–(4),
  Theorem 1 and introduction; Polymath15 Theorem 1.1 and its deformed coefficients;
  Platt–Trudgian 2021, Theorem 1 and Corollary 2; Newman–Wu 2020 (a survey); Montgomery 1973,
  pp. 181–185; Rudnick–Sarnak 1996, Theorem 1.2; Odlyzko 1987, eq. (2.23) and §6;
  Lagarias–Montague, eqs. (1.2), (1.6), (1.9) and Theorem 2.1; Cohen-Steiner–Edelsbrunner–Morozov
  2006, §§2, 4; Hughes–Keating–O'Connell 2000, Theorem 1.2, Remark 2.2 and Conjecture 1.3;
  Keating–Snaith 2000, eqs. (8) and (92); Bogomolny–Bohigas–Leboeuf–Monastra 2006, eqs.
  (19)–(20); Radziwiłł–Soundararajan 2017, Theorem 1; Mathlib's documentation (local version and
  proofs not checked).
- *Author statement checked; full proof not audited:* Ki–Kim–Lee 2009.
- *Corroborated second-hand; original unchecked:* Saouter–Gourdon–Demichel 2011 (Rodgers–Tao's
  table); Csordas–Norfolk–Varga 1988, te Riele 1991, Csordas–Ruttan–Varga 1991 and
  Norfolk–Ruttan–Varga 1992 (CSV 1994, p. 108, eq. (1.5); identification with zeros 34/35
  unconfirmed); Davenport–Heilbronn 1936 (Titchmarsh §10.27, binary forms only); Jacobi
  (credited by Riemann); the hypotheses of Hejhal's central limit theorem (as quoted by HKO).
- *Search snippets only:* Polymath15's kernel conventions; Odlyzko's later computations.
- *From memory, unchecked:* de la Vallée Poussin 1896, Sturm's zero-number property for the heat
  equation, Mumford, Cerf 1970 (exact passage not located), Gonek 1989, Hejhal 1989, Selberg's
  original, and the CUE log|Z_U| variance of order ½ log N (§6.1). Edwards 1974 is identified
  bibliographically only and stays unverified background reading: by owner decision the
  Riemann–Siegel coefficients cite Siegel 1932, the source actually checked (in English
  translation, §9); they were also validated numerically against `mpmath.siegelz`.

Only the items a reviewer reports as checked carry the stronger terms; every other item stays
unverified (§9).

## 2. What the project states, and what it does not

Locations are in main `8977619e` unless marked Math- (Math- links are pinned to `79cb314d`).

- **Fold law.**
  - The normal form x³/3 − h²x has gap = separation³/6
    ([`experiments/universality/fold_controls.py`](../universality/fold_controls.py) lines 28–31).
  - In the project's convention ℓ = k r³ with k = |F'''|/12 at contact
    ([`fold_structural_universality/PROOF.md`](../universality/fold_structural_universality/PROOF.md)
    lines 191–193 and 244–248).
- **Contact ledger.**
  - The pair intensity is ∝ r: (d−1) + 3 − (d+3) + 2 = 1, and no exponent can be inferred
    while omitting the contact-density or determinant-product orders (`PROOF.md` lines 390–404).
  - The cubic fold alone supplies the gap order, not the radial power (`fold_controls.py`
    lines 3–4). The exact toy controls give density powers −2/3, −1/3 and 0 for radial
    powers 0, 1 and 2; the note's proposed test is literally the case `cubic_2`
    (`results/theta/fold_controls_rerun.out` is the stdout of
    `python3 -B experiments/universality/fold_controls.py` run from the repository root).
- **Theorems.** These are distinct objects, which the note treats as one theorem.
  - The conditional structural theorem: COUNT, CONTACT, SELECT, ENVELOPE and FAR, d ≥ 1,
    with a positive coefficient required (`PROOF.md` lines 56–70).
  - The periodized-Gaussian adapter: fixed d ≥ 2 and L, existential constants, and no joint
    unbounded-d, variable-L, cutoff or model-family uniformity
    ([`periodized_gaussian_h0/PROOF.md`](../universality/periodized_gaussian_h0/PROOF.md)
    lines 54–66 and 80–83).
  - The 1D theorems: Math- `frontiers/d1_third_order_law_20261001` and
    `frontiers/d1_line_20261004`, ν(ℓ) = C₀ ℓ^{−1/3} + C₁ ℓ^{1/4} + ….
- **Geometry versus measure, already in the repositories.** In
  [Math- `reviews/collision_mechanism_20260925/NOTE.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/collision_mechanism_20260925/NOTE.md)
  Section D:
  - **Equation (D1)** is a marked collision-to-lifetime transfer theorem. With candidate-pair
    intensity r^α (α is the pair or gap exponent) and ℓ = h(r, z) ≈ κ r^m, the density exponent
    is β − 1 with β = (α+1)/m, under its hypotheses 1–4 (envelope, integrable negative mark
    moment, ratio and derivative limits, uniform comparison). In this packet β always has this
    Math- meaning, and κ in this sense is the fold coefficient, written k in Proposition 1.
  - **Subsection §D1** is a counterexample: cubic splitting alone need not give a density
    coefficient.
  - **Subsection §D2** gives an exact small-mark phase diagram for a mark density ∝ κ^q:
    exponent β − 1 if q + 1 > β, an extra log(1/ℓ) if q + 1 = β, and exponent q if q + 1 < β.
  - **Equation (D2)**, the cumulative statement as written in NOTE.md, was marked
    AMEND_REQUIRED by the PR25 nonauthor review
    ([`reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md)
    lines 60 and 184–197). That review accepted equation (D1), the subsection §D1
    counterexample and the subsection §D2 phase diagram. The reviewed replacement for the
    cumulative statement is
    [`CUMULATIVE_TRANSFER_CORRECTION.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md),
    which restores the ratio hypothesis h/(κ r^m) → 1. This packet cites that correction,
    not the unamended (D2), whenever it needs the cumulative fallback.
  - The amplitude counterexample at `fold_structural_universality/PROOF.md` lines 658–683
    shows that the small-amplitude boundary is not controlled by covariance or positive
    moments.
- **No prior zeta material.** Neither repository has prior material on ζ, Epstein zetas,
  GUE/CUE, de Bruijn–Newman or Lehmer pairs. The one exception is `history/2025/`, which its
  own README calls obsolete and wrong. **Do not cite it.**
- **Related open proposal** (unclaimed when this packet was written).
  [main#307 comment 6116121839](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116121839)
  (an OpenAI lane, 12 October 2026) proposes polynomial-first deterministic heat-flow fold
  controls with finite-interval H0 within the universality program (no path named). This
  packet does not take that scope; it touches only `experiments/zeta_fold_exploration/**`.
  The triple-collision example (§3 Remark 1), the gap-versus-bar countercontrol (§5.2) and
  the information-loss example (§7) are taken from that comment and re-checked by hand here
  in exact rational arithmetic. The density-transfer caveat (`CLAIMS.md` 5.4) and the
  Lagarias–Montague pointer (§4 item 3) are also taken from it; the latter has since been
  checked against the primary source by the slice-B reviewer (review 5486879914). No
  reproducible controls are built.

## 3. Proposition 1: the fold of a backward-heat solution (elementary)

**Proposition 1.** Assume:

- H(t, z) is real-valued and real-analytic near (t₀, z₀) ∈ ℝ², and solves ∂_t H = −∂_z² H.
- H(t₀, z₀) = ∂_z H(t₀, z₀) = 0.
- A := ½ ∂_z² H(t₀, z₀) ≠ 0.

Set B := ∂_z³ H/6 and C := ∂_z⁴ H/24, both at (t₀, z₀), and τ := t − t₀. Then there are
δ, ε > 0 such that, on W = (z₀ − δ, z₀ + δ):

**(a) Zeros.**
- For 0 < τ < ε, H(t, ·) has exactly two zeros z₋(τ) < z₊(τ) in W, and both are simple.
- For −ε < τ < 0 it has no zero in W. Its two zeros nearest z₀ are the non-real pair
  z₀ + O(|τ|) ± i √(2|τ|) (1 + O(|τ|)).
- At τ = 0, z₀ is the only zero in W, and it is double.

**(b) Gap and midpoint.** For 0 < τ < ε:
- r := z₊ − z₋ satisfies r² = 8τ g(τ), where g is real-analytic, g(0) = 1 and
  g'(0) = κ_r := 4C/A − 2B²/A².
- The midpoint is z₀ + 2(B/A) τ + O(τ²).

**Barcode convention** (used throughout the packet). Barcodes are degree-0 persistence of the
superlevel sets {F ≥ λ} of a continuous function on a compact interval with finitely many
critical points, with coefficients in a field (any field gives the same degree-0 barcode). A
component born at a local-maximum value b that merges into an elder component (one born at a
higher value) at λ = d is the finite bar (d, b]: it exists for λ ∈ (d, b], and its length is
ℓ = b − d. The global maximum gives the essential bar.

**(c) Persistence.** Assume also that H is defined, real-analytic and a solution on
(t₀ − ε₁, t₀ + ε₁) × U for an open interval U ⊃ W̄ (for H_t, U = ℝ). Fix a compact interval I
with W̄ ⊂ I ⊂ U and a base point z* ∈ I, and let F(t, z) = ∫_{z*}^{z} H(t, u) du on I. Since
H(t, ·) ≢ 0 is analytic, F(t, ·) has finitely many critical points on I, so its barcode is
defined. Then for 0 < τ < ε' (some ε' ≤ ε):
- The two critical points z± of F(t, ·) form a pair in the superlevel H0 barcode of
  F(t, ·)|_I. The one that is a local maximum is born at its value and dies at the value of
  the other.
- The bar length is

      ℓ(τ) = |∫_{z₋}^{z₊} H(t,u) du|
           = (|A|/6) r³ (1 + κ_ℓ τ + O(τ²))
           = (8√2/3) |A| τ^{3/2} (1 + κ_t τ + O(τ²)),

  with κ_ℓ = 6B²/A² − 48C/(5A) and κ_t = κ_ℓ + (3/2) κ_r = 3B²/A² − 18C/(5A).
- For τ < 0, F(t, ·) has no critical point in W.

Part (c) gives the local bar for small τ on a fixed surrounding interval. It says nothing
about arbitrary close pairs, or about the pairing at t = 0 (§5.2).

**Proof.** For (a) and (b), replacing H by −H preserves the equation and the zeros, so assume
A > 0. For (c), replacing H by −H alone would swap superlevel and sublevel persistence.
Instead use the simultaneous reflection H̃(t, z) = −H(t, 2z₀ − z), F̃(t, z) = F(t, 2z₀ − z)
(base point 2z₀ − z*, interval 2z₀ − I). Both sides of the equation change sign, so H̃ is a
solution; F̃ = ∫_{2z₀−z*}^{z} H̃; and the superlevel sets of F̃ are the reflections of those of
F, so the barcodes agree. The reflection maps (A, B, C) to (−A, B, −C), which leaves r, ℓ,
κ_r, κ_ℓ and κ_t unchanged and reflects the midpoint drift. So A > 0 may be assumed in (c)
too.

*(a), via a minimum argument.*
1. Choose δ and ε₀ so that |∂_z² H − 2A| < A on [t₀ − ε₀, t₀ + ε₀] × W̄. Then each H(t, ·) is
   strictly convex on W, so it has at most two zeros there, counted with multiplicity.
2. By the implicit function theorem, ∂_z H(t, ·) has a unique zero c(t) ∈ W for t near t₀,
   with c analytic and c(t₀) = z₀. The minimum of H(t, ·) on W is m(t) = H(t, c(t)).
3. m'(t) = ∂_t H + ∂_z H · c' = ∂_t H(t, c(t)) = −∂_z² H(t, c(t)). Hence m(t₀) = 0 and
   m'(t₀) = −2A < 0.
4. Convexity gives H(t₀, z₀ ± δ) ≥ (A/2) δ² > 0, and this stays positive for t near t₀.
5. So for small ε: m < 0 when τ > 0 (two simple zeros), m > 0 when τ < 0 (no real zero in W),
   and m = 0 when τ = 0 (a double zero).
6. For τ < 0, Hurwitz's theorem applied to the complexification gives exactly two zeros in a
   small disc around z₀. Neither is real, so they form a conjugate pair. Their asymptotics
   come from the scaling below with τ = −σ².

*(b), via scaling.*
1. On solutions, ∂_t^k ∂_z^j H = (−1)^k ∂_z^{j+2k} H. Regrouping the absolutely convergent
   Taylor series at (t₀, z₀) therefore gives H(t₀+τ, z₀+x) = Σ_{n≥2} c_n v_n(x, τ), where
   c_n = ∂_z^n H/n! and v_n(x, τ) = Σ_{k ≤ n/2} n!/(k!(n−2k)!) (−τ)^k x^{n−2k} are the backward
   heat polynomials. For example v₂ = x² − 2τ, v₃ = x³ − 6xτ and v₄ = x⁴ − 12x²τ + 12τ².
2. Since v_n(σy, σ²) = σⁿ v_n(y, 1), the function
   h(y, σ) := σ^{−2} H(t₀ + σ², z₀ + σy) = Σ c_n σ^{n−2} v_n(y, 1) is real-analytic near
   {σ = 0} × [−K, K].
3. It satisfies h(y, 0) = A(y² − 2) and the exact symmetry h(−y, −σ) = h(y, σ).
4. The implicit function theorem at the simple roots ±√2 gives analytic branches y±(σ), and
   z± = z₀ + σ y±(σ). By (a) these are the only zeros in W. The symmetry gives
   y₋(σ) = −y₊(−σ).
5. So r/σ = y₊(σ) + y₊(−σ) is even in σ. Likewise σ(y₊(σ) + y₋(σ))/2 is even in σ. Both are
   therefore analytic in τ = σ².
6. Substituting y± = ±√2 + y₁σ + … into
   h = A(y²−2) + σB(y³−6y) + σ²C(y⁴−12y²+12) + O(σ³) gives y₁ = 2B/A for both branches, and
   the stated κ_r.

*(c), the elder argument.*
1. Here (A > 0) z₋ is a local maximum of F with value M, and z₊ a local minimum with value
   m₀. F increases on [z₀ − δ, z₋], decreases on [z₋, z₊] and increases on [z₊, z₀ + δ].
2. Since ∂_z² H ≥ A on W, ∫_{z₊}^{z₀+δ} H ≥ (A/6)(z₀ + δ − z₊)³ ≥ Aδ³/48 for small τ.
   Meanwhile M − m₀ = −∫_{z₋}^{z₊} H = O(τ^{3/2}). Hence F(z₀ + δ) > M, and symmetrically
   F(z₀ − δ) < m₀.
3. For λ ∈ (m₀, M], the component of {F|_I ≥ λ} through z₋ is an interval inside W containing no
   other critical point.
4. At λ = m₀ it merges with the component through z₊. That component contains z₀ + δ, where
   F > M, so it was born earlier. By the elder rule the bar of z₋ is (m₀, M].

*The value of ℓ.*
1. Write ℓ = σ³ |J(σ)| with J(σ) = ∫_{y₋(σ)}^{y₊(σ)} h(y, σ) dy. The symmetry gives
   J(−σ) = J(σ), so ℓ/τ^{3/2} is analytic in τ.
2. Also J(0) = ∫_{−√2}^{√2} A(y² − 2) dy = −(8√2/3) A.
3. For the τ-coefficient, use the exact identity for consecutive zeros a < c,
   ∫_a^c H = −½ ∫_a^c (u − a)(c − u) ∂_u² H du
   ([`PERSISTENCE_BRIDGE.md`](../../docs/research-translation/20260930/PERSISTENCE_BRIDGE.md)
   lines 94–99).
4. Then ∂_u² H = 2A + 6Bx + C(12x² − 24τ) + O(τ^{3/2}), with x = u − z₀ = m + s and
   m = 2(B/A)τ + O(τ²). The needed moments are ∫(r²/4 − s²) ds = r³/6, ∫(r²/4 − s²) s ds = 0
   and ∫(r²/4 − s²) s² ds = r⁵/120.
5. Together with r² = 8τ(1 + O(τ)) these give κ_ℓ. Then κ_t follows from
   r³ = (8τ)^{3/2} g^{3/2}. ∎

**Checks.**
- `test_exact.py` derives κ_r, κ_ℓ, κ_t and the midpoint drift exactly, as power series in
  σ = √τ with coefficients in ℚ(√2), for rational A, B, C (A > 0 and A < 0, with extra v₅
  and v₆ terms), and checks the three kernel moments and J(0) exactly.
- `dbn/kappa_check.py` integrates a truncated heat-polynomial solution with
  (A, B, C, D) = (1, 0.3, 0.2, −0.15). At τ = 10⁻⁴:

  | Quantity | Measured | Predicted |
  |---|---|---|
  | κ_r | 0.61974 | 0.62 |
  | κ_ℓ | −1.37928 | −1.38 |
  | κ_t | −0.44979 | −0.45 |
  | midpoint / τ | 0.60024 | 2B/A = 0.6 |

  The differences are O(τ).
- Two independent review checks agree: `dbn/kappa_indep_degree6.py` (degree 6, A < 0,
  nonzero v₅ and v₆; agreement to about 10⁻⁵ at τ = 10⁻⁷, and Im z/√(2|τ|) → 1) and
  `dbn/kappa_indep_random.py` (random heat polynomials v₂–v₇ with Richardson extrapolation;
  agreement to 10 digits).

**Remarks.**
1. *Generic geometry.* ℓ = (|F'''|/12) r³ (1 + O(r²)) holds for every nondegenerate fold.
   Here F''' = 2A, so k = |A|/6. Neither ζ nor the heat equation enters this relation.
   Proposition 1 needs H_zz ≠ 0 (A ≠ 0). A degenerate (triple) collision such as
   H = v₃ = x³ − 6xτ follows a different law, with critical-value gap r⁴/4 between adjacent
   zeros r apart, and is not covered (example taken from main#307 comment 6116121839, §2).
2. *What the heat equation adds.* First, transversality is automatic: ∂_t H = −2A ≠ 0 at the
   double zero, so the discriminant is crossed transversally and the pair always opens in
   forward time. Second, the opening law r² = 8τ(1 + O(τ)), i.e. r² ~ 8τ, has no free
   constant in its leading term. The correction κ_r is generally nonzero, so r² = 8τ exactly
   is false in general.
3. *Consequence for H_t.* Folds of H_t occur only at t₀ ≤ Λ, since for t > Λ all zeros are
   simple [CSV 1994, Corollary 2, p. 114; primary, via review 5486879914]. Conversely, if
   Λ > 0 then H_Λ has a multiple real zero at finite height. That is a deduction from
   Polymath15 Theorem 1.5(i), not from the fold, and the zero need not be a nondegenerate double
   zero (`CLAIMS.md` 6.2). It is not an RH result.
   **Under RH, fold events occur only at t ≤ 0** (t = 0 only if Ξ has a multiple zero).
   Heuristically (the Lehmer-pair mechanism), collision times approach 0 from below as height
   grows; this is a heuristic, not a theorem of Rodgers–Tao or anyone else. So under RH a
   vineyard of F_t = ∫ H_t over t > 0 has no births or deaths inside a window; the only
   changes are pairing switches and zeros crossing the window ends.
4. *Persistence language is not new.* Describing this as one birth in a vineyard is standard
   Cerf theory [Cerf 1970, memory, exact passage not located; Cohen-Steiner–Edelsbrunner–Morozov
   2006, §§2, 4, stability and vineyards, primary, via review 5486879914, which does not prove
   this packet's fold coefficients].

## 4. Elementary restatements in the project's variables (novelty not claimed; no literature search)

These are one-line consequences of classical facts, written in the project's variables. They
were checked numerically. No literature search was done, and no novelty is claimed.

1. **ζ lives along the scale direction.** Let N_F(L, d) = Σ_{j∈ℤ^d} e^{−2π²|j|²/L²} be the
   project's Fourier normalizer, which equals Θ(2π/L²)^d. For Re s > d,

       ∫_0^∞ (N_F(L, d) − 1) L^{−s−1} dL = ½ (2π²)^{−s/2} Γ(s/2) Z_d(s/2).

   For d = 1 this is (2π²)^{−s/2} Γ(s/2) ζ(s) (Re s > 1). This is Riemann's Mellin formula after
   the substitution u = 2π/L², with nothing added. **[num]** Checked for d = 1 at s = 3 and
   2+i, d = 2 at s = 3, and d = 4 at s = 6. ζ appears only after integrating over **all torus
   sizes**, a direction the project disclaims.
2. **The periodization is invisible locally.** K_L(z) = Π_i ϑ₃(πz_i/L, q)/ϑ₃(0, q) with
   q = e^{−2π²/L²}. At L = 24 it differs from the planar squared exponential by
   O(e^{−L²/2} poly(L)), about 10⁻¹²⁵ in the jets at a point; K'', K⁗ and K⁽⁶⁾ at 0 are −1, 3
   and −15 to working precision. The project uses Poisson/Jacobi only to make every Fourier
   weight positive. The modular symmetry u ↦ 1/u plays no role.
3. **The barcode of ∫H depends on how H is normalized, not only on its zeros.** Multiplying H
   by a positive weight w keeps the critical points but changes bar lengths, and can change
   pairings. On Hardy-Z windows **[num]**:
   - with the Ξ weight instead of Hardy normalization, 80/113 pairs ([600, 900]) and 82/106
     ([5000, 5200]) are shared;
   - with Ξ's actual sign, 0 pairs are shared, because superlevel and sublevel roles swap;
   - with a mild weight e^{0.01t}, 112/113 and 104/106 are shared.

   **Weighted short bars need C^{1,1}.** Take a close pair z₋ < z₊ near a nondegenerate fold
   at z₀ (Proposition 1), with gap r and midpoint m, and let ℓ_w = |∫_{z₋}^{z₊} w H|. Write
   H = (u − z₋)(u − z₊) q(u). Near the fold q is bounded away from 0 and q' is bounded: this
   is the controlled asymmetry, which must be assumed for close pairs away from a fold. Let
   w > 0 have bounded w'' (w ∈ C^{1,1}) on a fixed neighbourhood of z₀. Then

       ℓ_w = w(x₀) ℓ (1 + O(r²)),   with x₀ = z₀ or x₀ = m.

   With s = u − m and kernel r²/4 − s², the linear part of w contributes the relative term
   (w'/w)(z₀)[(m − z₀) + ∫(r²/4 − s²) q s ds / ∫(r²/4 − s²) q ds]. Both pieces are O(r²): the
   midpoint offset is 2(B/A)τ + O(τ²), and the asymmetry of q makes the second piece
   (B/A) r²/20 + O(r⁴), not 0. The relative Taylor remainder is at most
   sup|w''| (r/2 + |m − z₀|)²/(2w(z₀)).
   For smooth w the ratio is even in √τ, and ℓ_w/(w(z₀)ℓ) = 1 + \[(12/5)(B/A)(w'/w) +
   (1/5)(w''/w)\](z₀) τ + O(τ²) (exact series in `test_exact.py`).

   For a merely C¹ weight only 1 + o(1) holds (in fact 1 + o(r), from the modulus of
   continuity of w'), not O(r²). *Counterexample* (review 5486843735; `test_exact.py`): for
   H = x² − 2t, a = √(2t), r = 2a and the fixed positive C¹ weight w = 1 + |x|^{1+η} with
   0 < η < 1, ℓ = 4a³/3 and ℓ_w/ℓ = 1 + 3a^{1+η}/((2+η)(4+η)). For η = ½ this is
   1 + (4/15)a^{3/2}. On a fixed interval containing the window, both are actual bars for
   small a.

   A pointwise formula is not a transfer statement. Keeping the exponent, or a coefficient
   reweighted by w^{−β} (β = (α+1)/m: w^{−1} for α = 2, w^{−2/3} for α = 1), across a
   height-dependent ensemble also needs weights bounded above and below by positive constants
   uniformly, C^{1,1} and asymmetry constants uniform over the bars, and the marked-measure
   hypotheses of (D1) for the reweighted marks.

   Prior work on the same antiderivative (up to scaling and an additive constant):
   Lagarias–Montague, arXiv:1106.4348, eqs. (1.2), (1.6), (1.9) and Theorem 2.1 [primary, via
   review 5486879914; pointer from main#307 comment 6116121839]. In their notation
   H_t(z) = Ξ_{t/4}(z/2)/8 and, normalized by F_t(0) = 0, F_t(z) = Ξ^{(−1)}_{t/4}(z/2)/4. Their
   Theorem 2.1 gives, for each real parameter, finitely many real and infinitely many nonreal
   zeros of that primitive (for parameter ≤ 0 its only real zero is 0). These are zeros of the
   primitive, not the zeros of its derivative that give the critical points of F used here, and
   adding a constant changes the zero set. No persistence or novelty conclusion follows.
4. **The lobe area as a weighted integral of ∂_tH.** For consecutive zeros a < c of H_t,
   ∫_a^c H_t = ½ ∫_a^c (x − a)(c − x) ∂_t H_t dx. This is the integration-by-parts identity
   used in the proof of Proposition 1, combined with the heat equation. **[num]** Both sides
   equal −6.31861166707933·10⁻⁴ between 2γ₁ and 2γ₂ (`dbn/lobe_identity.py`,
   `theta/project_theta_checks.py`).

## 5. Numerical results (exploratory)

Everything here is exploratory. The data were inspected before any prediction was frozen,
so nothing below is a confirmation. Nothing is interval-certified. All confidence intervals
are percentile cluster-bootstrap intervals that cover **sampling error only**; they do not
include the systematic finite-cutoff bias discussed in §5.3.

### 5.1 Three implementations of H_t

- `dbn/dbn_common.py`: composite Gauss–Legendre quadrature in mpmath.
- `dbn/hflow.py`: exact fixed-point trapezoid rule, step 1/160, cut-off u ≤ 1.25, 80–110
  digits. It produced the published collision points.
- `dbn/indep_ht.py`: tanh-sinh quadrature on subintervals (written for the math review).

**Agreement at the collisions.**
- The claims run (`dbn/check_collision.py`, 624 Gauss–Legendre nodes, 40 digits) agrees with
  the trapezoid code on the (9/10) collision to about 27 significant digits in t₀ and 29 in
  z₀; that reflects the claims run's precision, not the accuracy of the published points.
- **Independent quadratures agree to about 10⁻⁴³ for both collisions.** An independent
  Gauss–Legendre evaluation (2880 nodes, 90 digits) at the published (x₀, t*) gives Newton
  corrections of at most 3.3·10⁻⁴⁴ in x and 1.5·10⁻⁴⁵ in t, for both (9/10) and (34/35)
  (`results/dbn/indep_collision.out`).
- Recomputing with the trapezoid code at step 1/256, u ≤ 1.5 and 110 digits moves the points
  by Δt* ≤ 1.2·10⁻⁴⁰ and Δx₀ ≤ 2.5·10⁻³⁹ (`results/dbn/refine_check.out`).
- The tanh-sinh implementation, solving from coarse seeds, confirms both collisions to the 30
  printed digits (`results/dbn/indep_ht_c910.out`, `indep_ht_c3435.out`).

### 5.2 Two computed collisions of H_t zeros

The method follows each adjacent pair's critical point c_k(t) and records the sign change of
H_t(c_k(t)). It bisects to about 10⁻⁴⁵, then solves H = H_z = 0 by 2D Newton (`dbn/track.py`,
`dbn/fold_check.py`).

| Pair (zeros at t = 0) | t* | x₀ = z₀ | A = H_zz/2 | B/A | C/A | κ_r | κ_ℓ |
|---|---|---|---|---|---|---|---|
| 9/10 (γ = 48.0052, 49.7738) | −1.877909072972535264767 | 99.328469305575750948 | 2.76275·10⁻¹⁶ | −0.419058 | 0.0629861 | −0.0992746 | 0.448990 |
| 34/35 (γ = 111.0295, 111.8747) | −0.384561116627981414865 | 223.275780185615519914 | −1.76353·10⁻³⁶ | −0.486922 | 0.0700905 | −0.193823 | 0.749687 |

**Why these are the first collisions in their windows.**
- Direct zero counts (`dbn/count_check.py`): window z ≤ 103 has 10 real zeros at t = 0 and at
  t = −1.87, but 8 at −1.89; window z ≤ 270 has 46 at t = 0 and at −0.38, but 44 at −0.39.
- These counts confirm "first collision" because in s = −t, H solves the forward heat
  equation, and the number of sign changes on an interval is non-increasing apart from flux
  through the ends (Sturm / zero-number property [memory]). So real zeros in a fixed window
  cannot be created as t decreases, degenerate events included; Proposition 1 describes the
  nondegenerate case.
  Zero 46 sits 0.5 from the z = 270 edge but moves inward (269.513 at t = 0, 269.496 at
  t = −0.38), so there is no boundary flux.

**Later events and tracker completeness.**
- Lowest 46 zeros (z ≤ 270, step 0.01): after (34/35), the tracker found, further back in t,
  27/28 at −0.8811, 45/46 at −0.9600, 24/25 at −1.1439, 40/41 at −1.1519, 19/20 at −1.2755 and
  13/14 at −1.2853. It then stopped at t ≈ −1.355 when its own critical-point ordering check
  failed. That is a limitation of the code, not a mathematical event. **The event list is
  complete only down to t ≈ −1.355.**
- Lowest 10 zeros (z ≤ 103, step 0.02): after (9/10), the tracker found 4/5 at −3.9548, 7/8 at
  −4.0338 and 2/3 at −12.4945. Those first three events are reproduced at steps 0.01, 0.02,
  0.025, 0.04 and 0.05. Near t ≈ −6.38 the tracker's bookkeeping becomes step-size
  dependent (ordering failures at steps 0.01 and 0.04; a spurious event with a zero-count
  warning at 0.025 and 0.05; `results/dbn/track_low10_dt_sensitivity.out`). The event list for
  this window is therefore complete only down to about t ≈ −6.38, and the (2/3) event is
  reported without a completeness claim.
- **Every collision we found (lowest 46 zeros) is at t < 0.** Which collision is "first"
  depends on the window.

**Fold-law relative errors.** τ = t − t*. The rows "t = 0" use the actual zeta zeros.

| τ | pair 34/35: r²/(8τ) − 1 | pair 34/35: lobe area/(\|A\|r³/6) − 1 | pair 9/10: r²/(8τ) − 1 | pair 9/10: lobe area/(\|A\|r³/6) − 1 |
|---|---|---|---|---|
| t = 0 (τ = 0.3846 or 1.878) | −7.14% | +33.1% | −16.7% | +128% |
| 0.1 | −1.92% | +7.77% | −0.99% | +4.6% |
| 10⁻³ | −1.94·10⁻⁴ | +7.50·10⁻⁴ | −9.93·10⁻⁵ | +4.49·10⁻⁴ |
| 10⁻⁶ | −1.94·10⁻⁷ | +7.50·10⁻⁷ | −9.93·10⁻⁸ | +4.49·10⁻⁷ |
| 10⁻¹² | −1.94·10⁻¹³ | +7.50·10⁻¹³ | −9.93·10⁻¹⁴ | +4.49·10⁻¹³ |

The lobe area |∫_{z₋}^{z₊} H_t| is the critical-value gap of F_t. Its bar pairing is verified
only near the collision (Proposition 1(c); persistence check at τ = 0.01), not at t = 0. A gap
need not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only
finite superlevel H0 bar, born at the right end, has length 7/24 (countercontrol taken from
main#307 comment 6116121839, §2).

**How well the theory fits.**
- Over τ ≤ 10⁻⁴, both errors scale as τ^{1.000000} to τ^{1.000001}.
- They agree with κτ to a relative 6·10⁻⁶ – 3.5·10⁻⁵, which is the size of the O(τ²) term.
- The midpoint drift 2(B/A)τ agrees to the same accuracy.
- Below t*, Im z / √(2(t* − t)) = 1.00097, 1.0000097 and 1.0000001 at t* − t = 10⁻², 10⁻⁴
  and 10⁻⁶ (pair 34/35; pair 9/10: 1.00050, 1.0000050, 1.00000005).
- **What this convergence does and does not show.** The computed H̃ is a finite sum
  Σ W_k e^{t u_k²} cos(z u_k), which is itself an exact analytic solution of
  ∂_t H = −∂_z² H. The small-τ convergence therefore holds for any real-analytic
  backward-heat solution, including the quadrature approximant; it checks the κ formulas and
  the code. Agreement with ζ rests on the quadrature accuracy checks (H_0 = Ξ/8 and the
  independent quadratures of §5.1). The ζ-specific content is (t*, x₀, A, B/A, C/A) and the
  t = 0 rows.

**Persistence check** (pair 9/10, τ = +0.01, sampled antiderivative): there is exactly one
finite bar in the window, with length 1.04454·10⁻¹⁸. The exact lobe area is 1.0450·10⁻¹⁸. At
τ = −0.01 there is no finite bar.

**Reading.**
- The fold law describes H_t only near the collision time. At t = 0 the low zeta zeros are far
  from that regime.
- t* = −0.38456 for the (34/35) pair numerically coincides with the reported historical lower
  bound ≈ −0.385 attributed to Norfolk–Ruttan–Varga 1992. That value, with −50, −5 and −0.0991,
  is recorded in CSV 1994, p. 108, eq. (1.5) [corroborated second-hand; originals unchecked].
  The pair used there is not confirmed. The 1991 Csordas–Ruttan–Varga value −0.0991 is
  stronger than that 1992 entry, so the conventions or pairs in that table need checking.
- These collisions only give bounds Λ ≥ t* that Rodgers–Tao's Λ ≥ 0 supersedes.

### 5.3 Short-bar statistics: Gaussian process, CUE and zeta

**Notation and estimator.**
- α is the exponent of the pair or gap law near 0 (density ∝ r^α), so the small-gap CDF slope
  is α + 1. β = (α+1)/m is used only in the Math- sense of §2 (m = 3 here).
- s is the short-bar CDF slope (N(ℓ) ~ ℓ^s); the density exponent is s − 1.
- Bar lengths are in units of the median bar. s is the Hill (maximum-likelihood power-law)
  estimate on all bars below a cutoff.
- Intervals are 95% cluster-bootstrap percentile intervals, B = 400 (sampling-only). A cluster
  is one Gaussian torus, one CUE matrix, or a block of 2000 consecutive zeta lobes. Analysis
  RNG seed 424242.
- Barcode: superlevel H0 of the antiderivative (convention of §3) on the alternating sequence of
  critical values, by the elder rule. The infinite bar and bars born at path endpoints are dropped
  (`stats/analyze.py`). `stats/barcode_bruteforce.py` checks this against a brute-force
  elder-rule computation on 400 random multi-path cases and an analytic test function: 0
  mismatches, maximum difference 4·10⁻¹⁶. Note that `dbn/pers1d.py`, used on the sampled
  windows of §5.2 and §5.4, keeps endpoint-born bars; the two conventions differ by one or two
  bars per window.

**Samples.**

- **Zeta.** Riemann–Siegel main sum plus the corrections C₀, C₁, C₂ (`stats/rs_zeta.py`).
  - The correction functions are Chebyshev fits to mpmath derivatives (fit residual ≤ 3·10⁻¹⁵
    on the fit nodes; the end-to-end check below is the real validation).
  - Maximum |error| against `mpmath.siegelz` at 60 random points: 3.4·10⁻¹⁰ on [10⁴, 10⁵] and
    6.5·10⁻⁹ on [10⁵, 10⁶] (validation RNG seed 20261012); a fresh check at 14 points,
    including the midpoint of the closest pair, gives ≤ 2.6·10⁻⁹ (`results/stats/rs_spot_check.out`).
  - Zeros are bracketed on a grid of 20 points per mean spacing, plus a near-miss search that
    recovered 4 and 52 close pairs hidden inside one grid cell. The GUE expectation for such
    within-cell pairs is 4.4 and 55.1; gaps below 0.05 number 205 against a GUE estimate of
    220.6.
  - Zero counts are 127,927 and 1,609,077. Both equal `mpmath.nzeros(hi) − nzeros(lo)`
    exactly, as do the sub-windows [500000, 500300] (538/538) and [999700, 10⁶] (573/573).
  - Lobes use 16-point Gauss–Legendre quadrature. Relative error against
    `mpmath.quad(siegelz)` is 3·10⁻¹¹ to 5·10⁻⁹ on typical lobes, and 1.6·10⁻⁵ on the
    worst-conditioned lobe (a pair 0.003 apart at t ≈ 663318.508).
  - Scale: t is unfolded by the smooth counting function θ(t)/π + 1, and the integrand is
    Z(t)/√(log(t/2π)). Raw Z gives the same exponents.
  - We call this antiderivative, with endpoint-born bars dropped from its barcode, the
    **Hardy-Z primitive**. It is neither the ξ-weighted primitive of §4 item 3 nor a barcode
    of the de Bruijn–Newman F_t (§3), and every ζ statistic below is about it alone.
- **Gaussian.** The 1D case of the project's covariance, applied to H = F'
  (`stats/gauss_proc.py`).
  - Fourier weights q_n ∝ exp(−ω² n²/2) with ω = 1/M, M = 1000 (L = 2π·1000), so the
    covariance of H is ≈ e^{−x²/2}.
  - 500 samples, seeds 10000–10499, FFT on 2¹⁹ points. F is the exact spectral antiderivative.
    Lobe-area relative error against direct trigonometric sums is 6·10⁻¹¹.
  - Zeros come from cubic Hermite interpolation. `gauss_proc.py` flagged 17 grid cells as
    possible hidden pairs but did not resolve them. **5 hidden pairs resolved after the fact;
    effect on s ≤ 0.003** (`stats/gauss_hidden.py`: a Hermite subgrid search over 113
    candidate cells found 5 real pairs, against 6.0 expected; adding their bars moves s from
    0.6500 to 0.6486 at cut 0.01 and from 0.6567 to 0.6538 at cut 0.003, and the gap slope
    from 1.982 to 1.980). The Gaussian rows below do not include them; the correction pushes
    the cut-0.003 estimate slightly further below 2/3.
- **CUE.** Haar U(N) matrices, with Z_U(θ) = Π_j 2 sin((θ − θ_j)/2) evaluated as a direct
  product (`stats/cue.py`).
  - N = 12: 86,016 matrices. N = 128: 8,064 matrices. Each gives 1,032,192 lobes. Master
    seed 1. Unfolded to mean gap 1. Quadrature with 16 versus 32 nodes differs by ≤ 2·10⁻¹³.
  - **End/cut convention, which differs from the T-CUE target of §6.** Each circle is cut at
    θ₁, the first eigenangle after angle 0, and F = ∫Z_U is treated as a path from θ₁ to
    θ₁ + 2π. F is not periodic, because Z_U's constant Fourier mode is nonzero. Bars born at
    the path endpoints are dropped, so the two endpoint-adjacent lobes can never give short
    bars. Those positions are atypical: the wrap-around gap contains angle 0 and is
    length-biased (mean 1.178; P(gap < 0.2) = 0.0012 against 0.0089 for interior gaps), and
    the excluded lobes hold 11.25% of lobes below 0.01 × median at N = 12, not 2/12. Exponents
    are unaffected (lobe Hill 0.977 for all lobes against 0.973 for interior lobes; gap Hill
    3.003 against 3.003), but the coefficient at N = 12 is shifted by roughly 10% (a
    numerics-review estimate; the share figures are in `results/stats/cue_endpoint_check.out`).

**Short-bar CDF slope s.** −1/3 density gives s = 2/3; density ℓ⁰ gives s = 1.

| Sample | Lobes / bars | s (cut 0.1) | s (cut 0.01) | s (cut 0.003) | LS fit on [10⁻³, 10⁻²] |
|---|---|---|---|---|---|
| Gaussian | 998,836 / 498,674 | 0.631 [0.627, 0.636] | 0.650 [0.641, 0.659] | 0.657 [0.643, 0.671] | 0.643 [0.631, 0.657] |
| CUE N=128 | 1,032,192 / 501,103 | 0.791 [0.786, 0.796] | 0.935 [0.920, 0.951] | 0.971 [0.942, 1.001] | 0.949 [0.926, 0.972] |
| CUE N=12 | 1,032,192 / 386,701 | 0.901 [0.893, 0.909] | 0.980 [0.954, 1.007] | 1.019 [0.973, 1.070] | 0.982 [0.948, 1.018] |
| ζ, t ∈ [10⁴, 10⁵] | 127,926 / 63,962 | 1.002 [0.977, 1.029] | 1.049 [0.968, 1.145] | 1.001 [0.873, 1.147] | 1.034 [0.921, 1.173] |
| ζ, t ∈ [10⁵, 10⁶] | 1,609,076 / 804,536 | 0.947 [0.941, 0.953] | 0.980 [0.959, 1.001] | 0.972 [0.938, 1.009] | 0.978 [0.950, 1.007] |

**Finite-tail estimates of the density exponent at cut 0.01, next to the controls and their
theoretical expectations.** The values −1/3 and 0 are what theory predicts under stated
conditions. They are not calibrated answers for these finite controls: the fixed-N CUE full bar
law is the unproved T-CUE target (§6.1), and the Gaussian 1D theorem's hypotheses were not
checked for this primitive (below).

| Sample | Hill estimate of s − 1 at cut 0.01 (sampling-only CI) | Theoretical expectation | Interval contains it? | Estimate − expectation |
|---|---|---|---|---|
| Gaussian | −0.350 [−0.359, −0.341] | −1/3 | **no** | −0.017 (−0.018 with the 5 hidden pairs) |
| CUE N=12 | −0.020 [−0.046, +0.007] | 0 | yes | −0.020 |
| CUE N=128 | −0.065 [−0.080, −0.049] | 0 | **no** | −0.065 |
| ζ, t ∈ [10⁵, 10⁶] | −0.020 [−0.041, +0.001] | — | lower end 0.29 above −1/3; the upper end only just reaches 0 | — |
| ζ, t ∈ [10⁴, 10⁵] | +0.049 [−0.032, +0.145] | — | — | — |

**Reading of these numbers.** For the Hardy-Z primitive, windows and Hill estimator specified
above, the observed finite-tail estimates are substantially above −1/3 (the cut-0.01 point
estimate for t ∈ [10⁵, 10⁶] is 0.31 above it) and near 0. The sampling-only block-bootstrap
intervals exclude −1/3 at every reported cutoff in both windows (and, for t ∈ [10⁵, 10⁶], at
all three block sizes below). That is all they show. They do not bound finite-cutoff bias,
dependence in this deterministic sequence, estimator misspecification, post-inspection
selection or the joint gap/mark behaviour, and no limiting measure is shown to exist. So this
is exploratory evidence about a finite-window statistic, not a limiting density exponent and
not a calibrated rejection of −1/3 that accounts for all errors. The controls do not calibrate
those errors either: at cut 0.01 their estimates sit −0.017 to −0.065 from their theoretical
expectations. The value −0.020 [−0.041, +0.001] is not a measurement of 0, and these data
cannot separate ℓ⁰ from a mark-driven ℓ^q with |q| ≲ 0.04. At cut 0.003 the Gaussian interval
contains 2/3, and its local slopes on [10⁻⁴, 10^−3.5] and [10^−3.5, 10⁻³]
are 0.678 and 0.681. Finite-ℓ corrections are expected: the 1D theorem has a C₁ ℓ^{1/4} term,
whose sign and size for this F have not been computed, and the hypotheses of the 1D theorem
were not checked verbatim for an antiderivative with a zero mode.

**Block-size sensitivity of the zeta intervals** (`stats/block_sensitivity.py`,
`results/stats/block_sensitivity.json`). log|ζ| is approximately log-correlated, so
2000-lobe blocks need not be independent at the amplitude level. Recomputing for ζ, t ∈ [10⁵, 10⁶]
with blocks of 500, 2000 and 20000 lobes (3219, 805 and 81 clusters):

| Statistic | Value | 500 | 2000 | 20000 |
|---|---|---|---|---|
| s at cut 0.01 | 0.980 | [0.958, 0.999] | [0.960, 1.001] | [0.958, 1.004] |
| s at cut 0.003 | 0.972 | [0.936, 1.009] | [0.937, 1.008] | [0.940, 1.006] |
| flat-density coefficient proxy F(0.01)/0.01 | 1.038 | [1.016, 1.057] | [1.018, 1.062] | [1.013, 1.065] |
| fraction of bars < 0.01 × median in the lowest local-amplitude decile | 0.038 | [0.034, 0.042] | [0.034, 0.042] | [0.033, 0.043] |

The exponent intervals barely move; the amplitude-sensitive intervals widen by up to about a
quarter at 20000-lobe blocks, and those are the ones to quote for coefficients. (F is the
fraction of bars below x × median, so F(x)/x is the coefficient the CDF would have if the
density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.769, 0.910] with 20000-lobe
blocks (only 7 clusters; [0.786, 0.911] with 2000-lobe blocks); the two windows differ, which
is consistent with, but not evidence for, a slowly drifting coefficient.)

**Small-gap CDF slope (= α + 1, cut r < 0.2).**

| Sample | Slope | 95% CI |
|---|---|---|
| Gaussian | 1.982 | [1.965, 2.001] |
| CUE N=128 | 2.993 | [2.930, 3.053] |
| CUE N=12 | 3.003 | [2.940, 3.067] |
| ζ [10⁴, 10⁵] | 3.108 | [2.914, 3.350] |
| ζ [10⁵, 10⁶] | 2.930 | [2.882, 2.983] |

Two-term GUE expectations at cuts 0.4, 0.3, 0.2 and 0.15 are 2.84, 2.91, 2.96 and 2.98.

**Zeta agrees with CUE in the leading exponent only, not in gap shape.** At moderate r the zeta
gap shape is significantly off GUE and off CUE N=12. Gap Hill slope at cut 0.4: ζ [10⁴, 10⁵]
3.05 [2.98, 3.13]; ζ [10⁵, 10⁶] 2.896 [2.876, 2.916]; GUE two-term 2.836; CUE N=12 2.859
[2.837, 2.880]. An independent CUE simulation at cut 0.4 (`stats/cue_gap_shape.py`) gives
N=2: 2.924, N=3: 2.872, N=4: 2.864, N=6: 2.851, N=12: 2.855, so ζ [10⁵, 10⁶] lies between N=2
and N=3. This is in line with, though not a test of, the finite-height spacing model of
Bogomolny–Bohigas–Leboeuf–Monastra 2006 (eqs. (19)–(20)), which suggests an effective dimension
N_eff = log(T/2π)/√(12Λ₀), Λ₀ ≈ 1.57314 (an arithmetic constant unrelated to the de
Bruijn–Newman Λ), together with an additional spacing rescaling [primary, via review
5486879914]; our evaluation gives N_eff ≈ 2.2–2.8 on [10⁵, 10⁶]. Our CUE gaps are unfolded to
mean 1 only, without that rescaling. Their extension from the two-point calculation to all
correlations and nearest neighbours is conjectural, and nothing establishes the same matching
for barcode marks. Keating–Snaith (eqs. (8) and (92)) use N = log(T/2π) as a mean-density
matching rule in their model for value distribution and moments, with arithmetic corrections in
the moment conjecture; that is not a proved equality of finite-height ζ and CUE laws [primary,
via review 5486879914]. So in these models the gap part and the amplitude part of a short-bar
coefficient would be matched to CUE at very different N, and any coefficient-level comparison
needs a matching rule declared in advance. Exponents are unaffected.

**Fold check.** For gaps r < 0.35, the slope of log(lobe area) on log(gap) is 3.000
(Gaussian), 3.032 (CUE 128), 3.025 (CUE 12), 3.010 (ζ [10⁴, 10⁵]) and 3.009 (ζ [10⁵, 10⁶]).

**What makes zeta's short bars short.** Take the 8,352 ζ [10⁵, 10⁶] bars below 0.01 × median:
- 100% are an adjacent maximum–minimum pair, and 100% have a gap in the lowest gap decile;
- the mean log₁₀ shortfall is −2.70: gap −0.87, peak/local amplitude −1.90, local amplitude
  +0.05, shape +0.02; Spearman correlation with gap is 0.82, with local amplitude −0.02;
- **short bars lie in the lowest local-amplitude decile at 3.8%, the same as all close gaps
  (4.0% for gap < 0.1 and for gap < 0.2), so bar shortness adds no amplitude selection.**
  The local amplitude is the median peak of the 12 nearest lobes, and level rigidity makes a
  close pair's neighbours larger; with a far-neighbour proxy (skipping the nearest 6 lobes on
  each side) close gaps sit in the lowest decile at 8.0–8.1%. The far-neighbour proxy removes
  about 4 of the 6 points of deficit (10% → 4.0% becomes 10% → 8.0–8.1%). Roughly two-thirds of
  the deficit is the definition; the remaining ~2 points may be a real correlation
  (`results/stats/localamp_baseline.out`).
- CUE N=128 is the contrast: its short bars sit in the lowest local-amplitude decile at 33%,
  against 9.5% for its close gaps, which supports a mark (small-amplitude) effect there.

**Matched lower tail** (`stats/matched_tail.py`): s(lobe area) − s(gap³) at the same tail
fraction q.
- ζ [10⁵, 10⁶]: q = 0.01, 0.0004 [−0.0088, 0.0097]; q = 0.003, −0.014 [−0.031, +0.0005]
  (borderline).
- CUE N=12: q = 0.01, −0.019 [−0.032, −0.005].
- CUE N=128: q = 0.01, −0.034 [−0.052, −0.016]; q = 0.003, −0.024 [−0.058, +0.006].
- ζ [10⁴, 10⁵]: q = 0.003, +0.058 [0.007, 0.119] on 384 lobes (one of many comparisons).

**No mark effect larger than about 0.02 in s is detected for zeta at q = 0.01. Small effects
(about 0.02) appear in CUE N=12, and larger ones in CUE N=128, consistent with the subsection
§D2 mark mechanism at finite ℓ. The admission condition itself is untested.** This test
detects mark effects only of size |Δs| ≳ 0.02 at q ≥ 0.001. It cannot verify the mark
condition and cannot exclude ℓ^q with small negative q. The slower convergence of CUE N=128
is consistent with small-amplitude marks; that reading was made after seeing the data.

**Reading.** With the same cubic fold, the samples whose small gaps repel quadratically (CUE,
and ζ, with gap slopes ≈ 3, through the Hardy-Z primitive) give finite-window estimates near
0, while the Gaussian sample with linear repulsion gives estimates near −1/3. This is
**consistent with the equation (D1) prediction for α = 2, m = 3 (β = 1): heuristic for ζ,
theorem target for CUE (T-CUE, §6).** It does not establish a limiting exponent for any of
them. (D1) is a theorem about intensity
measures of random models under its hypotheses 1–4, including a negative mark moment under the
limiting collision-mark measure (`CLAIMS.md` 5.4); for the deterministic ζ none of these
objects is defined or verified. Pair correlation, the nearest-neighbour gaps measured here and
the elder-rule pairs are three different objects (§1.4). The analogue of the mark condition
here concerns how small |Z''| can be at close pairs in the chosen normalization; it is not
proved for ζ. Fits over moderate lengths give misleading intermediate exponents:
for example CUE N=128 below 0.1 × median gives density ≈ −0.21. The heights 10⁴–10⁶ are modest.

### 5.4 Small Hardy-Z windows

| Window | Zeros found / exact (`mpmath.nzeros`) | Riemann–von Mangoldt main term | Finite bars | Single-lobe | Slope log(area) on log(gap) | Residual sd (total) |
|---|---|---|---|---|---|---|
| [600, 900], step 0.02 | 228 / 228 | 228.0 | 113 | 80.5% | 2.995 | 0.257 (1.264) |
| [5000, 5200], step 0.015 | 214 / 214 | 213.2 | 106 | 82.1% | 3.054 | 0.351 (1.394) |

The main term is printed for orientation only; the exact counts are the completeness check.
The ten shortest bars in [600, 900] all have ℓ/gap³ between 0.50 and 1.01. These barcodes use
`dbn/pers1d.py`, which keeps endpoint-born bars (§5.3). No exponent is estimated from windows
of about 110 bars. On normalization dependence, see §4 item 3.

## 6. Research targets worth exploring

Ranked by value to the project's own goals (separating geometry from measure, and admission),
not by any relevance to RH. These are labelled targets, not results.

### 6.1 Theorem target T-CUE: the short-bar law for fixed-N CUE (first)

- **Question.** Let U be Haar on U(N), N ≥ 2, Z_U(θ) = Π_j 2 sin((θ − θ_j)/2), and
  F_U = ∫ Z_U on the deterministic arc [0, 2π) (cut at angle 0), with bars born at the arc
  ends dropped (N even if a periodic Z_U is wanted). Does the expected short-bar density
  satisfy ν_N(ℓ) → c_N ∈ (0, ∞) as ℓ ↓ 0, and is c_N the local marked-pair value

      c_N = (1/3) ∫ b₀ κ^{−1} dλ

  in the notation of equation (D1) with α = 2, m = 3, β = 1, or do other strata add to it
  (see Feasibility)? What is c_N explicitly (N = 2 in closed form), and what is the
  next-order term?
- **Why it matters.** To our knowledge (main `8977619e`, Math- `79cb314d`; open drafts not
  checked), it would be the first natural non-Gaussian law in which the project's transfer
  theorem is discharged with α ≠ 1. The fold is the same (m = 3); the contact law is
  different (r² from the Vandermonde |Δ|²); the predicted exponent becomes 0. This is exactly the
  altered pair-count tilt boundary prediction in
  [`experiments/universality/PROTOCOL.md`](../universality/PROTOCOL.md) lines 53–64. It would
  show that CONTACT (pair intensity ∝ r) is a necessary admission input, not a consequence of
  the fold, beyond the toy control `cubic_2` and the product-model phase diagram of subsection
  §D2.
- **Feasibility.** Moderate and finite-dimensional. The Weyl density and the Palm conditioning
  at a coincidence are explicit. Heuristically, a third eigenvalue near the pair has density
  ∝ |θ₃ − θ₀|⁴ and κ ∝ |θ₃ − θ₀|, so P(κ < ε) ≍ ε⁵ (q = 4 in subsection §D2) and E[κ^{−1}] < ∞.
  Elder selection near contact works as in Proposition 1(c). The main work is **all four (D1)
  hypotheses 1–4**, including domination (1–2); hypothesis 4 (fixed-constant comparison) is
  the one most likely to fail near triple coincidences. (D1) is a local marked-pair theorem.
  At β = 1 even a merely bounded density of short bars from other strata (nonadjacent
  pairings, deaths decided by distant critical values, near-triple coincidences) contributes
  at the same order as c_N. So the proof target must also exclude or control those strata and
  the global elder-rule selection; (D1) hypotheses 1–4 alone do not identify the full bar
  coefficient. Do N = 2 exactly first.
- **Fallback.** If uniformity fails, the cumulative theorem of
  `CUMULATIVE_TRANSFER_CORRECTION.md` still applies under its hypotheses: κ > 0; an envelope
  0 ≤ B ≤ G with B → B₀ and ∫ G κ^{−β} dλ < ∞; h continuous and strictly increasing with
  h(0+) = 0 and the retained ratio limit h/(κ r^m) → 1; and a lower comparison h ≥ c₀ κ r^m.
  It gives N(ℓ) ~ ℓ^β/(α+1) · ∫ B₀ κ^{−β} dλ and cannot be differentiated without further
  control.
- **What counts as a result.** A Math- frontiers proof candidate with explicit positive c_N, a
  stdlib checker (exact N = 2), and a coefficient-level comparison with CUE data. **Before any
  coefficient comparison**, recompute the CUE bars cut at angle 0 under exactly this
  convention, or derive c_N for the cut-at-θ₁ convention used in `stats/cue.py` (§5.3: about a
  10% shift at N = 12).
- **Risk.** Uniformity may fail near triple coincidences. The N → ∞ (sine-kernel) limit needs
  amplitude normalization (log|Z_U| at a fixed angle has variance of order ½ log N, after
  Keating–Snaith [memory]; like Selberg's theorem in §6.2, that is not a law for the selected
  local amplitudes). It says nothing about ζ without further conjectures.
- **Numerical grounding** (cut 0.01; cut 0.003 in parentheses): CUE N=12 s = 0.980
  [0.954, 1.007] (1.019 [0.973, 1.070]); CUE N=128 0.935 [0.920, 0.951] (0.971 [0.942, 1.001]).
  Gap CDF slopes 2.99–3.00; fold slope 3.03.

### 6.2 A frozen zeta short-bar comparison (second)

**Conjecture Z** (exploratory; motivated by GUE behaviour of small zero gaps and by a
small-mark condition for the amplitude-normalized contact coefficient, neither of which is
proved; the limit and the observable are stated explicitly below, and existence of the limit
is part of the conjecture).
- *Primitive and window.* For height T the window length is fixed as H = H(T) = √T (so
  H log T → ∞ and H/T → 0). Let g = Z(t)/√(log(t/2π)) in the unfolded variable θ(t)/π + 1,
  and let F = ∫ g on the path from the first to the last zero of g in [T, T+H]. Bars are the
  finite superlevel H0 bars of F (§3 convention); the essential bar and bars born at the two
  path endpoints are dropped, as in `stats/analyze.py`.
- *Local amplitude of a bar.* A lobe is the interval between consecutive zeros of g, and its
  peak is max |g| there; every peak is positive. A bar born at the critical point z_b, adjacent
  to its death point or not, gets σ_b = the median peak of the 12 lobes nearest z_b, six on
  each side. Lobes outside the window are read from g on ℝ, so σ_b > 0 is always defined.
- *Normalized bars.* ℓ̃_b = (ℓ_b/σ_b)/M_{T,H}, where M_{T,H} > 0 is the median of ℓ_b/σ_b over
  the window's bars: the median of the normalized bar lengths, not the amplitude median. Let
  μ_{T,H}(E) = #{bars with ℓ̃_b ∈ E} / #{zeros in [T, T+H]}. Section 5.3 measures ordinary
  bars and amplitude-normalized *lobes* separately, so neither is a measurement of μ_{T,H}.
- *Statement (iterated limit).* (i) There is ℓ₀ > 0 such that for every ℓ ∈ (0, ℓ₀] the
  limit G(ℓ) := lim_{T→∞} μ_{T,√T}((0, ℓ]), over all real T, exists and is positive.
  (ii) log G(ℓ)/log ℓ → 1 as ℓ ↓ 0. The conjecture is for this one window choice, H(T) = √T;
  whether the limit exists, or is the same, for other window lengths H(·) is not part of it.
  On a fixed window, ℓ → 0 alone is vacuous, which is why T → ∞ comes first. The Gaussian
  comparison is C₀ ℓ^{−1/3}, i.e. cumulative exponent 2/3 (Math- d1 theorems).
- *What (ii) does and does not say.* (ii) is a cumulative scaling exponent, G(ℓ) = ℓ^{1+o(1)}.
  It is not a density statement. The probability measure with mass 2^{−n} at each 2^{−n}
  (n ≥ 1) has ℓ < G(ℓ) ≤ 2ℓ for ℓ ≤ 1/2, hence exponent 1, but no Lebesgue density.
  "Density exponent 0" needs extra regularity or transfer assumptions on the limit.
- *Coefficient form (not part of the conjecture; only with a caveat).* At finite T, μ_{T,H} is
  a finite empirical measure: μ_{T,H}((0, ℓ]) = 0 below the window's shortest normalized bar,
  so it has no positive ℓ → 0 asymptotic c(T) ℓ. A finite-T coefficient c(T) can only be a
  finite-tail approximation μ_{T,H}((0, ℓ]) ≈ c(T) ℓ fitted on a declared range
  [ℓ_lo(T), ℓ_hi(T)] (fixed cutoffs, frozen in advance as in the protocol below, not the ℓ₀
  of (i)); it lies outside Conjecture Z. A comparison of such a c(T) with a CUE reference
  must allow for an arithmetic correction of unknown size. In the analogous moment
  conjectures the arithmetic factor (the
  Hughes–Keating–O'Connell a(k)) is part of the moment model, not a derived correction for
  this barcode observable. The comparison must also fix in advance which N defines the CUE
  reference: the BBLM finite-height spacing model suggests N_eff ≈ 2.2–2.8 for gaps on
  [10⁵, 10⁶] (our evaluation of their formula, §5.3), while Keating–Snaith match moments with
  N = log(T/2π), a rule in their model (§5.3). Without that, a "fail" on unnormalized bars
  would be uninterpretable. The unnormalized coefficient may fail to converge, but that
  expectation is heuristic. Selberg's central limit theorem describes log|ζ| at uniformly
  sampled heights (Radziwiłł–Soundararajan, Theorem 1: log|ζ(½+it)| for t uniform in [T, 2T],
  normalized by √(½ log log T)) [primary, via review 5486879914]. An analogous law for these
  selected local amplitudes, or slow drift of the bar coefficient, is heuristic and is not
  supplied by that theorem. Likewise, the CUE derivative-moment model has a −3/2 boundary
  (Hughes–Keating–O'Connell, Theorem 1.2 and Remark 2.2), and HKO conjecture a zeta analogue
  (Conjecture 1.3, under RH; negative moments also need simple zeros) [primary, via review
  5486879914]. This motivates an analogy, but proves neither the required negative moment of
  our collision mark nor coefficient drift. (The original Gonek 1989 and Hejhal 1989 proofs
  were not audited; the Hejhal central limit theorem that HKO quote assumes RH and a
  quantitative small-gap hypothesis.)

**Protocol.** Follow `PROTOCOL.md` lines 53–64: before generating new data, freeze the windows
(for example t ∈ [10⁶, 10⁷] with the existing Riemann–Siegel pipeline, about 1.9·10⁷ zeros, or
published zero tables at large height), the cutoffs, the estimator, the block size (20000
lobes or more for coefficient intervals), a bias allowance at each cutoff (informed by the
controls, which are not certified calibrators, §5.3), the CUE reference and its matching rule
(separate N for gaps and amplitudes, or a zeta-calibrated reference), and the decision rule.
The current windows were inspected, so they stay exploratory. **What counts as a result:** a
frozen prediction file plus a run giving pass/fail/indeterminate on the finite-window analogue
of (ii) (cumulative exponent 1) at the stated heights. It is not a theorem, not a limiting
exponent and not RH evidence.

### 6.3 Lower priority

- **Proposition 1 and the vineyard birth law** (§3): essentially done; slice A was read
  (review 5486843735, AMEND), and the delta readback of the applied amendments returned
  PASS (scoped) (review 5487707101 at `0969b327`). Novelty is essentially nil, and it must not
  be presented as new or as related to RH.
  Follow-up question: in the project's own scale or heat-flow families, does a PDE likewise
  force folds to be transversal? Deterministic polynomial heat-flow fold controls with
  finite-interval H0 are the scope of the open proposal in main#307 comment 6116121839 (§2),
  which this packet does not take.
- **Sensitivity of the barcode to normalization and marks** (§4 item 3): a short deterministic
  lemma (ℓ_w = w(x₀)ℓ(1 + O(r²)) for C^{1,1} weights on a fold neighbourhood, only 1 + o(1) for
  C¹; any w^{−β}-weighted coefficient transfer also needs uniform positive-weight and
  marked-measure hypotheses) and the existing table. It is the
  deterministic counterpart of the amplitude counterexample and of the subsection §D2
  small-mark boundary, and informs how `PROTOCOL.md` observables are normalized. Low value if
  it stays a remark.
- **Collision–lobe-area correspondence at t = 0** (optional, lowest): check the lobe area
  ℓ(0) ≈ (8√2/3)|A||t_c|^{3/2} against the backward collision time t_c for close pairs at
  moderate heights, using Polymath15's effective approximation of H_t; whether that lobe is a
  finite H0 bar must be checked separately (Proposition 1(c) covers only small τ).
  Illustration only; known in substance, and accurate only for very close pairs (at t = 0 the
  two low collisions are off by 19% and 73% in this form; `results/dbn/fold_analyze.out`).

## 7. What this packet does not claim or pursue

- **Any RH program.** The note's proposed route ends at showing that H_t has no real-zero
  merger for t > 0. RH (Λ ≤ 0) implies that statement. Conversely, a positive Λ forces a
  multiple real zero of H_Λ at finite height, by a deduction from Polymath15 Theorem 1.5(i)
  recorded in `CLAIMS.md` 6.2. So, read as "no multiple real zero of any multiplicity for
  t > 0", the statement is equivalent to RH, not a route to it. Read as "no nondegenerate fold
  for t > 0", it is not shown equivalent: the zero at Λ may have higher multiplicity. No
  compact bound is available at Λ = 0. No packet content is progress toward RH, support for
  RH, or a bound on Λ beyond known results.
- **That −1/3 has any bearing on where ζ's zeros lie.** It has none. Nor does this packet
  settle ζ's short-bar exponent. For the specified Hardy-Z primitive, windows and Hill
  estimator, the finite-tail estimates are substantially above −1/3 and near 0, and
  sampling-only intervals exclude −1/3 at the reported cutoffs. That is exploratory evidence
  about a finite-window statistic, not a limiting exponent, and nothing is claimed for the
  ξ-weighted primitive or the de Bruijn–Newman barcode.
- **New lower bounds on Λ from collisions.** Every collision we found (lowest 46 zeros) is at
  t < 0; for example t* ≈ −0.3846 numerically coincides with the reported −0.385 bound (pair
  used not confirmed). Rodgers–Tao's Λ ≥ 0 supersedes all such bounds.
- **Numerical searches for positive-time folds.** Such a search is an RH counterexample search.
  RH has already been verified numerically to large height (3,000,175,332,800, Platt–Trudgian
  Theorem 1; primary, via review 5486879914). That verification does not certify this packet's
  `nzeros` calls.
- **The theta/modular route.** At L = 24 the periodization is invisible (about 10⁻¹²⁵ in the
  jets); ζ needs a Mellin transform over L, which the project disclaims; some Epstein zetas
  (Z_4; binary forms with class number > 1) have zeros off their symmetry line, so theta
  structure plus a functional equation selects nothing; and a formal
  theta bridge in Lean would add nothing (Mathlib's official documentation for
  `RiemannZeta`, `AbstractFuncEq` and `JacobiTheta.TwoVariable` shows ζ's functional equation
  and the theta/Mellin infrastructure [primary, via review 5486879914; the local Mathlib
  version and proofs were not checked], and the Math- Side24 Lean scratch already uses
  Mathlib's Poisson/theta lemmas).
- **Looking for new theta-derived restrictions on the zeros of H_t.** Restrictions of the kind
  the note proposes exist already (de Bruijn's strip lemma, Newman, Ki–Kim–Lee, Rodgers–Tao;
  §1.2), and the question is outside the project's competence and mission.
- **Applying the project's Gaussian theorems to ζ or H_t.** They are random-field results with
  admission interfaces. ζ is deterministic and is not admitted.
- **A covariance-matched Gaussian null model with covariance Ξ(z/2)/Ξ(0).** It exists by
  Bochner's theorem (Φ is even and positive), and the Math- line theorem's (R1)–(R3) probably
  apply, but its zero set has constant density and encodes nothing about ζ's zeros.
- **Using real-axis persistence to detect off-line zeros.** Counting real zeros against N(T)
  from the argument principle (Turing's method) already does this, and better; it is how RH
  is verified numerically. Real-axis barcodes alone discard this information: on [−L, L],
  H = 1 + L²/3 and H = z² + 1 have increasing antiderivatives, (1 + L²/3)z and z³/3 + z, with
  the same endpoint values, hence identical barcodes (one essential bar, no finite bar), yet
  H = z² + 1 has zeros ±i (example taken from main#307 comment 6116121839, §2).
- **Fitting exponents at moderate lengths or on small windows.** Fits over 0.1–1 × median give
  misleading intermediate values (−0.21 for CUE N=128), and windows of about 110 bars cannot
  estimate an exponent.
- **Calling the persistence phrasing of collisions new.** It is Cerf theory and vineyards; the
  local fold algebra is not new either, and §4 claims no novelty.
- **Treating the existing zeta/CUE/Gaussian runs as confirmations.** The data were inspected
  first. Any confirmation needs a frozen protocol on fresh data (§6.2).
- **Citing or reviving `history/2025/`.** That retired chapter contains a tautological
  critical-line result and describes itself as obsolete and wrong.
- **Committing the forwarded note verbatim or naming its author.** Its claims are paraphrased
  in `CLAIMS.md`; the note is referred to only as an external note forwarded by the project
  owner on 12 October 2026.
- **RH-adjacent framing on the public site or reader home before nonauthor review.** Packet and
  PR titles use "zeta zeros" and "de Bruijn–Newman flow".
- **The d ≥ 2 Epstein analogues, or a modular-symmetry version of the persistence picture,**
  without a concrete project question.
  Neither has a link to the admission problem.

## 8. Reproduce

Run from this directory. Requirements are pinned in `requirements.txt`: `numpy==2.5.3` (the
same pin as `experiments/periodic_h0`) and `mpmath==1.4.1`. The runs here used Python 3.13.16.
Pillow is optional, pinned separately in `requirements-plot.txt` (as `experiments/periodic_h0`
pins matplotlib), and only rasterizes PNG previews (`pip install -r requirements-plot.txt`,
then `stats/make_plots.py --png`).

**Tests.** The exact tests need only the standard library; the numeric tests skip unless numpy
and mpmath are importable and take a few seconds.

    python3 -B -S -m unittest -v test_exact          # stdlib only
    python3 -B -m unittest -v test_exact test_numeric
    # from the repository root:
    python3 -B -E -S -m unittest discover -s experiments/zeta_fold_exploration -p 'test_exact*.py'

The repository's own `python3 -B -S -m unittest discover -s tests` does not enter
`experiments/`, and the script directories here are not packages, so test discovery does not
pick up the scripts. `python3 -B make_manifest.py --check` verifies every committed file
against `MANIFEST.json`.

**Scripts.** Each script finds its inputs and outputs relative to its own file, so it can be
run from anywhere. Small outputs go to `results/<dbn|theta|stats>/`; large samples go to
`results/raw/` (set `ZETA_FOLD_RAW` to use another directory). Where a script prints to stdout,
the saved `.out` file named in its docstring is that stdout. Approximate runtimes:

- `dbn/check_dbn_basic.py` (~90 s), `dbn/check_collision.py` (~95 s) then
  `dbn/check_collision_persistence.py`.
- `dbn/validate_H0.py`, `dbn/count_check.py 1 270 -0.39` (and the other five count runs),
  `dbn/refine_check.py` (~30 s), `dbn/indep_collision.py` (seconds), `dbn/kappa_check.py`,
  `dbn/kappa_indep_degree6.py`, `dbn/kappa_indep_random.py`, `dbn/zero_velocity.py` (seconds),
  `dbn/smoke_hflow.py`.
- `dbn/track.py 270` (~3 min) and `dbn/track.py 103 -40 0.02 low10` (minutes), then
  `dbn/collision_table.py` (writes `results/dbn/collision_events.csv` from the two event logs);
  then `dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35` and
  `dbn/fold_check.py -1.8779090730 99.3284693056 pair9_10` (seconds each), then
  `dbn/fold_analyze.py pair34_35 pair9_10` (also writes `fold_tau_r_ell.csv`).
- `dbn/track.py 103 -40 <DT> dt<DT>` for DT = 0.01, 0.02, 0.025, 0.04, 0.05 (minutes each),
  concatenated into `track_low10_dt_sensitivity.out`.
- `dbn/indep_ht.py 60 xi` (seconds), `dbn/indep_ht.py 60 c910` (~2 min),
  `dbn/indep_ht.py 60 c3435` (~7 min),
  `dbn/lobe_identity.py` (~8 min).
  The `60 c910` and `60 c3435` commands above, and in the script's docstring, do not reproduce
  the committed `indep_ht_c910.out` and `indep_ht_c3435.out`:
  - **Current script.** With `dbn/indep_ht.py`, `60 c910` matches every line except the
    iteration-4 step. `60 c3435` does not meet its stop rule within 12 iterations, although its
    final `t*`, `z0`, A, B/A, C/A and κ lines match.
  - **Where the committed files come from.** They are the stdout of the review-math run of the
    earlier script, blob `c561d95b`. That run is retained, with the script, under
    `recovery/session_01S16677s1SGeAJmN3QMz4ip/zeta_raw/review-math/` (commit `5d298b9e`).
  - **Exact reproduction.** With Python 3.13.16, mpmath 1.4.1 (python backend) and
    `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `python -B <c561d95b> 55 c910` and
    `… 90 c3435` reproduce the two files exactly apart from `time`.
  - **Limit.** The original command lines were not recorded, so this is a sufficient
    invocation, not a log (§9).
- `theta/check_theta.py` and `stats/check_gue_transfer.py` (seconds),
  `theta/project_theta_checks.py` (~7 min).
- `python3 -B experiments/universality/fold_controls.py`, run from the repository root
  (seconds; a repository script outside this packet); its stdout is
  `results/theta/fold_controls_rerun.out`.
- `stats/sample_Z.py 600 900 0.02` and `stats/sample_Z.py 5000 5200 0.015` (several minutes
  each), then `stats/check_Z_persistence.py Z_grid_600_900.npz` and the same for the 5000 window.
- `stats/rs_zeta.py validate`, `stats/rs_zeta.py A 1e4 1e5`, `stats/rs_zeta.py B 1e5 1e6`,
  `stats/gauss_proc.py 500 4`, `stats/cue.py 12 86016 4`, `stats/cue.py 128 8064 4`, then
  `stats/analyze.py`, `stats/matched_tail.py`, `stats/make_plots.py` (about 7 minutes in
  total; `analyze.py` stdout duplicates `summary.json` and is not kept). The sample generators
  use a process pool, so set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to avoid BLAS thread
  oversubscription (without it `cue.py 128` ran more than ten times slower here); the same
  setting also makes the regenerated samples byte-identical.
- Review cross-checks on the same samples: `stats/check_gauss.py`, `stats/check_zeta_quad.py`,
  `stats/barcode_bruteforce.py`, `stats/gauss_hidden.py`, `stats/localamp_baseline.py`,
  `stats/cue_gap_shape.py`, `stats/rs_spot_check.py`, `stats/cue_endpoint_check.py`,
  `stats/block_sensitivity.py` (each under a minute).

**Reproduction status in this packet.** After moving the scripts here, the following were
re-run and gave byte-identical outputs (apart from timing fields): `check_collision`,
`check_collision_persistence`, `check_dbn_basic`, `refine_check`, `indep_collision`,
`track.py 270`, `track.py 103 -40 0.02 low10`, `collision_table`, `fold_check` for both pairs
(CSV, JSON and stdout), `fold_analyze` (including `fold_tau_r_ell.csv`), `validate_H0`,
`count_check` (window [1, 103] at t = −1.89), `check_gue_transfer`, `analyze`, `matched_tail`,
`make_plots`, `check_zeta_quad`, `barcode_bruteforce`, `gauss_hidden`, `localamp_baseline`,
`rs_spot_check`, `fold_controls_rerun`. The re-run of
`theta/check_theta.py` and `theta/project_theta_checks.py` reproduced every scripted line
(the earlier logs had one extra ad hoc line and a shell footer, now gone);
`check_Z_persistence` now also prints the exact `nzeros` counts. Raw `.npz` samples (about
200 MB) are not committed; their SHA-256 values from the exploratory run are listed in
[`MANIFEST.json`](MANIFEST.json). Regenerated in a scratch copy with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, the seven raw samples other than the two Z_grid
files (not regenerated) — `zeta_A`, `zeta_B`, `gauss`, `cue12`, `cue128`, `rs_cheb_coeffs` and
`ecdf_samples` — were byte-identical to the recorded hashes, and `summary.json`, the ECDF CSVs,
`matched_tail.json`, `local_slopes.csv` and both SVGs were reproduced. An earlier regeneration
without thread pinning gave `zeta_A.npz` values differing at relative 10⁻¹⁰ (BLAS summation
order), with identical counts, so byte-identity is expected only with thread pinning.

## 9. Review status

A nonauthor read from a non-Anthropic provider was requested, in four slices:

- **A.** Proposition 1 and the κ coefficients. **Read: AMEND** (review 5486843735; the local
  normal form and all three κ coefficients check; findings applied, §10.1). **Delta readback:
  PASS (scoped)** (review 5487707101 at `0969b327`).
- **B.** `CLAIMS.md` and the citations, ideally by a lane with working arXiv/journal access,
  confirming every item marked [snippet] or [memory], and the statement of Lagarias–Montague
  (arXiv:1106.4348). **Read: AMEND** (review 5486879914; the core classical claims are
  supported, citation scopes amended and provenance retagged, §10.2). The originals it did not
  read or audit stay unverified: Saouter–Gourdon–Demichel 2011, the 1988–1992 lower-bound
  papers, Davenport–Heilbronn 1936, de la Vallée Poussin 1896, Jacobi, Mumford, Cerf, Gonek,
  Hejhal and the 2009 Ki–Kim–Lee full proof; Edwards 1974 was identified bibliographically
  only, its coefficients being left to slice C. **Delta readback: PASS (scoped)** (review
  5487692150 at `0969b327`).
- **C.** The numerics, re-run independently, with zero counts checked against `nzeros` and a
  different seed or an independent Riemann–Siegel check. **Different-provider replay
  PERFORMED; record PASS (scoped).** Cursor/xAI Grok 4.7 replayed all 59 commands from the
  source-bound packet at `5cab7663`; all exited 0, 50 committed outputs were byte-identical,
  14 were timing-only, two residual trails differed, and all nine raw-sample hashes matched.
  The replay re-executed the packet's own `mpmath.siegelz`/`zetazero`/`nzeros` comparisons with
  the packet's same fixed seeds. It added neither a different seed nor an independent
  Riemann–Siegel check beyond the packet's existing validation. See the
  [replay delivery](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6282286422)
  and exact-head nonauthor [PASS_RECORD_SCOPED review](https://github.com/d6g8k5htny-coder/main/pull/458#pullrequestreview-5581698876).

  **Timing-only outputs re-run with retained bytes: PASS (scoped).** The replay did not retain
  13 of its 14 timing-only byte streams (the 14th, `rs_spot_check.out`, was hash-reconstructed).
  Its own performer, [xAI Grok 4.7](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304356801)
  (Cursor agent `bc-8084f8ef`; source-exposed, organizational-independence credit 0), re-ran
  the 14 commands behind those outputs in the pinned environment from `main` `157b6a97`, where
  the scripts run have the same blobs as at `5cab7663`. The bytes are retained on draft
  [main#469](https://github.com/d6g8k5htny-coder/main/pull/469) at `29b2bfef`, a
  retention-only record not meant for merge. Under the published mask
  (`reviews/zeta_fold_slice_c_rerun_20261019/comparison/timing_mask_rules.txt`) all 14 equal
  the committed outputs, and the only unmasked differences are timing tokens.
  The [hash-table correction](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304367431)
  supersedes the first table, and an [Anthropic recomputation from the bytes](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304374360)
  (same provider as the packet author, no independence credit) agrees with it. This is a new
  execution of the same programs and fixed inputs, not the replay's own streams, and it makes
  no scientific acceptance.

  **Residual trails reproduced exactly.** The replay's two differing files are
  `results/dbn/indep_ht_c910.out` and `indep_ht_c3435.out`. At the documented dps 60,
  `c910` matches every line except the iteration-4 step, and `c3435` differs from
  iteration 3 on and does not meet its stop rule. The final `t*`, `z0`, A, B/A, C/A
  and κ lines are identical.
  - **Provenance.** The committed files are the same git blobs as the review-math outputs kept
    in the packet author's recovery checkpoint, under
    `recovery/session_01S16677s1SGeAJmN3QMz4ip/zeta_raw/review-math/` (commit `5d298b9e`).
    Those were written by the earlier script `c561d95b`.
  - **Mechanism.** That script sets the working precision before it computes the subinterval
    constant `U = mpf('1.3')`. The current script (`736760f2`) computes `U` at 60 digits first.
    At dps 60 the two scripts compute the same `U`. At another requested precision the
    breakpoints differ, and the data note records that this shifts the quadrature noise in
    the iteration-4 step.
  - **Reproduction.** In the pinned environment, `c561d95b` reproduces both committed files
    exactly, apart from `time`, as `55 c910` and `90 c3435`. These are the only matches in
    the ranges tested: dps 50–70 for `c910` and dps 85–100 for `c3435`.
    ([data note](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6322382147);
    outputs retained on draft [main#471](https://github.com/d6g8k5htny-coder/main/pull/471) at
    `95a00ce4`.)
  - **Separate fact.** With the current script, the documented `60 c3435` does not converge
    ([6304057120](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304057120)).
  - **Limits.** The original command lines were never recorded, so this is a sufficient
    invocation, not a log of what was run. The reported values (`t*`, `z0`, A, B/A, C/A, κ)
    were never affected. The diagnostic is Anthropic, the same provider as the packet author,
    with organizational-independence credit 0.
  - **Non-Anthropic re-execution.** A nonauthor
    [xAI Grok 4.7 re-execution](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6322666857)
    (Cursor agent `bc-8084f8ef`; source-exposed, organizational-independence credit 0) ran
    `c561d95b` at `55 c910` and `90 c3435` in the pinned environment. Each output differs from
    the committed file only in its `time` line. It also confirmed that the committed files are
    the review-math blobs.

  **Additional fresh-seed evidence REPORTED; record applicability PASS (scoped).**
  Separately, [OpenAI GPT-6's R2a report](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6279371506)
  records a separate summation implementation with seed **31415927**, 192 points against
  `mpmath.siegelz`, and four fresh zero-count windows at coarse and half-step grids agreeing
  with `mpmath.nzeros`. The nonauthor [record-applicability assessment](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6289356717),
  performed by OpenAI Codex `/root`, Capability Frontier observer (CAPFRONT), was source-exposed,
  not blind: the reviewer was not the packet author or R2a executor, but shared R2a's OpenAI
  provider and the project's GitHub principal. That assessment
  accepts this as reported additional different-seed evidence for this component, not a
  new execution or independent verification of those numerical outputs. R2a used
  Python 3.13.5 / numpy 2.3.5 / mpmath 1.3.0, not the pinned environment, and did not execute
  the exact `rs_zeta.py` bytes. Its raw JSON was not uploaded with the report or retrieved
  by the applicability reader; its reported hash and results were not independently
  verified there, and custody elsewhere remains unknown. This evidence has organizational-
  independence credit 0 and supplies no uniform error bound, interval certificate,
  zero-completeness proof or scientific acceptance. Reimplementing the quoted formula does
  not verify it against a printed source; that comparison is recorded below.

  **R2a independently re-executed: PASS (scoped).** A nonauthor
  [xAI Grok 4.7 run](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6302968686)
  (Cursor agent `bc-8084f8ef`, neither Anthropic nor OpenAI, not R2a's executor;
  source-exposed, organizational-independence credit 0) re-executed R2a's described method
  in the pinned environment (Python 3.13.16 / numpy 2.5.3 / mpmath 1.4.1). At R2a's printed
  precision (`%.6e`), five of the nine maximum errors match. The other four — the three-term
  entry in each of the three windows and the two-term entry at 900,000–980,000 — are lower by
  7, 6, 1 and 1 units in the last printed digit. Against R2a's printed floats those four
  differ by 6.66e-16, 6.31e-16, 6.29e-16 and 1.24e-15. The largest absolute gap among all nine
  is 4.85e-12, on the one-term entry at 150,000–240,000, which still prints the same digits.
  Every max |Z| matches R2a's three decimals, and all eight zero counts match exactly. The
  fit-node residual scalars agree in size but not in printed digits (relative gaps 2.3%, 6.2%
  and 1.2%, at the 1e-15 to 1e-17 level). On the same draws, `stats/rs_zeta.py` (blob
  `8b97a1f6`) gives per-window maximum errors within 1.78e-15 of the separate sum's. Its
  result JSON (SHA-256 `36d21857…f1816a4`) is posted in that comment. R2a's own
  raw JSON remains unpublished and unverified. These are observed differences, not error
  certificates, and the re-execution makes no scientific acceptance.

  **Coefficients compared with Siegel 1932: PASS (scoped).** A nonauthor
  [xAI Grok 4.7 read](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6294430745)
  (Cursor agent `bc-8084f8ef`, source-exposed, organizational-independence credit 0)
  compared `stats/rs_zeta.py`'s Ψ, C0–C2, expansion variable, prefactor and main sum with
  Siegel's eqs. (44)–(45), as printed in the Barkan–Sklar English translation
  (arXiv:1810.05198, p. 17; *Gesammelte Abhandlungen* I p. 290). It found them in agreement
  after the change of variables δ = √(2π)(p − ½), with an independent 80-digit check. It
  read neither Edwards's pages nor Siegel's German original. The translators state that
  their German is limited.

  **Citation corrected: the packet cites the source actually checked.** By Dylan Roy's
  [decision of 20 October 2026](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6295674043)
  ([wording](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6295755538)), the
  packet cites Siegel 1932 for these coefficients instead of Edwards 1974. Edwards stays in
  `CLAIMS.md` as unverified background reading; its pages were not read, and nothing here
  claims that Edwards was verified. The docstring of `stats/rs_zeta.py` still names Edwards. That
  file is left byte-unchanged so that the replay and R2a records stay bound to its blob
  `8b97a1f6`. The replay's two residual-trail discrepancies now have the exact reproduction
  above. Its own 13 timing-only streams stay unretained (their classification now rests on the
  retained re-run above), and R2a's own raw JSON is unpublished; no record closes all of
  slice C.
- **D.** A wording audit of the non-claims. **Read: AMEND** (the same review, §10.1). **Delta
  readback: PASS (scoped)** (review 5487707101, with slice A).

The slice A/D read is OpenAI dot / Codex (`/root/review_zeta_fold_mathematics`), delegated AI
review for Dylan Roy: nonauthor and source-exposed, not blind, with organizational-independence
credit 0. It is static inspection plus independent symbolic arithmetic. It is not a numerical
reproduction, a citation-wide audit or an acceptance. Its delta readback of the applied
amendments, review 5487707101 at `0969b327`, is PASS (scoped), with one non-blocking note
(§10.1). The slice B read is OpenAI dot / Codex (`/root/verify_zeta_packet_citations`),
delegated AI review for Dylan Roy, likewise nonauthor and source-exposed, not blind, with
organizational-independence credit 0. It inspected public primary papers, author manuscripts
and official documentation (two scanned papers by local text/OCR inspection), and ran no
repository program, test, numerical pipeline or Lean proof. It is a
bounded citation comment, not approval, readiness, numerical reproduction, a full literature
search, a novelty determination or theorem-wide acceptance. "Supported" there means that a
source supports the stated claim within its hypotheses, not that its proof was reconstructed.
Its delta readback, review 5487692150 at `0969b327`, is PASS (scoped). Neither delta readback
is an acceptance or a numerical reproduction. The same-provider passes (§10.3) were
Anthropic Claude reading Anthropic Claude work and earn no organizational-independence
credit.

The slice C performer reported Cursor/xAI Grok 4.7; the record review was a separate OpenAI
Codex lane, source-exposed and not blind, with organizational-independence credit 0. The
review checked the 59-command multiset, hashes, counts, environment, discrepancies and packet
binding at exact replay head `78976ae7`. It did not independently reapply timing masks to 13
timing-only replay byte streams that were not retained. The replay used the same programs and
fixed inputs: it is not a second implementation or a scientific acceptance. Both residual
trails were later reproduced exactly with the earlier review-math script (item C). The original
command lines were not recorded, so that is a sufficient invocation, not a log. Scientific
effect stays NONE.

The packet was integrated into main as `3e589ae0` on 12 October 2026 by a third OpenAI dot /
Codex lane (`/root/integrate_zeta_exploration_packet`), distinct from both reviewer lanes. The
automatic Codex review did not run (NOT_RUN, quota).
Integration changes no scientific status: the scientific effect stays NONE.

## 10. Review findings applied

### 10.1 Nonauthor review 5486843735, slices A and D (applied in v2)

| # | Finding | Where applied |
|---|---|---|
| ZF-AD-01 | A C¹ weight does not give ℓ_w = w(x₀)ℓ(1 + O(r²)); counterexample w = 1 + \|x\|^{1+η} | §4 item 3: C^{1,1} with controlled asymmetry, exact τ-coefficient, only 1 + o(1) for C¹, the counterexample, uniform weight and marked-measure hypotheses for any transfer; §6.3; `test_exact.py` |
| ZF-AD-02 | Finite-window estimates stated as a settled exponent and a calibrated total-error result | In brief item 3; bottom-line table; §1.4; §5.3 (Hardy-Z primitive named, controls as theoretical expectations, both readings); §6.1; §7; `CLAIMS.md` 5.4 (limiting collision-mark measure; r⁴ example in `test_exact.py`); `stats/check_gue_transfer.py` docstring (product model); `results/stats/effective_exponent_bars.svg` labels (finite windows, (D1) expectations) |
| ZF-AD-03 | Proposition 1(c) domain, conventions, A < 0 and scope; exact r² = 8τ | §3: compact I in the analytic domain with z* ∈ I, barcode convention, simultaneous reflection, local-bar scope, Remark 2 r² = 8τ(1 + O(τ)); In brief item 1; `CLAIMS.md` 3.13 |
| ZF-AD-04 | Real-zero ODE asserted at t = 0 without hypotheses | §1.2 and `CLAIMS.md` 3.4: simple-zero velocity in general, real-zero sum for t > Λ, RH at t = 0 (simplicity of z_j stated separately, not as part of RH), the 300-zero comparison a consistency check; docstrings of `dbn/check_dbn_basic.py` and `dbn/zero_velocity.py` |
| ZF-AD-05 | Conjecture Z's limit and observable; log-CDF versus density; T-CUE strata | §6.2: iterated limit over all T with the window fixed as H(T) = √T, amplitude of every bar, boundary, positivity, which median, the 2^{−n} example, "well posed" withdrawn; §6.1: nonlocal strata and global selection; `CLAIMS.md` "What is kept" |
| Info | Λ > 0 forces a finite multiple real zero (Polymath15 Theorem 1.5(i)) | `CLAIMS.md` 6.1, 6.2 and error 5; §3 Remark 3; §7 |
| Delta note | Non-blocking note in the delta readback (review 5487707101): §6.2 stated μ((0, ℓ]) ~ c(T) ℓ at finite T, but a finite empirical measure has no positive ℓ → 0 asymptotic below its shortest bar | §6.2 coefficient form: a finite-tail approximation fitted on a declared range [ℓ_lo(T), ℓ_hi(T)] of fixed cutoffs (not the ℓ₀ of Conjecture Z (i)), outside Conjecture Z (applied after integration; not read back) |

### 10.2 Nonauthor review 5486879914, slice B (applied in v2)

| # | Finding | Where applied |
|---|---|---|
| ZF-B-01 | Selberg's CLT (log\|ζ\| at uniformly sampled heights) does not give the selected local-amplitude law or coefficient drift | §6.2 coefficient form (Radziwiłł–Soundararajan Theorem 1; the amplitude law and drift stated as heuristic); §6.1 risk; `CLAIMS.md` provenance |
| ZF-B-02 | HKO Theorem 1.2 (CUE, boundary Re k > −3/2) is distinct from their ζ Conjecture 1.3 (RH; simple zeros for negative moments); a(k) is part of the moment model | §6.2 (analogy only; proves neither our mark's negative moment nor drift; Gonek and Hejhal originals not audited); `CLAIMS.md` provenance |
| ZF-B-03 | "The heat flow destroys the Euler product" is broader than the evidence | `CLAIMS.md` 4.6: b_{mn}^t/(b_m^t b_n^t) = exp((t/2) log m log n) ≠ 1 for the deformed series of Polymath15's approximation; not a theorem about every Euler product of H_t |
| ZF-B-04 | Ki–Kim–Lee's conclusions are supported, but the "ζ-specific arithmetic input" classification of Ki–Kim–Lee and Rodgers–Tao is not | §1.2 (classification removed; Rodgers–Tao's contradiction via local equilibrium and known zero statistics; Ki's report and its extensions); `CLAIMS.md` 3.6, 3.8, 7(b), provenance (2009 full proof not audited) |
| ZF-B-05 | Statistical sources need explicit scopes | §1.4 (Montgomery under RH; Rudnick–Sarnak Theorem 1.2, Σ\|ξ_j\| < 2; bulk sine-process limit; nearest-neighbour density a different observable; 32/π² the Wigner surmise); §5.3 and §6.2 (BBLM model "suggests" N_eff; Keating–Snaith a matching rule in their model); `CLAIMS.md` 5.1, 5.3, provenance |
| Provenance | Supported source checks 1–9 and residual originals | Three-term vocabulary (§1, `CLAIMS.md` top); locators and links per item in `CLAIMS.md`; Lagarias–Montague scaling and primitive-zeros scope (§4 item 3, `CLAIMS.md` 5.6); Titchmarsh ch. III §3.1; CSV's sufficient condition (`CLAIMS.md` 3.10); Platt–Trudgian not certifying `nzeros` (§7, `CLAIMS.md` 6.3); residual originals left unverified (§9) |

### 10.3 Same-provider passes (applied in v1)

Two same-provider review passes (mathematics; numerics) read draft v0. Every finding was
applied:

| # | Finding (severity) | Where applied |
|---|---|---|
| M1 | "All collisions at t < 0" was stated as fact (high) | "Every collision we found (lowest 46 zeros) is at t < 0": §5.2, §7 |
| M2 | "Numerically decisive" overstated; CIs sampling-only (high) | Bottom-line table, §5.3 control-comparison table and reading |
| M3 | §4 called "new" (medium) | §4 retitled; novelty not claimed |
| M4 | Cumulative (D2) unamended; (D1) vs §D1/§D2 labels (medium) | §2, §6.1 fallback cites `CUMULATIVE_TRANSFER_CORRECTION.md` |
| M5 | Theta transformation attributed to Riemann (medium) | §1.3 |
| M6 | β used in two senses (medium) | α for the gap/pair exponent throughout; β only in the Math- sense (§2, §5.3) |
| M7 | "Under RH every fold is at t < 0, accumulating at 0⁻" (medium) | §3 Remark 3 |
| M8 | Conjecture Z ill-posed (medium) | §6.2 restated |
| M9 | ζ result called "exactly" the transfer theorem (medium) | §5.3 reading |
| M10 | Kernel-generic vs ζ-specific literature (low) | §1.2 (the ζ-specific classification was later withdrawn, ZF-B-04) |
| M11 | "Reproduced" collisions; −0.385 match (low) | "computed", §5.1–5.2, §7 |
| M12 | Φ positivity wording (low) | §1.2 |
| M13 | Zero ODE summation; exact velocities (low) | §1.2, `dbn/zero_velocity.py` |
| M14 | Davenport–Heilbronn, Z_4, domains (low) | §1.3, §4 item 1 |
| M15 | Hypotheses 1–4 consistently (low) | §2, §6.1 |
| M16 | w^{−1} weight; O(r) (low) | §4 item 3, §6.3 |
| M17 | "time derivative", "later events", "explained by" (low) | §4 item 4, §5.2, §5.3 |
| N1 | "Decisive ≈ 0"; control biases (medium) | §5.3 |
| N2 | 10% amplitude baseline (medium) | §5.3 short-bar composition |
| N3 | Zeta gap shape vs CUE N; matching rule (medium) | §5.3, §6.2 |
| N4 | CUE cut convention vs T-CUE (medium) | §5.3 Samples, §6.1, `stats/cue.py`, `stats/cue_endpoint_check.py` |
| N5 | τ-convergence is generic (low) | §5.2 |
| N6 | Collision accuracy statements (low) | §5.1; `refine_check.out`, `indep_collision.out` committed |
| N7 | Matched-tail limits (low) | §5.3 |
| N8 | 17 unresolved Gaussian cells (low) | Report wording, §5.3; `stats/gauss_hidden.py` |
| N9 | Block bootstrap (low) | Sensitivity run, §5.3 |
| N10 | Main-term counts; endpoint conventions (low) | §5.4, §5.3; `check_Z_persistence.py` prints `nzeros` |
| N11 | Barcode, RS and completeness confirmed (info) | §5.3; brute-force barcode test in `test_numeric.py` |
| N12 | Tracker completeness; monotonicity (info) | §5.2 |

Two further same-provider checks (accuracy, 16 findings A1–A16; reproducibility, 6 findings
R1–R6) were then applied, followed by a third same-provider round (accuracy, wording overlap
with the note, and reproducibility/CI; 18 findings, some overlapping). None of these earns
independence credit.
