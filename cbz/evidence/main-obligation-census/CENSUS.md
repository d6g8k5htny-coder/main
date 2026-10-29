# Census of main's proof obligations

Read at `main` `81acc060341e8a31cc4cd4f9523ef71fb6c06175`. These are draft rows for Root's `PROOF_INDEX.md`, and they are not dispositions. Every cited path, blob, line range and anchor below was checked mechanically against that revision by `build_census.py`. Each claim restates the document's own words; it is not a new mathematical review.

| ID | Draft disposition | Kind | Path |
|---|---|---|---|
| OBL-MAIN-01 | QUARANTINED | counterexample (negative result) | `docs/research-translation/20260930/cap-assumption-ablation/PROOF.md` |
| OBL-MAIN-02 | QUARANTINED | conditional analytic theorem, author-side | `experiments/periodic_h0/spectral_shrinking_bins/PROOF.md` |
| OBL-MAIN-03 | QUARANTINED | conventional analytic theorem | `experiments/universality/finite_candidate_lifetime/PROOF.md` |
| OBL-MAIN-04 | QUARANTINED | conventional analytic theorem | `experiments/universality/finite_contact_genericity/PROOF.md` |
| OBL-MAIN-05 | QUARANTINED | deterministic analytic theorem | `experiments/universality/finite_elder_transfer/PROOF.md` |
| OBL-MAIN-06 | QUARANTINED | analytic theorem, research preparation | `experiments/universality/finite_fold_chart/PROOF.md` |
| OBL-MAIN-07 | QUARANTINED | conventional analytic interfaces | `experiments/universality/finite_gaussian_contact/PROOF.md` |
| OBL-MAIN-08 | QUARANTINED | composition theorem | `experiments/universality/finite_h0_lifetime/PROOF.md` |
| OBL-MAIN-09 | QUARANTINED | conventional analytic estimate | `experiments/universality/finite_weighted_fold_window/PROOF.md` |
| OBL-MAIN-10 | QUARANTINED | structural theorem | `experiments/universality/fold_structural_universality/PROOF.md` |
| OBL-MAIN-11 | QUARANTINED | deterministic and probabilistic theorems | `experiments/universality/higher_homology_fold/PROOF.md` |
| OBL-MAIN-12 | QUARANTINED | process limit theorem | `experiments/universality/iid_short_bar_process/PROOF.md` |
| OBL-MAIN-13 | QUARANTINED | admission theorem for one law | `experiments/universality/periodized_gaussian_h0/PROOF.md` |
| OBL-MAIN-14 | QUARANTINED | exact regime theorem | `experiments/universality/spectral_amplitude_crossover/PROOF.md` |
| OBL-MAIN-15 | QUARANTINED | admission theorem | `experiments/universality/spectral_gaussian_admission/PROOF.md` |
| OBL-MAIN-16 | QUARANTINED | coefficient evaluation with outward rational bounds | `formal/sources/side24_v1/PROOF.md` |
| OBL-MAIN-17 | QUARANTINED-ARCHIVAL | incoming custody copy: lemma package | `incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma I Package.md` |
| OBL-MAIN-18 | QUARANTINED-ARCHIVAL | incoming custody copy: adjudication | `incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma MODZONE CountLaw Package.md` |
| OBL-MAIN-19 | QUARANTINED-ARCHIVAL | incoming custody copy: theorem table | `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/RETURN_06/THEOREM_TABLE.md` |
| OBL-MAIN-20 | QUARANTINED | review with HOLD disposition | `reviews/lb_rate_thm023_landing_20260926/REVIEW.md` |
| OBL-MAIN-21 | CLOSED-CANDIDATE | Lean 4 targets (31) | `formal/manifest.json` |

## Prose and archival obligations

### OBL-MAIN-01: `docs/research-translation/20260930/cap-assumption-ablation/PROOF.md`

- **Blob** `7158f3788ad9`, 255 lines. **Kind:** counterexample (negative result).
- **Claim:** The interior-height hypothesis C1 of the D1 cap theorem cannot be dropped, even keeping C2, C3, every pin jet and arbitrarily good uniform closeness, for smooth Morse functions with distinct critical values on a fixed compact torus.
- **Statement:** lines 15-51; lines 231-255. **Stated limits:** lines 3-8; lines 244-251.
- **Reviews:** No adjacent review record. The text describes itself as one delegated AI derivation.
- **Inputs:** D1's cap theorem (PR196, AUDIT.md), which this note does not re-audit.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Only the explicit perturbation family of sections 2-4 is shown to break C1 while keeping C2, C3 and the pin jets. Nothing here refutes or accepts the D1 theorem. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-02: `experiments/periodic_h0/spectral_shrinking_bins/PROOF.md`

- **Blob** `1a4b544d018d`, 452 lines. **Kind:** conditional analytic theorem, author-side.
- **Claim:** A finite error bound (5) and its asymptotic consequence (7) for shrinking lifetime bins of grid samples coupled to a Gaussian Fourier field, under the imported hypotheses D, C and A.
- **Statement:** lines 15-125; lines 164-206. **Stated limits:** lines 441-452.
- **Reviews:** Self-declared at lines 3-4: source-bound nonauthor review pending. No adjacent review record.
- **Inputs:** The imported continuum density D and count C interfaces; the M4-M5 coefficient interface.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Conditional on D, C and A exactly as stated in section 1. No evaluated constant, grid threshold or historical-experiment certification. Unreviewed. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-03: `experiments/universality/finite_candidate_lifetime/PROOF.md`

- **Blob** `a27a84150b3e`, 577 lines. **Kind:** conventional analytic theorem.
- **Claim:** For each fixed finite Fourier Gaussian law (1) with K>=2, the ordered maximum-saddle candidate measure has a finite continuous density nu_cand(ell) ~ C_loc ell^(-1/3) with the explicit coefficient (4), and cumulative count (3/2) C_loc t^(2/3).
- **Statement:** lines 98-156. **Stated limits:** lines 154-156; lines 553-577.
- **Reviews:** REVIEW.md: nonauthor PASS within the requested scope and a no-must-fix full-source verdict (OpenAI/Codex, organizational independence 0).
- **Inputs:** External: the Math- parent P (UNIFORM_MATRIX_CAP_AND_LIFETIME.md at 13af1089) with its errata E1 and E2.
- **Links, from its own text:** OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Candidate pairs only: not actual persistence bars, no O(1) remainder, Gaussian coefficient law only (not the finite-word sampler). Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-04: `experiments/universality/finite_contact_genericity/PROOF.md`

- **Blob** `569a49dee2ec`, 473 lines. **Kind:** conventional analytic theorem.
- **Claim:** For each fixed law, target, basepoint and frame, almost surely under the canonical conditional law: the contact is an isolated critical point; every other critical point is Morse; the critical set is finite; the other critical values are distinct and differ from b.
- **Statement:** lines 97-120. **Stated limits:** lines 109-113; lines 455-473.
- **Reviews:** REVIEW.md: full mathematical PASS, a bounded PASS, and a new full-source PASS (OpenAI/Codex).
- **Inputs:** External: the frozen PR321 candidate and chart sources.
- **Links, from its own text:** OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** One fixed conditional law at a time; no statement over uncountably many targets or frames; the whole conditional field is not claimed Morse. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-05: `experiments/universality/finite_elder_transfer/PROOF.md`

- **Blob** `69f84eae401b`, 458 lines. **Kind:** deterministic analytic theorem.
- **Claim:** Under global C4 convergence, the exact finite pins (1) and the limit contact data (2) with finite Morse, distinct other critical values: for large j the pinned pair (M_j, S_j) is the actual finite H0 superlevel elder pair, with lifetime exactly k_j r_j^3.
- **Statement:** lines 116-154. **Stated limits:** lines 151-154; lines 441-458.
- **Reviews:** REVIEW.md: fresh full exact-file nonauthor PASS of the incorporated proof (OpenAI/Codex).
- **Inputs:** OBL-MAIN-04 supplies its hypotheses almost surely under the contact law.
- **Links, from its own text:** OBL-MAIN-04, OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The threshold may depend on the realization; no deterministic ensemble-wide threshold. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-06: `experiments/universality/finite_fold_chart/PROOF.md`

- **Blob** `6febb6e697a7`, 551 lines. **Kind:** analytic theorem, research preparation.
- **Claim:** Uniform derivative-controlled fold charts for C4 fields on the embedded box, under the pins, the fixed windows and condition (A).
- **Statement:** lines 53-134. **Stated limits:** lines 513-551.
- **Reviews:** REVIEW.md: no-must-fix successor verdict and a full-source PASS for the bounded analytic theorem (OpenAI/Codex).
- **Inputs:** Its stated assumptions; consumed by OBL-MAIN-09.
- **Links, from its own text:** OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Exactly the stated assumptions and windows; no compiled Lean application of these bounds (line 543). Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-07: `experiments/universality/finite_gaussian_contact/PROOF.md`

- **Blob** `5f345201309c`, 554 lines. **Kind:** conventional analytic interfaces.
- **Claim:** For positive finite Fourier Gaussian spectra (K>=2), the named contact, divided-difference, regression, compact-mark normalizer, envelope, off-diagonal and local-jet interfaces (sections 2-8), and a finite-r physical Hessian bridge (section 10).
- **Statement:** lines 10-93. **Stated limits:** lines 425-479.
- **Reviews:** REVIEW.md: two ACCEPT_SCOPED readings and a PASS of the sections 9-10 physical-interface scope (OpenAI/Codex).
- **Inputs:** External: the Math- parent P (UNIFORM_MATRIX_CAP_AND_LIFETIME.md at 13af1089) with E1, CAP and E2.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The ten catalogue spectra at L=24, K=2 or 3, and the general finite law (1). Not an admission to the full persistence theorem. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-08: `experiments/universality/finite_h0_lifetime/PROOF.md`

- **Blob** `7a3518edd9bd`, 526 lines. **Kind:** composition theorem.
- **Claim:** Consuming OBL-MAIN-05 for the same fixed law, the finite H0 elder-bar intensity has a density nu_bar ~ C_loc ell^(-1/3) with the candidate coefficient, and cumulative count (3/2) C_loc t^(2/3).
- **Statement:** lines 89-146. **Stated limits:** lines 495-526.
- **Reviews:** REVIEW.md: PASS readings of the preparation and fresh full nonauthor PASS readings of the incorporated proof; metadata AMEND requests are recorded at lines 107-118 (OpenAI/Codex).
- **Inputs:** The candidate, genericity and elder-transfer rows it composes (see its links).
- **Links, from its own text:** OBL-MAIN-03, OBL-MAIN-04, OBL-MAIN-05, OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The fixed finite law only; no numerical enclosure or angular/area factor. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-09: `experiments/universality/finite_weighted_fold_window/PROOF.md`

- **Blob** `f5473c0e4502`, 357 lines. **Kind:** conventional analytic estimate.
- **Claim:** With constants chosen from the uniform moments and normalizer floor, for 0<r<=r_* the weighted law gives Q^W(R<=T and |A|>=a_0) >= 1-epsilon, removing the deterministic fold-chart windows.
- **Statement:** lines 231-290. **Stated limits:** lines 322-357.
- **Reviews:** REVIEW.md: PASS, delta PASS and a new full-source PASS (OpenAI/Codex).
- **Inputs:** External: the compact-mark normalizer theorem.
- **Links, from its own text:** OBL-MAIN-06, OBL-MAIN-07. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The fold theorem's continuous parameter dependence remains conditional (line 339). Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-10: `experiments/universality/fold_structural_universality/PROOF.md`

- **Blob** `483d779cfdfd`, 804 lines. **Kind:** structural theorem.
- **Claim:** For a regular cubic fold on a compact manifold, the five interfaces COUNT, CONTACT, SELECT, ENVELOPE and FAR are sufficient for the H0 lifetime exponent -1/3, through a dimensional cancellation (section 6).
- **Statement:** lines 19-73; lines 379-449. **Stated limits:** lines 25-27; lines 691-750.
- **Reviews:** REVIEW.md: fresh full PASS of the original preparation and a fresh FULL PASS of the current file (OpenAI/Codex).
- **Inputs:** Its sufficiency statement uses no other row; the rows it links are adapters and examples. Admission for a given field is a separate obligation.
- **Links, from its own text:** OBL-MAIN-03, OBL-MAIN-04, OBL-MAIN-05, OBL-MAIN-07, OBL-MAIN-08, OBL-MAIN-13. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Sufficiency only. Verifying the five interfaces for a particular field is not done here; no higher-singularity class law. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-11: `experiments/universality/higher_homology_fold/PROOF.md`

- **Blob** `8e18a7900e7d`, 911 lines. **Kind:** deterministic and probabilistic theorems.
- **Claim:** Attaching obstructions and isolated-fold unit incidence in degree p for C4 fields with one isolated degenerate critical point; an actual Hp lifetime density (section 10) and a degree-marked iid-copy corollary (section 11).
- **Statement:** lines 25-78; lines 562-677. **Stated limits:** lines 755-796.
- **Reviews:** REVIEW.md: two FULL PASS readings of the preparation, a FULL PASS of the current proof, and a README wording AMEND that is retained (OpenAI/Codex).
- **Inputs:** Its section 13 binds the operational sources; SG is OBL-MAIN-15.
- **Links, from its own text:** OBL-MAIN-05, OBL-MAIN-10, OBL-MAIN-12, OBL-MAIN-13, OBL-MAIN-15. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Ordinary finite Hp intervals only; no extended, relative, zigzag or torsion-valued claim (line 440). Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-12: `experiments/universality/iid_short_bar_process/PROOF.md`

- **Blob** `cd7ef8b8a364`, 618 lines. **Kind:** process limit theorem.
- **Claim:** For the periodized Gaussian law (1), fixed d>=2 and L: a single field's short-bar process has an empty limit, while independent whole copies give a nonempty marked Poisson limit; with an explicit first-moment counterexample (section 8).
- **Statement:** lines 19-74. **Stated limits:** lines 519-618.
- **Reviews:** REVIEW.md: metadata PASS and scoped mathematical PASS readings (OpenAI/Codex).
- **Inputs:** The all-bar first moment IA (its text, lines 30-41).
- **Links, from its own text:** OBL-MAIN-05, OBL-MAIN-08, OBL-MAIN-10, OBL-MAIN-13, OBL-MAIN-16. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Independent whole copies of one fixed law; qualitative dominated convergence, not a certified numerical window (line 536). Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-13: `experiments/universality/periodized_gaussian_h0/PROOF.md`

- **Blob** `b27e2bc84839`, 637 lines. **Kind:** admission theorem for one law.
- **Claim:** For the exact periodized Gaussian law K_L (fixed d>=2, L>0), the five interfaces are derived and compose to the actual H0 lifetime intensity, with an exact covariance-derived coefficient (section 8).
- **Statement:** lines 17-84; lines 498-568. **Stated limits:** lines 569-637.
- **Reviews:** REVIEW.md: fresh full PASS of the preparation and a FULL PASS of the current file (OpenAI/Codex).
- **Inputs:** OBL-MAIN-16 for the L=24 instance (its lines 17-27).
- **Links, from its own text:** OBL-MAIN-03, OBL-MAIN-05, OBL-MAIN-10, OBL-MAIN-16. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The one law K_L; it does not import a finite-cutoff limit or a historical remainder. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-14: `experiments/universality/spectral_amplitude_crossover/PROOF.md`

- **Blob** `5d5b7e36beea`, 813 lines. **Kind:** exact regime theorem.
- **Claim:** Three amplitude regimes (sub-, critical and supercritical) with exact density and cumulative asymptotics and process limits for independent whole copies (the section 11 table).
- **Statement:** lines 669-712. **Stated limits:** lines 686-689.
- **Reviews:** REVIEW.md: two AMEND verdicts on the original preparation, retained; PASS on the corrected preparation; a fresh FULL PASS of the current file (OpenAI/Codex).
- **Inputs:** The stated base premises; see its section 1.
- **Links, from its own text:** OBL-MAIN-10, OBL-MAIN-11, OBL-MAIN-12, OBL-MAIN-15. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Independent whole copies only; no within-one-field Poisson statement or coalescing-contact factorial estimate. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-15: `experiments/universality/spectral_gaussian_admission/PROOF.md`

- **Blob** `72c896474073`, 826 lines. **Kind:** admission theorem.
- **Claim:** Admission of regular-fold H0 lifetimes and iid-copy processes for declared stationary Gaussian laws, with the concrete spectral verifier (P5)+(A4).
- **Statement:** lines 19-92. **Stated limits:** lines 706-755; line 698.
- **Reviews:** REVIEW.md: two FULL PASS readings, a FULL PASS of the current source and a metadata PASS (OpenAI/Codex).
- **Inputs:** The rows it links.
- **Links, from its own text:** OBL-MAIN-10, OBL-MAIN-12, OBL-MAIN-13. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** Fixed declared laws; family uniformity is a further condition; C4 regularity, not smooth samples. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-16: `formal/sources/side24_v1/PROOF.md`

- **Blob** `44b66f04f89f`, 216 lines. **Kind:** coefficient evaluation with outward rational bounds.
- **Claim:** Rational-endpoint enclosures 0.07340691930603427103 < c_2,24 < 0.07340691930603427104 and 0.04177593184059834334 < c_3,24 < 0.04177593184059834335 of the coefficient defined in the parent's equation (15.2).
- **Statement:** lines 6-31. **Stated limits:** lines 27-31; line 175.
- **Reviews:** Its own line 4 still reads 'nonauthor review open', but STATUS.md's D3 row records the review on main #65. That status disagreement belongs on the G8 status-truth list.
- **Inputs:** The Math- parent UNIFORM-MATRIX-CAP-LIFETIME (sha256 9350ad6e...), whose elder-selection and Kac-Rice proof this note does not accept.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The enclosures of the specified expression only; interpreting it as the finite-bar constant stays conditional on the parent. Its arithmetic skeleton is OBL-MAIN-21; the analytic steps are prose. Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A).

### OBL-MAIN-17: `incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma I Package.md`

- **Blob** `5ec0e33efe98`, 36 lines. **Kind:** incoming custody copy: lemma package.
- **Claim:** Its own text calls Lemma I 'resolved', but by measured or numerically verified components.
- **Statement:** lines 1-13. **Stated limits:** line 32.
- **Reviews:** None on main.
- **Inputs:** C007-C009 snapshots off main.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED-ARCHIVAL. **Boundary:** A historical incoming copy, not adopted by main. Measured or numerical components are not proof. Nothing on main depends on it.

### OBL-MAIN-18: `incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma MODZONE CountLaw Package.md`

- **Blob** `55c8712a17dd`, 134 lines. **Kind:** incoming custody copy: adjudication.
- **Claim:** OBL-BETA-MODZONE adjudicated FAILED-AS-WRITTEN on both routes and superseded by OBL-BETA-RELEVANCE.
- **Statement:** lines 1-14. **Stated limits:** lines 117-134.
- **Reviews:** None on main.
- **Inputs:** C023 snapshots off main.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED-ARCHIVAL. **Boundary:** A historical incoming copy of a failed-as-written obligation. Nothing on main depends on it.

### OBL-MAIN-19: `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/RETURN_06/THEOREM_TABLE.md`

- **Blob** `5a800ba09283`, 28 lines. **Kind:** incoming custody copy: theorem table.
- **Claim:** A graded table of carrier statements (EXACT, CERTIFIED INTERVAL, conservative bound, measured estimate, diagnostic), including conditional ones.
- **Statement:** lines 1-28. **Stated limits:** line 13.
- **Reviews:** None on main.
- **Inputs:** Carriers in its SOURCE_CAPSULE, off main.
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED-ARCHIVAL. **Boundary:** A historical incoming copy; its rows carry their own grades and are not adopted by main.

### OBL-MAIN-20: `reviews/lb_rate_thm023_landing_20260926/REVIEW.md`

- **Blob** `b612adab9048`, 140 lines. **Kind:** review with HOLD disposition.
- **Claim:** LB-RATE / KIMI-THM-023: the constant c = 0.9144 is not accepted as a proved uniform theorem constant.
- **Statement:** lines 1-12. **Stated limits:** line 8.
- **Reviews:** This file is the review: HOLD, a conditional assembly draft (xAI, organizational independence 0).
- **Inputs:** Its float-based verifiers under history/carrier20_recovery_20260928/ (01XJMJ's A3 census, issue 684 comment 6503611192).
- **Links, from its own text:** none. Links include references and examples as well as logical dependencies.
- **Draft disposition:** QUARANTINED. **Boundary:** The obligation is open and must not be indexed as CLOSED; its history/ verifiers are not A3-admissible certificates.

## OBL-MAIN-21: the 31 Lean targets in `formal/`

Each target's exact Lean statement, with the manifest's allowed axioms propext, Classical.choice and Quot.sound. CLOSED only once G3 (build), G4 (kernel recheck) and G2 (axioms) are observed in the final run. Root reports a fresh build, leanchecker, axioms and negative controls passing at c3377cc8 (issue 684 comment 6503255828), which is not yet the final run. Alignment of each target with its informal anchor stays PENDING_INDEPENDENT_REVIEW, and each target's does_not_claim text bounds it.

| Target | Module | Does not claim |
|---|---|---|
| `UniversalLaw.Side24.pairing_sum_six` | `UniversalLaw/Side24/Ledger.lean` | the derivative bound 76|x|^6 e^(-|x|^2/2) itself |
| `UniversalLaw.Side24.pairing_sum_le_six` | `UniversalLaw/Side24/Ledger.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.pairing_terms_six` | `UniversalLaw/Side24/Ledger.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.sixth_moment_double_factorial` | `UniversalLaw/Side24/Ledger.lean` | that Gaussian derivative contractions at zero are bounded by 15 |
| `UniversalLaw.Side24.lattice_shell_constant` | `UniversalLaw/Side24/Ledger.lean` | the lattice-point count or the geometric tail sum |
| `UniversalLaw.Side24.geometric_ratio_constants` | `UniversalLaw/Side24/Ledger.lean` | that the ratio is below 1/2 |
| `UniversalLaw.Side24.period_exponent` | `UniversalLaw/Side24/Ledger.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.image_constant_value` | `UniversalLaw/Side24/Ledger.lean` | the inequality |D^q K_24(0)-D^q K_infty(0)| < E |
| `UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten` | `UniversalLaw/Side24/Ledger.lean` | that the partial sum is below exp(288/125) (positivity of the omitted terms is analysis) |
| `UniversalLaw.Side24.covariance_relative_bound` | `UniversalLaw/Side24/Ledger.lean` | the spectral-norm bound 20E or C_ref >= I/3 |
| `UniversalLaw.Side24.odd_block_eigenvalues` | `UniversalLaw/Side24/Ledger.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.odd_block_shifted_minors` | `UniversalLaw/Side24/Ledger.lean` | Sylvester's criterion |
| `UniversalLaw.Side24.smaller_eigenvalue_exceeds_third` | `UniversalLaw/Side24/Ledger.lean` | monotonicity of the square root |
| `UniversalLaw.Side24.exponent_bounds` | `UniversalLaw/Side24/Ledger.lean` | the log/exp estimates that follow |
| `UniversalLaw.Side24.exponent_values` | `UniversalLaw/Side24/Ledger.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.ratio_constant_ordering` | `UniversalLaw/Side24/Ledger.lean` | the elementary log/exp bounds |
| `UniversalLaw.Side24.density_comparison_below_reported` | `UniversalLaw/Side24/Ledger.lean` | the multiplicative density comparison |
| `UniversalLaw.Side24.conditional_third_derivative_variance` | `UniversalLaw/Side24/ConeMoments.lean` | the Gaussian conditioning formula or the covariance derivatives |
| `UniversalLaw.Side24.transverse_covariance_entries_thirds` | `UniversalLaw/Side24/ConeMoments.lean` | the delta formula itself |
| `UniversalLaw.Side24.cone_moment_m1_thirds` | `UniversalLaw/Side24/ConeMoments.lean` | centred Gaussian symmetry |
| `UniversalLaw.Side24.trace_and_traceless_variances` | `UniversalLaw/Side24/ConeMoments.lean` | independence beyond zero covariance, which is Gaussian |
| `UniversalLaw.Side24.fourth_moment_of_trace` | `UniversalLaw/Side24/ConeMoments.lean` | the Gaussian fourth-moment identity |
| `UniversalLaw.Side24.exponential_moment_of_trace` | `UniversalLaw/Side24/ConeMoments.lean` | the Gaussian exponential-moment identity |
| `UniversalLaw.Side24.cone_moment_m2_algebra` | `UniversalLaw/Side24/ConeMoments.lean` | the integral identity or the Rayleigh cone integration |
| `UniversalLaw.Side24.cube_root_simplification` | `UniversalLaw/Side24/ConeMoments.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.pin_determinant_and_joint_dimension` | `UniversalLaw/Side24/ConeMoments.lean` | the analytic statement the arithmetic is attached to; the coefficient bound; the parent theorem |
| `UniversalLaw.Side24.hessian_block_eigenvalues` | `UniversalLaw/Side24/ConeMoments.lean` | the spectral decomposition |
| `UniversalLaw.Side24.d2_endpoints_adjacent` | `UniversalLaw/Side24/Endpoints.lean` | that c_2,24 lies between them |
| `UniversalLaw.Side24.d3_endpoints_adjacent` | `UniversalLaw/Side24/Endpoints.lean` | that c_3,24 lies between them |
| `UniversalLaw.Side24.d3_below_d2` | `UniversalLaw/Side24/Endpoints.lean` | anything about the true coefficients; the ordering d3 upper < d2 lower is a derived artifact-consistency check of the displayed endpoints |
| `UniversalLaw.Side24.endpoints_in_unit_tenth` | `UniversalLaw/Side24/Endpoints.lean` | anything about the true coefficients; the range (0, 1/10) is a derived artifact-consistency check of the displayed endpoints |

## Pointers, not new rows

- **`STATUS.md` rows D1-D6, SARD-G and C6** are Math- obligations, with their sources in Math-. They belong to Math-'s index, whose candidates DB642 extracted (issue 684 comment 6503371863). `STATUS.md` is dated 2026-09-29, so it needs the G8 refresh.
- **The two float-based `history/carrier20_recovery_20260928/public_carriers/` verifiers** should be labelled as 01XJMJ proposed. They back OBL-MAIN-20.
