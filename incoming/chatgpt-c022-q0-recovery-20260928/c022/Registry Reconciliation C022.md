# Registry_Reconciliation — obligations ledger and record hygiene (as of C022, 2026-07-09)

Supersedes Registry_Reconciliation_C021.md; companion to Constants_Table_Canonical_C022.md,
Lemma_LB_ARCH_Package.md, Lemma_FD_Package.md, R0UNIF_and_LBRestate.md,
C022_Observed_Update.json. Supersede-never-overwrite throughout.

## 1. Obligations ledger (complete, current)

| ID | content | status | discharge / carrier |
|---|---|---|---|
| OBL-LB-ARCH | rigorous positivity of ∫Λ·AO dA | **CLOSED (C022)** at derived-and-verified; two cited-standard steps flagged inline (Bogachev 3.6.1 small-ball at CM points; Morse–Smale basin continuity). Five certificates (margins 10^38–10^61, both rungs) + explicit RKHS barrier with frozen-pin rigidity (ε₀ ≈ 0.008 macroscopic) + CM support step. | Lemma_LB_ARCH_Package.md |
| OBL-FAR-DECAY | quantitative off-pair decorrelation (γ-LOC) | **far-field core DISCHARGED (C022)** derived-and-verified (19-row exact tables to d = 6, both rungs, envelope beyond); tube-local part remains architecture, scope unchanged | Lemma_FD_Package.md §2 |
| OBL-BETA-FAR | rigorous far-zone β ceiling | **fixed-radius core DISCHARGED (C022)**: P(β-far) ≤ 12.41·r³ at d₀ = 0.3, rung-independent, all r ≤ 0.05; per-rung three-piece cover complete (no gap at r = 0.05). Residual re-scoped → OBL-BETA-MODZONE | Lemma_FD_Package.md §3 |
| **OBL-BETA-MODZONE** | r-uniform moderate-zone (2r, d₀) sharpening: replace Cauchy–Schwarz Hessian ceiling by the near-fold \|det H\| suppression | **NEW (C022)** — the successor residual, precisely scoped; per-rung numbers already rigorous (84.2 / 334.9 ·r³) | Lemma_FD §3 scoping |
| OBL-R0-UNIF | 18×18 nondegeneracy uniformity | **DISCHARGED (C022)**: continuity + compactness + universal interpolation off the analytic null set; mp anchors 9.73e−21 / 3.73e−23 | R0UNIF_and_LBRestate.md Part A |
| OBL-LB-RESTATE | scope Lemma LB to moderate-r | **DISCHARGED (C022)**: supersede-in-scope note; asymptotic chain of custody = LB0 → R0 → LB-ARCH → FD | Part B |
| **OBL-LB-RATE** | rigorous sharp lower rate via C012/C013 limit objects | **NEW (C022, optional)** — not required for Θ(r³) | LB-ARCH §6 |
| OBL-NEEDLE-TREND | needle-ratio drift 0.068 → 0.086 | carried (non-load-bearing; one third-rung datum) | C021 |
| OBL-LOOP-LOCATE | exhibit/bound-away loop population | carried (bookkeeping) | UB0 Addendum A |
| OBL-LBB-1 | two-point at d ∈ {0.02, 0.05, 0.1} | carried | — |
| OBL-C019-2 | MID mechanism / polish hygiene | carried | — |

Previously discharged (unchanged): OBL-UB-PALM, OBL-C019-1, OBL-CLIP-ARCH, OBL-C017-1/2,
OBL-LB-ADJ (superseded), File-5 ghost debt; OBL-RIM-REFINE optional; κ_β moot.

## 2. C022 record notes

- **Construction dead end (not a registry kill):** the 15/18-pin barrier variant rejected-as-built
  (max|w| = 1.27×10¹⁴, evaluability destroyed; natural y*-Hessian max-typed). The 9-pin +
  shaping-bump route is the construction of record.
- **Instrument incident (closed same-session):** first r = 0.025 barrier replication ran with the
  unpatched module-global ℓ (S-pin f-err −1.82×10⁻⁵ = exactly the ℓ discrepancy). Root-caused,
  fixed, re-run clean; both runs retained in the record.
- **Honest-scoping note:** the FD moderate-zone bound is per-rung rigorous but not r-uniform
  as-written (crude ceiling ~d⁻⁴ vs true Λ-scale mass); the gap is now a *named, scoped*
  obligation rather than an implicit one.

## 3. Program state (of record)

**Theorem A:** measurement-complete (C021) **+** lower-side positivity closed at derived grade
**+** rigorous O(r³) upper envelope (C022). The Θ(r³) statement holds with exponent at
proof-adjacent grade and constant measured. Remaining to full proof grade: OBL-BETA-MODZONE,
basin-continuity formalization (if referee-demanded), γ-LOC tube-local, manuscript unification.
**Theorem B:** unaffected; its ledger stands as of C015.
**Submission paths:** experimental-mathematics package — CLEAR (assembly only); methodology
paper — CLEAR; exact-lemma cluster — CLEAR; proof-grade Theorem A — the four items above.

## 4. Reconciliation notes carried unchanged
C014 dual-account resolution; C014p2 snapshot-absence flag; date-field corrections; C020 ballfix
supersede chain (0.9702 → 0.9728) — as in the C020/C021 registries.
