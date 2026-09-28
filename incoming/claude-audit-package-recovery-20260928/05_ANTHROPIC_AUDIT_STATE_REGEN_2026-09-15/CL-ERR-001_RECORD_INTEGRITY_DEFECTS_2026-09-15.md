# CL-ERR-001-v1.0 — RECORD-INTEGRITY DEFECT REGISTER, 2026-09-15 (with proposed dispositions)

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-15 · CLASS: ERR — errata / defect register
STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE — **no theorem in the tree is weakened by any item**;
  every item is a label, display, or provenance defect with conservative or non-operative polarity.
DISCIPLINE: frozen carriers are NOT edited (corpus rule). Every disposition below is an amendment, a new
  constant for the next issuance, or a gate ck — never an overwrite.
VERIFICATION: `cl_reexec_20260915.py` (fail-closed, both modes byte-identical,
  digest 631564ef88b92629d7cf1c6dcc0448e3bfa7236b6c6820241a2330377226df2e) reproduces E1–E2 and §"confirmations".

## E1 — "exact bracket 21.92788306" is not exact (D3 remote bracket)

- **Where:** `D3_REMOTE_AMENDMENT_v2.md` lines 49–51: `I_far = 4.247483056`, `sum 21.92788306`.
- **Exact:** 2.5282637 × 1.68 = **4.247483016**; bracket = 17.6804 + 4.247483016 = **21.927883016**.
- **Slip:** +4.0e-8 in I_far, +4.4e-8 in the bracket. Implied κ_far behind the stated I_far = 0.6800000158.
- **Propagation:** the string "exact bracket 21.92788306…" appears in `D1_ASSEMBLY_v2_2.md:153`,
  `RETURN_06/ADVERSARIAL_REPORT.md:53`, `RETURN_06/OBLIGATION_LEDGER.md:39`, `RETURN_06/00_EXECUTIVE_STATE.md:62`,
  and is hard-coded as `B_REMOTE_EXACT` in `d1_falsify_v3.py:118`.
- **Polarity:** conservative — B_remote is consumed as an upper bound inside Ĩ_hi; overstating it weakens, never
  breaks. Display 21.9279 remains a valid round-UP of both values; Ĩ_hi = 650.1827·r³ unaffected.
- **Why it still matters:** four carriers say "gate v3 rejects anything below 21.92788306" and the correctly
  recomputed bracket sits below that floor; the gate passes only because ck `C-bracket-recompute` tolerates
  1e-7. Same class as LPW v1 defect D1 (published decimal not following from exact arithmetic).
- **Proposed disposition (D1 v2.3 / d1_falsify_v4):** amendment line in D3 lane correcting I_far and the
  bracket; `B_REMOTE_EXACT = Decimal("21.927883016")`; `C-bracket-recompute` tolerance → exact equality
  (both operands are exact decimals); prose floor restated as "rejects consumption below the certified display
  21.9279"; keep `B_REMOTE = 21.9279`. **DONE in the v2.3 DRAFT + gate v4 (D1_v2_3_DRAFT/).**

## E2 — a retracted LPW display is live in the executive state and the decision ledger

- **Where:** `RETURN_06/00_EXECUTIVE_STATE.md` (LPW theorem line) and `DECISION_LEDGER.md:325` publish
  `c = 1.1357762846347807215698804e-43` labelled "exact Fraction".
- **Fact:** 260/(2082000000000·2⁴⁰·10²¹) = 1.13577628463478070056633120196926…e-43. The published 29-digit
  string was float-rendered and sits **2.1e-60 above** the exact value — retracted by H1 v2-hardening finding
  F-C29 (its own expansion 1.1357762846347807005663312019e-43 verified here as a downward truncation).
- **Polarity:** UP on a lower-bound constant = unsafe direction; operative floors (≥ 1.13e-43 ≥ 10⁻⁴³) safe.
- **Now doubly superseded:** LPW v3 (9.80e-23) and v4 (2.2291086664054237e-10, r₀ = 1/2278031360) —
  see CL-STATE-001 row 2.
- **Proposed disposition:** replace both occurrences at the next issuance with the v4 pair; add a gate ck
  forbidding the retracted 29-digit token tree-wide (the `F-retracted-string-absent` pattern already used for
  the H4-JC retraction). **Gate v4 forbids the token in the v2.3 body.**

## E3 — seven frozen-body extraction rules live in one tree; one claim unreproducible

See `CL-REG-001`. Same-day siblings verify under different rules (`PERC_DECAY.md` raw-only;
`D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` strip-only; the H5 rung certificates under a seventh rule,
`before_hashline`). `LEAD_INTENSITY_DERIVATION.md` (H4 FREEZE "own convention") reproduces under none of nine
candidate rules. **Disposition:** rule-id in every freeze record; authoring rule "no blank lines adjacent to
markers"; gate ck on (carrier, rule-id) pairs. Owner for the unresolved item: H4.

## E4 — H5 rung-2/3 generating code unpinned

See `CL-PIN-001`. The rung certificates pin totals JSON + own body only; `h5_run.py` and the four per-rung
drivers have no current-bytes pin and were edited after the totals were written. Every other lane that landed
2026-09-15 (B4LOC, PERC, H3, LPW v3/v4) pins every script and transcript. The H5 lane already had one
merger-code contamination episode this week. **Disposition:** the rung lane re-pins at next issuance
(driver + merger + kernel sha256 inside each rung certificate, as B4LOC does); CL-PIN-001 holds custody until then.

## E5 — DECISION_LEDGER.md ends at 2026-09-13

Last entry: H4 (2026-09-13). Missing: D1 v2.0→v2.2, H4JC-R1, H5 rungs 2–3 and the totals errata, B4LOC-R1,
PERC-DECAY restatement, H3 band floor + ceiling, LPW v3/v4, BRANCH errata, W8 DMAX instances 3–4, W3 deaths.
**Disposition:** `CL-LEDGER-001` supplies a dated addendum in the ledger's own format for the operator to
append or fold.

## E6 — RETURN_06 superseded by its own addenda; CURRENT_STATE_DELTA in Drive is stale

`CURRENT_STATE_DELTA_2026-09-15.md` (Drive, 01_CURRENT_ASSEMBLY_AND_STATE) lists five premises with B4.loc
"asserted-but-not-established" and PERC as "o(r³) pricing", both superseded by `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md`
in the same folder. **Disposition:** `CL-STATE-001` (regenerated state) proposed as the replacement; rule going
forward — regenerate state documents, do not append to them.

## E7 — 92-vs-91 checks (cosmetic)

`00_EXECUTIVE_STATE.md` says the D1 gate has "92 cks"; the gate prints 91 `PASS` ck lines plus a summary
line. Recorded so a future re-execution is not misread as a discrepancy.

## E8 — B4LOC per-rung probability displays are nearest-rounded (E-B4LOC-1)

`b4loc_driver.py` prints `P_r(B4) <= C_RN*sqrt(Q1+Q2)` via `mpmath.nstr(·, 3)` — NEAREST rounding to three
significant figures — so `1.22e-9 / 1.52e-45 / 1.03e-197` may sit up to 0.41% BELOW the assembled value. The
driver's o(r³) gate uses the exact value, so the certificate is sound; under round-UP display discipline an upper
bound must be consumed at `1.23e-9`. **Disposition:** v2.3 consumes 1.23e-9 and records the nit; B4LOC's next
amendment prints round-UP or four digits.

## E9 — RN-UNIF engine (2026-09-16, see CL-RNU-001)

`d3_rn_unif.py` (2,240 lines, Kimi 09-15 07:43) exists unfrozen and unmentioned in every state document.
Defects found while validating it: (E-RNU-1) `mean_grad_exact` omits −TC·G6inv·dYCᵀ in every row and
−dTC·G6inv·YCᵀ in the Y-target rows — not exact, FD mismatch up to 185×; corrected version FD-validated to 3e-42.
(E-RNU-2) `chi2_grad_bound` is 1e19 too loose (1.57e14 vs true 1.56e-5): absolute-value summation through the
rigid Σ_pair⁻¹. Neither touches any frozen certificate (both live only inside the unrun Piece-1 net machinery).

## Confirmations (no defect) established this pass

- Headline chain `650.1827 / 650.1826140 / 628.2548 / 647.8048` reproduces exactly from
  `I_hi/r³ − 19.55 + 21.9279 = 650.182614016`; Ĩ_hi = 8.1272827e-2 is above the exact assembly.
- LPW v1 defect D2 re-derived: E[(|X|+|Y|)⁴] = 12 + 32/π (from E|X| = √(2/π), E|X|³ = 2√(2/π), E|X|⁴ = 3);
  12 + 16/π is false. v2 amplitude majorant: E[R⁴] = 8 exact, E[R] = √(π/2) ≤ √2.
- LPW v3 and v4 exact fractions reproduce digit-for-digit from their stated recipes; v4/v3 = 4B₃/U exactly.
- B4LOC-R1 assembly, c_eff ladder, o(r³) threshold 6.5047, κ·r invariance all reproduce.
- H5 ladder, step percentages, C_unif margin, C1 containments reproduce (rung 3 uses the literal r = 0.035355).
- H3 band floor margins 59.88% / 42.78% vs c_Z reproduce; floor < ceiling; U ≤ 4.
- The RN-UNIF exact-derivative engine `kappa_far_ds` (value, gradient, Hessian) matches FD to 1e-26 at three
  stations.
