# 00_READ_FIRST — K3 SWARM: SIDE24/q0 LOWER-RATE RIGOROUS REPAIR

Work order: K3_SWARM_MASTER_WORK_ORDER_—_SIDE24_q0_L1.txt (2026-08-05, sha256 recorded in CANONICAL_STATE.json).
Lead coordinator: KIMI line. This directory is the campaign's deliverable root. All dependency references inside
deliverables use relative paths only.

## Controlling facts established at intake (2026-08-05)

1. **The K3 blocking defect is CONFIRMED, numerically and symbolically.** The WP "certified upper bound"
   used the factor `p_grad · sqrt(E[det(H)^2 | grad=0]) · min(P_type, P_window)`. The Cauchy–Schwarz chain gives
   `... · sqrt(P(type ∩ window)) ≤ ... · sqrt(min(P_type, P_window))`. Since p ≤ sqrt(p) on [0,1], the missing
   square root makes the reported quantity SMALLER — not a valid generic upper bound. At the certified witness
   point (−0.04, −0.58): missing-sqrt value 1.31310e-4 vs CS-valid 2.30247e-2 (175.3× inflation).
   Consequences (all downstream uses INVALID as upper bounds): the advertised I_cs ≈ 1.30e-5; the WP rung
   "upper-bound" table; the fitted 5.5e-3·r^1.6 envelope; the AO reliability coefficient's WP-ub branch;
   THM-023 v1.1's WP row and its ub-based constant display.
   NOT affected: LB-1's above-b saddle channel (rho_CS/rho_sad carry the sqrt — valid); the two-sided
   quadrature ESTIMATE I_true ≈ 4.76e-6 (an estimate, never a bound claim; its own certification is a
   separate open task); the conditional-law witness values; the m(P*)/m(Q*) premise-falsification points;
   Task 4c's Kantorovich enclosures; DER-027a's variance channel (independent machinery).
2. **THM-023 v1.1 is a noncontrolling HOLD draft** (unsealed PLACEHOLDER-BODY-HASH, [PENDING] slots,
   an unfinished certificate path with a KeyError: verifier accesses tbl['0.005'] absent from its RUNGS).
   Its bytes are preserved unchanged in the research tree; nothing in this campaign consumes it.
3. **The 0.9144036 coefficient is a candidate measured/mixed-tier asymptotic anchor**, not a proved fixed
   theorem constant. The successor assembly must keep the four tiers distinct (proved constant / limiting
   coefficient / measured anchor / direct numerical estimate).
4. **The manuscript delivered 2026-08-05** (SIDE24_pre_peer_review) consumed the invalid bound in its
   §3.10.6/3.10.9/§6 (the 1.30399e-5 "certified upper bound", the 5.5e-3·r^1.6 envelope, and the ub branch
   of the constant ledger). A corrected revision is owed after the WP adjudication; the invalid items are
   quarantined here as evidence pending W2/W3/W4 conclusions.
5. Corpus: 578 uploaded files + 20 research carriers inventoried byte-exact (INPUT_MANIFEST.sha256,
   FILE_LEDGER.tsv). No MISSING objects at intake. The orphaned pre-update THM body 40596829… remains
   declared UNRECOVERABLE (KIMI-DATA-028; successor citations re-point to ba3974c0…).
6. Estimand firewall: p_r, q_r, q_r^adj, elder-rule pairing, and the ratified upper theorem are distinct;
   every typed-to-adjacent transfer retains the η_r channel. The ratified upper chain (AO48-OPR-045) is
   outside this campaign. This work changes no Boolean/operator status.

## Swarm map (workstream → owner → deliverable)

- W1 corpus/provenance: lead (this directory's intake artifacts).
- W2 symbolic WP/Kac–Rice auditor: independent agent (freezes derivation before seeing numerics).
- W3 rigorous WP numerics: independent agent (interval enclosures).
- W4 independent WP implementation: independent agent (different representation; freeze-then-compare).
- W5 C030/C031 forensic: agent (provenance of 3e-15 and 0.213r³).
- W6 uniform-in-r analyst: agent (scaled asymptotic law or interval coverage).
- W7 γ-LOC: existing DER-026 agent (context retained).
- W8 Λ-side: existing DER-027b agent (context retained; full-grid transcripts running).
- W9 DER-027a/027c scope: existing DER-027a/027c agents.
- W10 dependency-minimization: agent (event-inclusion recomposition; what each channel must really prove).
- W11 literature/methods: agent (primary sources).
- W12 independent adversarial reviewer: dispatched after W2/W3/W4 freeze.
- W13 reproducibility/export: dispatched with the assembly.

## Phase discipline

W2, W3, W4 must NOT see each other's conclusions until each files a frozen report (hash-bound). The lead
routes cross-context only after the freeze receipts exist. No theorem successor is minted until gates
G0–G12 pass; a conspicuously marked noncontrolling shell is prepared first.

## Corrections (red-team findings, 2026-08-05)

- **B2 (Isserlis display, lead).** Fact 1's canonical witness absolutes were computed with a fourth-moment
  display that drops mean terms (overcount exactly 2(det nu nu^T)^2; inflation sqrt(1.34488)). Corrected:
  defective form 1.31310e-4 -> 1.13229e-4; CS-valid 2.30247e-2 -> 1.98542e-2; the defect ratio 175.35 and
  the D1 quarantine are unchanged; direction conservative. See LEAD_SYMBOLIC_WP_CORRECTION_ADDENDUM.md.
- **B1 (W6 modulus).** E_WP = 4.2*r^{3/2} is invalid for r <~ 0.003 (near-cluster CS singularity artifact);
  E_WP(0.05) survives; W6 is repairing via zone split. WP-min is therefore NOT yet delivered (G6 open).
- **B3 (F1 quantification).** W6's premise F1 was mis-quantified (worst gap sup 4.236 > 4.2 under the
  stated premise; 4.129 under measured slope 0.20). Part of the W6 repair.
- **B4 (gamma-LOC scope).** W7's retirement holds for Forms B/C/D; the exit node of Form A still consumes
  P-NMZ-gamma. Re-scoped accordingly.
