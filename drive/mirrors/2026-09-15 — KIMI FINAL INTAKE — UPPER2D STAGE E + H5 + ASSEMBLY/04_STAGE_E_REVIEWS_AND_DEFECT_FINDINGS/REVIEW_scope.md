# REVIEW_scope.md — QUANTIFIER/SCOPE-ONLY independent review, D1 v2.0 (U2D-UPPER)

Reviewer role: quantifier order, scope, asymptotic grade, hypothesis propagation,
epistemic labels. Mathematics not re-proved; the review checks that what is
CLAIMED is precisely what is PROVED. Prior exposure: NONE beyond the task
prompt (no prior sight of any file in this package).

**DISPOSITION: FAIL** — three exact scope violations (F1, F2, F3 below), all in
hypothesis/constant propagation into the frozen theorem carrier
`D1_assembly/D1_ASSEMBLY_v2_0.md`. Custody (hashes) is CLEAN; the failures are
logical-shape failures, not custody failures.

---

## 0. Custody verification (body hashes; all MATCH)

Extraction per each file's own declared rule (marker segment for most; whole
file for C1/H3; separator rule for H5). Recomputed independently:

| carrier | hash | verdict |
|---|---|---|
| D1_assembly/D1_ASSEMBLY_v2_0.md | 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 | MATCH (17,988 B) |
| B1_taxonomy/B1_TAXONOMY.md | 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 | MATCH (= its FREEZE.txt) |
| C1_alpha_intensity/C1_ALPHA_INTENSITY.md | ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f (whole) | MATCH (= C1_RECEIPTS.txt) |
| C2_B_classes/C2_B_CLASSES.md | 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 | MATCH (= its FREEZE.txt) |
| D2_branch_control/D2_BRANCH_CONTROL.md | 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa | MATCH |
| D3_percolation/D3_PERCOLATION.md | 8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47 | MATCH |
| H3_closure/H3_CLOSURE.md | 387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637 (whole) | MATCH (= MANIFEST) |
| H4_closure/H4_CLOSURE.md | a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 | MATCH (= its FREEZE.txt) |
| H5_closure/H5_CLOSURE.md | 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 | MATCH (separator at byte 15162, as v2.0's ledger states) |
| D1_assembly/D1_ASSEMBLY.md (v1.0) | 006b8a7d011443a0a98401f30335fbcfe44ef5996b4fc41dd9735640d958de40 | MATCH (preserved) |
| D1_assembly/D1_ASSEMBLY_v1_1_ADDENDUM.md | 634338b43a000502dfdda62a9019895434db7b50607a9989caaa1fd5873167e2 | MATCH (preserved) |
| D1_assembly/D1_ASSEMBLY_v1_2_ADDENDUM.md | a7e1958c0d37d4eb63929823edd3190f83216fb4b127a4e85ad7e2e4da86ba67 | MATCH (preserved) |

Companion check: `d1_falsify.py` re-run, exit 0, `D1-V2-FALSIFIER PASS`,
stdout byte-identical to frozen `t_normal.txt`. `D1_V2_RECEIPTS.txt`
consistent with the above.

---

## 1. THE FAILURES (exact sentences, exact violations)

### F1 — D3-LEMMA-RN-UNIF (OPEN in v2.0's own register) is silently dropped between D3 and Theorem (1)/§8

D3's own declared grade for the spine the assembly consumes
(`D3_PERCOLATION.md`, §5 grade summary):

> "the event form, the cover lemma D3-SPLIT, the count split D3-COUNT, the
> unconditioned plateau closed forms and J(ℓ), the spine's polarity-safe form
> (modulo (i) and the zone-uniformity step (ii)), and all certificates are
> theorem-shaped …"

Blocker (i) (two-sided normalizer) is closed by H3 (v1.1). Blocker (ii) IS
D3-LEMMA-RN-UNIF, and v2.0's own register (§5, item 3) marks it **OPEN**.
D3-THM's validity domain is "for 0 < r ≤ 0.1", i.e. it includes the rung
r = 0.05; the zone-sup attainment the lemma names is exactly what turns the
D4-net displays into the 17.02·r³ / 2.5283·r³ bracket bounds at the rung.

Yet Theorem D1 v2.0(1) consumes the bracket with only the rim/axis caveat:

> "where E_w(0.05) = ∫_{T²} ρ_w dA ≤ I_hi = 9.142887704e-02 … is a CERTIFIED
> INTERVAL upper bound (H5; machine-certified box cover of the whole chart
> plus the D3-certified remote bracket 19.55·(0.05)³ = 2.4438e-3 consumed
> exactly once inside it; the rim and axis envelope regions stand at
> named-lemma grade — H5-RIM, H5-AXIS v1+v2+v3 …)"

and §8 asserts:

> "Theorem-grade today, unconditionally of every open item: the exact
> partition; Lemma R; the normalizer two-sided bound (H3); the joint-carrier
> and RN machinery (H4-JC, H4-RN); the remote spine at D3's grade; and the
> CERTIFIED RUNG statement Theorem D1 v2.0(1) at r = 0.05 …"

Scope violation: an OPEN named lemma on which the remote bracket stands (per
D3's own text) does not appear in Theorem (1)'s disclosure (which enumerates
the named-lemma-grade regions as rim/axis only) nor in §8's "unconditionally
of every open item" inventory, while the same document's register lists it
OPEN and Theorem (2) names it. H5's frozen closure likewise labels the part
"D3-certified (consumed)" (H5 §1, R5 row) with no mention of the lemma.
Same omission in §8's closing completeness claim:

> "If OBL-D1-PROMOTE does not land, what remains is EXACTLY: (i) … (ii) the
> H5-RIM / H5-AXIS production paths … (iii) the H1/dir branch items …
> Nothing else stands between the current state and the full theorem."

— (i)–(iii) do not cover D3-LEMMA-RN-UNIF (remote-zone RN uniformity is not a
chart-envelope, rim/axis, or H1/dir item).

### F2 — the certified `(1 + κ_far)` factor is silently dropped from the consumed remote bracket (evidence-grade value in certified position)

D3-THM (the statement the assembly consumes), verbatim:

> "I_far(r) = E_{P_r}[N_w({d ≥ 5})] ≤ (576 − 25π)·J(ℓ)·(1 + κ_far(5))
> = 2.5283·r³·(1 + κ_far), κ_far(5) ≤ 0.63 + ε_QMC (certified cross bound
> 0.63 axis-worst + QMC RN 1.00±0.01)"

v2.0's constant bookkeeping, verbatim:

> "The remote bracket 19.55·r³ IS the D3 spine (I_ann ≤ 17.02·r³ + I_far ≤
> 2.5283·r³) and appears exactly once — inside I_hi."

and §3's mapping table: "the D3 spine bracket (I_ann 17.02 + I_far 2.5283 =
19.5483 exactly, consumed at the conservative round-up 19.55·r³ …)".

Scope violation: "I_far ≤ 2.5283·r³" instantiates κ_far at its QMC-EVIDENCE
value (RN ratio 1.00±0.01, explicitly labeled evidence in D3 and v1.2). The
certified bound is κ_far ≤ 0.63, i.e. the certified bracket is
17.02 + 2.5283·1.63 = 21.14·r³, which exceeds the consumed 19.55·r³ by
1.59·r³. The alternative crude-cap route for d ≥ 5 (plateau 0.55·r³ per unit
area × area 497.46 ≈ 273·r³) cannot rescue the smaller number. Note v1.2's
theorem-level spine form KEPT the factor ("E_w(r) ≤ ( C_raw + 17.02 +
2.5283·(1 + κ_far) )·r³ [spine form]", v1.2 §B4); the drop entered via H5's
consumption ("D3-certified envelope (17.02 + 2.53)·r³ = 19.55·r³", H5 §1 R5)
and is canonized in v2.0. Consequence: the 19.55·r³ remote part inside
Theorem (1)'s "CERTIFIED INTERVAL" is certified only up to an evidence-grade
RN ratio — an estimate appearing syntactically as a theorem constant.

### F3 — PERC-DECAY is not among Theorem (2)'s hypotheses, but the conclusion's B4.rem summand (and the o(r³) parenthetical for B2/B4) consumes it

Theorem D1 v2.0(2):

> "Assume the named hypotheses of §5 — H1 (OBL-B1-BRANCH, including
> OBL-B1-BRANCH(dir) = OBL-D2-AO-SHARP(iv) with its exact C¹ sufficiency
> threshold), the remaining OBL-D2-AO-SHARP sharp-constant items,
> D3-LEMMA-RN-UNIF, and OBL-D1-PROMOTE (§4). Then there exist r₀ > 0 … and a
> finite constant C such that 1 − q(r, 6/5) ≤ C · r³ for all 0 < r ≤ r₀.
> Under H1 the dir, B2, and B4 terms are o(r³) (the dir term via the C¹ tube
> threshold; B2 corridor via the certified profile; B4 loc super-algebraic) …"

The consumed carriers price the remote B-class summands ONLY via the
percolation input:

- C2 §4.3: "The remote part `B4.rem` … is controlled by the same tube
  control applied along the whole M-ward channel plus the global excursion
  geometry **[CONSUMES: OBL-B1-PERC]**."
- v1.0 §4 ledger: "B4 rem | excursion-geometry control | o(r³) | H2"; §5
  register: "H2 | OBL-B1-PERC | … | A.rem; B2 far; B4 rem".
- v1.2 §B2 scope note: "The B4.rem remote excursion term remains attached to
  the named PERC-DECAY/excursion-geometry input and to the open H1 tube
  item"; and for B2: "The B2-far raw count is O(r³) at the same structural
  grade … its o(r³) upgrade consumes PERC-DECAY."
- v2.0's own register, item 5: "PERC-DECAY | subcritical level-s⁺ cluster
  decay; **o(r³) upgrades + B4.rem only** | … | OPEN (upgrade-only; the
  O(r³) form does not consume it)".

Scope violation: (a) the conclusion 1 − q ≤ C·r³ requires P_r(B4.rem) ≤ C·r³;
no carrier prices B4.rem at any grade without PERC-DECAY (v1.0's ledger gives
B4.rem exactly one line: "o(r³) | H2"), so Theorem (2) silently assumes an
OPEN hypothesis it does not name. (b) The register row is internally
inconsistent: its content says PERC-DECAY is consumed for "… + B4.rem only"
while its status parenthetical says "the O(r³) form does not consume it" —
both hold only if an O(r³) pricing of B4.rem without PERC-DECAY exists, and
none is stated anywhere. (c) The parenthetical "Under H1 the dir, B2, and B4
terms are o(r³)" over-claims: B2-far's o(r³) and B4.rem's pricing consume
PERC-DECAY per the quotes above; only B2-corridor (H1) and B4.loc (H1) are
o(r³) under H1 alone. (This gap exists already in v1.2's conditional theorem,
which lists only H1 and H5; v2.0 inherits it.)

---

## 2. Checklist items that VERIFY CLEAN

**(1) Quantifiers.** Theorem (1) is everywhere a single-rung statement
("[CERTIFIED RUNG, r = 0.05]"); no sentence in any package file upgrades it to
all-small-r (H5 §5: "No exponent fitted from two rungs (discipline)"; v2.0
§4(e): "No all-small-r claim is made from rung sampling; the crude 0.025 rung
is a labeled display, never a premise"). Theorem (2)'s r₀ is explicitly
existential ("EXISTENTIAL: H3's convergence has no certified modulus, and no
explicit radius is claimed"), matching H3 §2 ("r_0 > 0 (the small-r range)
existential") and v1.1's honest boundary ("unexecuted extension — not
claimed"). C is existential ("a finite constant C"). D3-THM's "0 < r ≤ 0.1"
carries its named blockers. H3's Z_r statement "0 < r ≤ r_max" existential —
consistent.

**(2) Constants (apart from F2).** Certified-interval: I_hi =
9.142887704e-02 (consumed as the frozen, conservative-larger value — the
drift note honestly records the live json now reads lower; the falsifier cks
the direction fail-closed, and I verified the live `h5_totals.json`
I_hi = 8.6319e-02 < frozen). C_RN(0.05) ≤ 3.46 theorem-grade explicit —
consistent with H4's ladder 3.44/3.46/3.46/3.47 and its uniform explicit
3.47; recomputed: √31.23/1.615489… = 3.459 ✓. c_Z = c₀/2 ≥
1.615489267643502474 (conservative truncation of the longer certified
string) ✓. dir threshold recomputed: √(12 ln(1/r) + 4 ln C_RN + 2 ln 2) =
5.83/6.50/7.12/7.68 at r = 0.1/0.05/0.025/0.0125 ✓. Projected values are
labeled and never consumed: "~6–7e-2 total, DOCUMENTED, NOT consumed, never
presented as current" (v2.0 §3, §6); H5_PROMOTE's "C_unif ~ 150–400"
projection labeled "PROJECTION (labeled, not consumed)". C*_env = 1.284 is
labeled "the evidence-grade C*_env = 1.284" in v2.0 §2 — and note v2.0
retires v1.2's "I_hole ≤ C*_env·r³ = 1.284·r³", which had an evidence-grade
constant in theorem position; that is an improvement.
Minor: "I_hi = 9.142887704e-02 = 731.43·(0.05)³" is a rounded display —
731.43·(0.05)³ = 9.1428750e-02, short of I_hi by 1.27e-7 (and the chained
"I_hi = C_chart·(0.05)³ + 19.55·(0.05)³" undershoots by the same amount);
the falsifier explicitly windows these as displays ("display-731.43",
tolerance 0.01) and cks the ledger conservative (≤ I_hi); Theorem (1)
consumes I_hi itself, so no certified statement is weakened — flagged as
display-rounding only.

**(3) Asymptotic grades.** v2.0's own body contains no Θ(r³) claim for E_w
or 1−q, no convergence claim for (1−q)/r³, no "limiting coefficient"
language (grep-clean), and no two-sided claim about 1−q ("No two-sided
language anywhere", §6). "The normalizer two-sided bound (H3)" (§8) IS
supported: H3 §2/S6 proves `c_Z r² ≤ Z_r ≤ C_Z r²` (upper side analytic,
independently derived) — legitimate two-sided at the normalizer, not at
1−q. v1.0 §9's "Historical headline for orientation only: 1 − q =
C*·r³·(1+O(r³)) two-sided at architecture grade … nothing historical is a
premise" is labeled historical/orientation — acceptable in the preserved
predecessor. Residual note: H4_CLOSURE §2.4/§4 and v1.2 §B3/§B4 carry
"E_w = Θ(r³) power law theorem-grade (modulo H5 … OPEN)" — Θ with an open
constant, qualified in the same sentence; v2.0 §4(d) effectively regrades
this (structural ledger + single-rung certificate + OBL-D1-PROMOTE) under
the blanket "supersedes their status lines", but never explicitly says the
v1.2/H4 Θ-grade phrasing is amended. Had the Θ upper side stood at theorem
grade with constants, OBL-D1-PROMOTE would be trivially closed — v2.0's own
§4(d) contradicts that reading, so the operative grade is v2.0's; noted as a
documentation gap, not a live over-claim.

**(4) Hypothesis propagation (apart from F1/F3).** The §5 register is
complete in form: every obligation has exact content + falsifier + owner
(OBL-B1-BRANCH(dir) threshold 5.83/6.50/7.12/7.68 with saturation-checked
falsifier; OBL-D2-AO-SHARP(i–iii, v) matching D2's G-table (G2/G4; G0/G1
closed — verified against D2 §grades); D3-LEMMA-RN-UNIF; OBL-D1-PROMOTE with
content + two-route falsifier + owner; PERC-DECAY; H5-RIM/H5-AXIS with H5 §1
flatness hypotheses and §6 production paths). H5-RIM/H5-AXIS reach Theorem
(1) with disclosure ✓; they reach Theorem (2) only via OBL-D1-PROMOTE's
content (i) ("the shell_hi_fi rim extension (retires H5-RIM), the
factored-det expansion (machine-certifies the H5-AXIS wedges and the
S-disk)") — adequate but implicit; Theorem (2) does not name them
separately. D2's OBL-D2-AO-SHARP(iv) = OBL-B1-BRANCH(dir) identification is
consistent across D2 §G3, v1.2 §B3, v2.0 §5 item 1a.

**(5) Labels.** C1's evidence/display discipline is explicit ("Nothing
labeled evidence is used as a proof ingredient"; C* comparison "a consistency
display only … measured values are never theorem constants"). N/B3 "exact
zero" traces to Lemma R (a proof), not to a zero sample count; discrete
mirrors (453/1437/433 ensembles, "zero leaks") are labeled mirrors of proved
lemmas. The 0.025 rung is "LABELED CRUDE" with no fitted exponent ✓.
I_lo labeled "partial by construction" ✓. The C1 containment
(1.605315e-4 ∈ [I_lo, I_hi]) labeled consistency display ✓ (verified:
1.605315e-4 < 9.142887704e-02; 569× ratio = 569.5 ✓).

**(6) The v1.2 → v2.0 semantic diff (enumeration).**
(i) H5 landed: v1.2's open "C_raw pending H5 (evidence 1.284)" is replaced
by the certified interval I_hi = 9.142887704e-02 with C_chart = 711.88; the
569× width is stated honestly. Survives amended, explicitly.
(ii) The remote mapping: v1.2's separate I_hole + I_ann + I_far terms are
retired into the whole-torus joint-carrier integral; the remote bracket
appears once inside I_hi (v2.0 §2, §3) — explicitly stated; BUT see F2 (the
(1+κ_far) factor present in v1.2's spine form is dropped in v2.0's
bookkeeping — an unannounced semantic change, regression).
(iii) C_RN at the rung: v1.2's uniform explicit ≤ 3.47 (ladder
3.44/3.46/3.46/3.47) → v2.0's C_RN(0.05) ≤ 3.46 — consistent with the
ladder at r = 0.05; not a weakening; v2.0 does not flag the change but none
is needed (rung statement).
(iv) New: Theorem (1) (certified rung), OBL-D1-PROMOTE, H5-RIM/H5-AXIS
named lemmas — all with content/falsifier/owner.
(v) v1.2's "E_w(r) = Θ(r³) power law theorem-grade" phrasing → regraded by
§4(d) (see §3 note above) — effectively amended, not explicitly flagged.
(vi) v1.0/v1.1/v1.2 bodies preserved byte-identical (hashes MATCH, §0);
v2.0's supersession sentence: "This document is the current strongest form
and supersedes their status lines" — the v1.x status lines only; the
obligations they name survive in v2.0's register.
(vii) v1.2's hypothesis list for the conditional theorem (H1 + H5) → v2.0's
(H1 incl. dir, remaining OBL-D2-AO-SHARP, D3-LEMMA-RN-UNIF, OBL-D1-PROMOTE):
H5 is replaced by its landed rung + promotion obligation — correct in shape;
D3-LEMMA-RN-UNIF correctly added; PERC-DECAY still missing (F3).

**(7) Theorem table.** v2.0 contains no LPW/LPW_CONSTANT/W8/W3 theorem
table as such; the required separations are present: LPW lower side "a
separate obligation sharing only the taxonomy placement G_r ⊂ A.loc ⊂ A …
no LPW conclusion is a premise here" (§6; v1.0 §8 "Relation to the LPW
qualitative theorem: NONE"); "The 3D theorem appears in no 2D dependency
(the v1.1 amendment and MU-5)" (§6) — and H3's 3D→2D typing amendment is
carried in v1.1; no "program complete" language anywhere ("closes the day
OBL-D1-PROMOTE and the H1/dir items land", §8). W8/W3 appear only in
plan.md ("continue as auxiliaries at their own scopes") — their absence from
v2.0 is consistent with "(if present)".

---

## 3. Minor observations (non-dispositive)

- **H5-AXIS "v3" referent gap:** Theorem (1) and §3 name "H5-AXIS v1+v2+v3"
  and "full S-disk d_S < 0.062", but the frozen H5_CLOSURE.md defines only
  v1 and v2 (v2 disk d_S < 0.03, §1); "v3" appears once undefined in the
  frozen §0 headline ("the v3 S-disk skip"), and the 0.062 exemption lives in
  H5_STATE.md and the unfrozen, PENDING tightening amendment. The named
  lemma version consumed by v2.0 is not fully stated in the pinned frozen
  carrier (its receipts exist: h5_results_r0.05_sdisk.jsonl). Documentation
  gap, not necessarily a grade error.
- v2.0's drift note quotes the live `h5_totals.json` as
  "I_hi = 9.1399568624…e-02"; the on-disk json now reads 8.6319e-02 (further
  refinement). Direction still downward; the falsifier cks direction only,
  so the stale quoted number is cosmetic.
- H5_STATE.md quotes BODY_HASH.txt as "465c97c07cf633…" — the actual
  BODY_HASH.txt and frozen hash are "465c97c0553e…". State-memo staleness
  only; the authoritative pins match.
- "≈ 33× the v1.2 spine display" (v2.0 §2): not exactly reconstructible
  (711.88/20.83 ≈ 34.2; 690.55/20.83 ≈ 33.2); hedged with "≈" — display.
- Theorem (2) subsumes H5-RIM/H5-AXIS under OBL-D1-PROMOTE without naming
  them; acceptable, but an explicit "via OBL-D1-PROMOTE(i)" would close the
  loop.

## 4. CANNOT-VERIFY (none blocking; recorded separately from FAIL)

- Whether D3's zone-integral trapezoid assemblies ("deterministic evidence
  with SEs/refinement certificates", D3 §5 grade summary) amount to a
  certified sup per zone independent of D3-LEMMA-RN-UNIF. D3's own text
  attaches blocker (ii) to the spine, so F1 stands on the documents as
  written; a foundations countersignature of (ii) would downgrade F1 to a
  labeling complaint.
- Whether some unstated polarity argument makes κ_far = 0 exact for d ≥ 5
  (would void F2). No carrier states one; D3-THM carries the factor.
- Whether an unstated O(r³) pricing of B4.rem without PERC-DECAY exists
  (would void F3(a)/(b)). No carrier states one.

## 5. Load-bearing dependencies (paths + hashes)

- D1_assembly/D1_ASSEMBLY_v2_0.md — body 86882dca5d5258e9…ffd62d0 (MATCH)
- D3_percolation/D3_PERCOLATION.md — body 8e7fef6b4fb98940…b3eac47 (MATCH) — F1/F2 source quotes (§3.3, §4, §5)
- C2_B_classes/C2_B_CLASSES.md — body 2aba099d630dc759…dadf3d2 (MATCH) — F3 source quote (§4.3 [CONSUMES: OBL-B1-PERC])
- H5_closure/H5_CLOSURE.md — body 465c97c0553ecc0a…8603d301 (MATCH) — F1/F2 propagation point (§1 R5 row)
- H4_closure/H4_CLOSURE.md — body a67d50b9a24c82f9…6b1d3 (MATCH)
- H3_closure/H3_CLOSURE.md — whole 387e4bae4881a036…b81637 (MATCH) — two-sided normalizer support
- D1_assembly/D1_ASSEMBLY.md (v1.0) — body 006b8a7d011443a0…58de40 (MATCH) — F3 ledger row "B4 rem | o(r³) | H2"
- D1_assembly/D1_ASSEMBLY_v1_2_ADDENDUM.md — body a7e1958c0d37d4eb…a86ba67 (MATCH) — F2/F3 v1.2 forms
- D1_assembly/D1_ASSEMBLY_v1_1_ADDENDUM.md — body 634338b43a000502…73167e2 (MATCH)
- B1_taxonomy/B1_TAXONOMY.md — body 7d7ddbec6e63b4c2…a68930 (MATCH)
- C1_alpha_intensity/C1_ALPHA_INTENSITY.md — whole ca2c02b81a05e4cb…72853f (MATCH)
- D2_branch_control/D2_BRANCH_CONTROL.md — body 9e95648acbe5307d…0b8fa (MATCH)
- d1_falsify.py re-run: PASS, exit 0, transcript byte-identical to t_normal.txt.

---

---

REVIEW_scope.md — scope/quantifier review of D1 v2.0. DISPOSITION: FAIL
(F1: D3-LEMMA-RN-UNIF dropped at Theorem (1)/§8; F2: (1+κ_far) dropped from
the consumed 19.55·r³ bracket; F3: PERC-DECAY missing from Theorem (2)'s
hypotheses while B4.rem/B2-far-o(r³) consume it). Custody: all 12 frozen
carriers hash-MATCH.
