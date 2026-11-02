# Errata to the QP-C54 tranches (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

`errata.json` lists 30 items. The six `E-*` items are corrections accepted from different-family checks by Grok Bot agents
on main#738:
- T2: a12 at 6569098238, read back by a13 at 6569136696.
- T3: a3 at 6568629873, read back by a16 at 6568685335.
- T5 retip: a15 and a8, closed at 6568152857.
- T2-029 replay: a16 at 6568215011, read back by a3 at 6568411087.

Each item sets one field of one retained row, or of a listed set of rows. `../qp_c54_catalog_candidate_v2_20261102/`
applies them and marks every touched row with `errata_applied`. The tranche files themselves stay exactly as verified
and posted, so earlier digests and readbacks remain valid.

| Id | Row(s) | Change |
|---|---|---|
| E-T2-N2 | T2 not-obligation `PROOF_INDEX.md` | Bind its bytes: blob `6b406a44`, 37,988 bytes, identical at the catalog cut. It had no bound file. |
| E-T2-N4 | QP-C54-T2-015 | `floating_point` now reads "NONE on the enclosure path" and names the binary64 [NUM] slack the record already describes |
| E-T3-N2 | QP-C54-T3-001, -004, -008, -014, -035, -037, -038, -039, -040 | `standalone_basis: ADDED_TO_REVIEW` |
| E-T3-N3 | QP-C54-T3-015 | `floating_point` now reads "USED (Decimal, non-directed rounding; no binary float)" |
| E-T5-HELD | T5 held group `source_derivative_gaussian_20261025` | Math-#514 merged at `c88c34b2`; registration belongs with the kernel-index lane |
| E-A3-T2-029 | A3 advice for QP-C54-T2-029 | Adds the executed replay: 0 shortfalls on the published path, latent defect, patch routed to Math-#219. The verdict stays INDETERMINATE. |

The remaining nits were README corrections, made in each tranche's README. Two were not taken: T3 nit 4, kept as the
reviewer judged acceptable, and T6's `**` markup in quoted reasons, which are not checked statements.

## Items from Claude 54's own adversarial reviews (`E-R-*`)

The 24 `E-R-*` items come from two same-family adversarial reviews of catalog candidate v2: round 1 (workflow
`wf_1e2e73e5-c9f`) and round 2 (`wf_26ea8588-f37`, which also re-checked the round-1 items). It ran
four lenses: builder fidelity, validator gaps, the tranche-4 assembly, and catalog-level double dispositions. Each finding was
then reproduced by an independent refuter that defaulted to "not real". Only the reproduced findings are applied. These items
are same-family work and await a different-family check.

| Ids | Row(s) | Change |
|---|---|---|
| E-R-T4-025-KIND, -TARGET | QP-C54-T4-025 | NEW_RECORD with no target. Q-2012 mentions the audit only in a `nonclaims` quotation. |
| E-R-T5-003-KIND, -TARGET | QP-C54-T5-003 | Pinning amendment to adopted Q-1146, which already quarantines the `CapFirstExit.lean` package |
| E-R-T1-015-018-KIND, -TARGET | QP-C54-015 to -018 | Pinning amendments to adopted Q-3008, which quarantines the four preserved sources |
| E-R-T4-005-KIND, -TARGET | QP-C54-T4-005 | Pinning amendment to adopted Q-10085, the legacy alias of the same lemma |
| E-R-OVL-T4-021 | QP-C54-T4-021 | `claim_overlaps`: the transcribed Theorems A, B, C and thm:cap-main are held by Q-1011 and Q-1014; narrow on adoption |
| E-R-OVL-T4-018, -T4-020, -T2-029, -T2-008 | T4-018 with T2-029; T4-020 with T2-008 | `claim_overlaps` both ways: the Math- packets are primary for the enclosures and closed forms the supplement transcribes; Q-2016 holds the c_{2,24} and c_{3,24} enclosures |
| E-R-OVL-T4-012 | QP-C54-T4-012 | `claim_overlaps`: D₂ = 29/6 − √6 is held by Q-10041 and Q-1001, and is CLOSED in the (s, x, y) form as K-SPEC-075 with K-SPEC-045 and K-MAIN-024 |
| E-R-OVL-T5-001 | QP-C54-T5-001 | `claim_overlaps` to Q-3007, which restates the companion's formal content at summary level. Kept as a new record because Q-3007 is a README summary, not a Lean-package claim |
| E-R-OVL-T4-023 | QP-C54-T4-023 | `claim_overlaps`: item (a) is the d = 2 case of the cap theorem held by Q-1014 and Q-10013 |
| E-R-OVL-T4-010, -T4-015 | QP-C54-T4-010, -015 | `claim_overlaps`: Theorems A and C of the parent are held by Q-1011; (3.13a) is held by Q-1048 and Q-1076 |
| E-R-OVL-T6-005, E-R-T6-005-REASON, -BLOCKING, -RESUMPTION | QP-C54-T6-005 | `claim_overlaps` to Q-10041 and Q-1001, and to the CLOSED declarations K-SPEC-075 and K-SPEC-045 (K-MAIN-024 on main), which prove the identity for every ConeMomentInterface inhabitant (no inhabitant is constructed; alignment pending). The reason, blocking reason and resumption cost are corrected. Round 1 had wrongly said the identity was not CLOSED; round 2 caught it. |
| E-R-OVL-T6-007, -T6-008 | QP-C54-T6-007, -008 | `claim_overlaps`: inside adopted Q-9001's engineering regression region, which does not quarantine their mathematics |

In round 2, the citations in E-R-OVL-T4-018 and -T4-020 were corrected and the T4-021 adoption note was completed. Each
corrected item's `source` says so.

`claim_overlaps` entries are typed by relation, as the candidate's adoption boundary defines:
- HOLDS_* and PRIMARY_FOR_TRANSCRIBED_PART narrow the row;
- TRANSCRIBED_BY marks the row as primary;
- CLOSED_DECLARATION names a CLOSED kernel declaration;
- INSIDE_ADOPTED_ENGINEERING_REGION is a location note only.

The builder rejects an erratum whose target matches no row, a section target that matches more than one row, and an a3 target
that was not audited. The validator's check 13 recomputes every item against the candidate. It confirms the tagged rows (for a
section target, its tranche and object), the values, and that every `errata_applied` tag names an item.

