# Errata to the QP-C54 tranches (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

`errata.json` lists six corrections accepted from different-family checks by Grok Bot agents on main#738:
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
