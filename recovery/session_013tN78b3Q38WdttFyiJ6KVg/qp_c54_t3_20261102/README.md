# QP-C54 tranche 3: the 102 Math- review directories, and the A3 audits (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6566788130. No packet code was run.

## Part 1: the 102 `reviews/` directories that no record names (Math- `19a55750`)

Facts are in `rb1.json`–`rb3.json`: each directory's files, with blob, bytes, SHA-256 and Markdown headings.

**Pipeline.**
1. A classifier per batch sorts each directory as REVIEW_OF, STANDALONE_CLAIM or NOT_OBLIGATION, with file:line evidence.
2. An independent verifier per batch corrected **75 problems** (lens `fidelity`).
3. An extra, independent Layer A pass over every standalone record (`workflow_t3b.js`, batches `t3la1.json`–`t3la4.json`,
   output `t3b_layera_result.json`) corrected **172 problems** (lens `layerA`) and reclassified one directory,
   `reviews/pr25_typed_transfer_nonauthor_20260925`, from standalone to REVIEW_OF.
   - Its fixes cover floating-point classes judged from code, our-voice certification language, missing open steps in
     `exact_boundary`, A8 where a directory's own tool is the only certifier, and procedural blocking reasons.
4. `assemble_t3.py` checks mechanically:
   - every quoted statement is verbatim at its lines;
   - every directory appears exactly once;
   - every REVIEW_OF names what it reviews;
   - every standalone record has statements and a classified floating-point field;
   - the bindings come from Git.

   It also resolves every reviewed object against the candidate catalog, staging's CLOSED declarations, the
   `non_obligation_support` rows and the pinning sweep.
5. Rerunning the assembler on `verdicts_verified_t3.json` reproduces `records_verified_t3.json` byte for byte. The
   inputs and their SHA-256 values:
   - the catalog candidate built by `../qp_c54_catalog_candidate_20261102` (`0f956ace…`);
   - staging's `all-closed-declarations.json` (`c25b7e46…`);
   - `../pinning_sweep_20261102/pins.json` (`854cc400…`).
6. Manual checks:
   - every NO_SCRIPTS field matches a directory with no Python;
   - an overclaim lint found 23 hits, all of them "closed form", "closed declaration" or A8 sentences such as "certified
     only by the directory's own tool";
   - the privacy scan is clean.

**Result (`records_verified_t3.json`).**

| Class | Count | What it means |
|---|---|---|
| STANDALONE_CLAIM | **41** | `QP-C54-T3-001` to `-041`: PROPOSED records in the catalog's schema. Floating point: NONE 33, USED 5, NO_SCRIPTS 3. |
| REVIEW_OF | 37 | No record is needed. Every reviewed object that exists has a disposition: adopted, proposed, CLOSED package, non-obligation support, or a pin through the sweep. Seven resolve only through record text; 34 references are PR numbers or claim IDs and stay unresolved. |
| NOT_OBLIGATION | 24 | Protocols, workflow repairs, custody audits and similar records, each with a cited reason |

The standalone claims include, for example:
- new rational enclosures (T3-001, the Math-#223 closed forms at 2⁻²⁴⁰; T3-041, SIDE24 d=2 at small L);
- an erratum (T3-012, DL-AXIS-K-01);
- a retracted conjecture (T3-005);
- many D2, LM006 and IBA2-012 lemmas.

None of these directories is named by any catalog record.

## Part 2: the A3 audits (`a3_audits.json`, `a3_sliver_check.py`)

These are read-only interval-arithmetic audits of QP-C54-T2-008, -016 and -029, each followed by an independent
cross-verifier.

- **T2-008 `c2_exact`: SOUND_AS_INTERVAL_REASONING.**
- **T2-016 `c8_elder_failure_coefficient`: SOUND_AS_INTERVAL_REASONING.**
- **T2-029 `cusp_coefficient_certified`: INDETERMINATE.** `d3._sub` (`d3.py:40-41`) can end a v-panel 2⁻⁵⁷ short of `b` on
  the published d = 3 path. `a3_sliver_check.py` reproduces this without running the packet's code.
- These verdicts are advice for the records' A3 field, not certifications.

## Limits

- All agents are Anthropic Claude. This is engineering accounting, not a review of the mathematics, and no record is a
  verdict on any directory's mathematics.
- The REVIEW_OF / STANDALONE_CLAIM line is a judgement, and each classification is open to challenge.
