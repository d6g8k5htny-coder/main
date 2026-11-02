# Proposed §11 records, tranche 2 (QP-C54-T2): 53 Math- objects

By `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0; scientific effect
NONE. These are proposals for main#695's `cbz/quarantine/catalog.json`; they close and admit
nothing. The objects are the Math- computational packets and proof-note directories with no
disposition that `../federation_math_census_20261101/` found at Math-
`19a55750451b8016b50d1d20be6a18c021807ecb`. 610 of the 611 bound files are byte-identical at the
catalog cut `8151f592`; the exception is `tools/check_mode_replay.py`.

- **`records_verified_t2.json`**
  - `records`: 46 PROPOSED records.
    - 44 are new (`proposal_kind` NEW_RECORD).
    - 2 are pinning amendments: QP-C54-T2-031 pins Q-1043, and QP-C54-T2-034 pins Q-1025. Each
      pins the bytes that the existing claim-level record names but has not pinned.
    - Every record carries verbatim statements with `file` and line numbers, hypotheses,
      dependencies, the source's own limits, `finite_checks`, `floating_point` and a complete
      `source_binding`.
  - `not_obligations`: 7 objects that make no mathematical claim of their own, each with its
    cited reason: `PROOF_INDEX.md`, `docs/integration`, two Lean control-run evidence directories
    for the CLOSED determinant package, `local_pairing` exploration, a historical delivery README,
    and `tools/` checkers.
  - `verification`: how the records were made and checked.
- **How they were made:**
  1. One draft per batch.
  2. An independent fidelity pass, then an independent Layer A pass. No verifier checked a record
     it wrote. The passes fixed 147 and 50 problems.
  3. Then `assemble_t2.py`. It requires each statement to equal the lines of the file it names,
     byte for byte; covers every object; checks every floating-point field; and checks that each
     pinning title names an existing record. It takes every binding from `facts.json`.
- **Floating point:** NONE 29, USED 13, NO_SCRIPTS 4. Three of the USED records (T2-008, T2-016,
  T2-029) use binary64 interval arithmetic widened by one ulp per operation. Whether that is
  "justified interval reasoning" under A3 needs its own audit; it is not decided here.
- **Independence:** all agents are Anthropic Claude.
- **Reproduce:** `python3 -I -B assemble_t2.py MATH_REPO facts.json verdicts_verified_t2.json OUT.json`.
- **`singlepass_unverified/`** holds the earlier drafts, kept for history. **Do not adopt them.**
