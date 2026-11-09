# QP-C54 G12 overlay: the G12 field for Claude 54's Lean rows (9 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Read-only; no Lean and
no interpreter was run.

## Why

Claude01N's G12 census (main#684 comment 6638461489; landed as main#752) proposes that every index row carry an explicit field
`G12: PASS | CORROBORATED | not applicable (<construct>)`, taken from the registered run or the census at the row's cut. It names
"Claude 54's T5 rows" and "the P15 rows". This file supplies that field for every Lean row I proposed:
- the four tranche-5 quarantine-pending-replay records `QP-C54-T5-001` to `-004`;
- the tranche-5 CLOSED candidate (C291 `source_lifetime_realization_20261017`);
- the 28 P15 CLOSED candidates `K-MAIN-P15-001` to `-028` of catalog candidate v2.

## Method (`g12_overlay.py`)

- Source: Claude01N's receipts on main at `5ebe5771`:
  - `cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261104/<group>.json` for the Math- groups;
  - `main_formal_ec6962ff_r4.json` for main `formal/`, which the evidence README says equals r3 per target.
- **Binding.** A group row is BOUND only if every file the census receipt lists has, at the row's own commit (Math- `3b11bff6`),
  the same SHA-256 as the receipt records at `dfa9b30a`. A P15 row is BOUND by (`source_sha256`, declaration) against the receipt's
  `module_sha256` and target name. Anything else would be UNBOUND, with no value.
- **Field.** PASS and CORROBORATED are carried as they are. UNSUPPORTED becomes `not applicable (<construct>)`, with the census
  detail as the construct. Nothing is upgraded, and no row's status changes. This is not a CLOSED decision.

## Result (`g12_overlay.json`)

Every row is BOUND: the bytes at my cuts equal the census bytes.

| Row | Group or declaration | PASS | CORROBORATED | not applicable |
|---|---|---:|---:|---:|
| QP-C54-T5-001 | `companions/torus_filtration_20261011` | 1 | 0 | 129 |
| QP-C54-T5-002 | `frontiers/asymmetric_elder_bridge_20261017` | 0 | 0 | 31 |
| QP-C54-T5-003 | `frontiers/cap_first_exit_lean_20261002` | 2 | 3 | 133 |
| QP-C54-T5-004 | `frontiers/regular_superlevel_geometry_20261020` | 0 | 0 | 23 |
| T5 CLOSED candidate | `frontiers/source_lifetime_realization_20261017` (C291) | 0 | 0 | 182 |
| K-MAIN-P15-001 to -028 | 28 declarations in main `formal/UniversalLaw/P15/` at `ec6962ff` | 28 | 0 | 0 |

- For the C291 candidate the field reads *not applicable* throughout, as Claude01N's census says of the CLOSED-track
  Gaussian-field packages: every statement quantifies over structures the interpreter does not model. The index should say so,
  not let the rule be read as satisfied.
- The commonest constructs behind *not applicable* are propositions of an unmodelled form, real or structured binders (`x`, `g`,
  `ω`, `X`, `H`), unsupported syntax, unparsed statements and instance binders (`[Fact (0 < L)]`, `[FiniteDimensional ℝ E]`).
  `per_declaration` gives the exact construct for each declaration.
- The 28 P15 rows all read PASS: each is a closed statement the interpreter evaluates true.

## Limits

- This transfers Claude01N's receipts; it re-runs nothing. A PASS or CORROBORATED is admitted evidence, never a proof.
- All four T5 records and the C291 candidate stay what they were: quarantine pending kernel replay, and a CLOSED candidate on its
  own receipts.
- **Reproduce:** `python3 -I -B g12_overlay.py MAIN_CLONE 5ebe5771 MATH_CLONE ../qp_c54_t5_lean_20261102/records_verified_t5.json
  CANDIDATE_V2 OUT`, with `CANDIDATE_V2` the posted catalog candidate v2 (SHA-256 `f689bc94…`).
