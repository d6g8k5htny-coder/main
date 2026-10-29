# C183 scope-perimeter source crosswalk

Status: **engineering evidence only; scientific effect NONE; NOT A RESULT**

This note supplements the source-accounting perimeter of
[`RETRO-MATH-BIND-01.md`](RETRO-MATH-BIND-01.md), without changing that
inventory or its denominator. The frozen inputs are:

- `d6g8k5htny-coder/main@897b6fb1a4a09bce87f98f172574af6b9c5cebc5`
  - `docs/retro/RETRO-MATH-BIND-01.md`, blob
    `1af2fb1f8aad2e7036e3d142758f1caa5c8c61b5`
  - `docs/retro/RETRO-MATH-BIND-01.slices.json`, blob
    `3c9f374dbccfd6f902a5ad252d6ff46df622c717`
- `d6g8k5htny-coder/Math-@9f3fb6918b28e6378c034ebc19e76d432f5391cb`

## Why these files are outside the 313-unit denominator

RETRO-MATH-BIND-01 counts 313 research artifact *units*: selected second-level
directories under the Math- roots `frontiers/`, `reviews/`, `imports/`,
`coefficients/`, `certificates/`, `companions/`, `formal/`, `repairs/`,
`specifications/` and `claims/`, plus selected `main` experiment/formal
units. It records 22 graph-tied units and 291 orphan units.

The eight C183 objects below are standalone files under Math- `docs/`,
`tools/` and `tests/` (including `tests/fixtures/`). Those roots/files are
not units in that directory-based denominator. Their absence from the 313 rows
therefore does **not** mean that a mathematical artifact, graph node or
implementation is missing. It is a scope-boundary observation.

## Frozen public source chains

The arrows below record explicit repository relationships and file roles. They
are not v2 bindings and do not assert that a program has implemented, executed
or validated a theorem.

### Q2/Q5 supplied-data preparation

| Role | Immutable Math- object |
|---|---|
| Documentation | [`docs/C183_EXPERIMENT_INPUTS.md`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/docs/C183_EXPERIMENT_INPUTS.md), blob `5e6816e41278e454131e7b830bf933d640921a38` |
| Offline supplied-data helper | [`tools/c183_experiment_inputs.py`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tools/c183_experiment_inputs.py), blob `34cb241b65cc8d56bb214a9ebd7c1dcab6b3dd1e` |
| Synthetic controls | [`tests/test_c183_experiment_inputs.py`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/test_c183_experiment_inputs.py), blob `712fb5ceac116b964243487666f1f3d37e4cb87e` |
| Synthetic fixture | [`tests/fixtures/c183_experiment_input_cases.json`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/fixtures/c183_experiment_input_cases.json), blob `cf03cdb0643ab673657103463da7344919017386` |

Recorded relationship:
`documentation → supplied-data helper → synthetic controls → fixture`.

The documentation and helper explicitly preserve
`collector_qualification=NOT_RUN`, `live_payload_suppression=NOT_RUN` and
`experiment_execution=NOT_RUN`. The helper describes itself as supplied-data
preparation, never environment acquisition. These files do not attest a live
collector, payload suppression or completion of the proposed experiment matrix.

### Q1/Q6 offline qualification ledger

| Role | Immutable Math- object |
|---|---|
| Documentation | [`docs/C183_QUALIFICATION_LEDGER.md`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/docs/C183_QUALIFICATION_LEDGER.md), blob `cd4d5e2c136d736ccabecd807a75ef33d944e11e` |
| Offline ledger helper | [`tools/c183_qualification_ledger.py`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tools/c183_qualification_ledger.py), blob `ec5dc2f28702c83e5d765fcd91978da12c5b2e3e` |
| Synthetic and historical-byte controls | [`tests/test_c183_qualification_ledger.py`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/test_c183_qualification_ledger.py), blob `004e702276434e57a5447afa7e605d66a64b4506` |
| Fixture | [`tests/fixtures/c183_qualification_ledger_cases.json`](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/fixtures/c183_qualification_ledger_cases.json), blob `adffcf08ff9787e3dc8bd6cd08ff8bfdd8f37568` |

Recorded relationship:
`documentation → offline ledger helper → controls → fixture`.

The helper states that it is never an experiment runner. Missing archive,
source and historical-member inputs remain `NOT_RUN`; the tests distinguish
synthetic controls, supplied historical bytes and experiment execution. The
fixture's retained historical outputs and synthetic cases are evidence inputs,
not a fresh replay or scientific verdict.

## Retained boundaries

- Math- PR #397 remains **HOLD / NOT_READY / NO_MERGE**; this note does not
  determine or alter its historical cleanup cause.
- Qualification and experiment execution remain **NOT_RUN** where the source
  says so. No command in either chain was executed for this crosswalk.
- No source in Math- was changed. No private source, Drive or Dropbox object was
  read or republished.
- No `GRAPH.json` node, `architecture/graph-artifact-bindings.v2.json`
  record, scientific register, status/index/claim, proof byte or acceptance
  predicate is created or changed.
- A future graph node or typed binding would require its own exact-source
  proposal, eligible nonauthor review and guarded integration. This note
  supplies no such authorization.
- The 313-unit, 22 graph-tied and 291 orphan counts remain the historical
  results of RETRO-MATH-BIND-01 at its stated cuts. This supplement is not a
  revised census.
