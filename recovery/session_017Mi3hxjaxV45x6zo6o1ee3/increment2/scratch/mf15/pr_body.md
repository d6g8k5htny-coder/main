Decision record: issue #15, finding 6129626684, decision [6133399141](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6133399141) as corrected by [6133564511](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6133564511); coordinator activation [6133578956](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6133578956) (its two wording corrections are applied). Scientific effect NONE. Organizational-independence credit 0.

## What this changes

The four schemas state the **canonical producer form**; `tools/contract_validation.py` defines **acceptance** with exactly two reviewed allowances: any major-1 decimal `schema_version` spelling the grammar admits (`1.1`, `1.10`, `01.0`, `1.00`), and an omitted `runtime_package` in a repository role (never synthesized). Schema validation alone is insufficient; nothing else the reader happens to accept is part of the contract (the integer-hex coercion is recorded as a defect for a separate reviewed repair).

- `README.md`: one sentence in "Federation source contracts".
- `docs/ARCHITECTURE_CONFORMANCE.md`: new section "Producer form and reader acceptance".
- `tests/test_contract_compatibility.py` (new, additive): 16 standard-library tests pinning the version grammar and refusal messages on all five examples (both `source_ref` branches), semantic refusals under accepted minor spellings, omitted versus explicit-null `runtime_package` with input/returned/copy behaviour, the canonical form of the examples, and the literal schema facets (`const "1.0"` in all five declarations, `runtime_package` required with type `["string","null"]`, required lists, sandbox enum membership).

No change to `schemas/`, `tools/`, `examples/`, workflows, `registry.json`, markers or the duplicate-loader branch; no existing test modified.

## Evidence at head `ede7eb11da68d9ccd84333105de6bd0d5a91579f` (parent `a7b12fbe`, tree `40420482`)

- `python -B -S -m unittest discover -s tests -p 'test_*.py'`: 60 tests OK (normal) and 60 tests OK (`-O`), zero skips; the new file alone: 16 OK in both modes.
- `git diff --check` clean; `git diff --quiet a7b12fbe HEAD -- schemas tools examples .github registry.json` confirms the excluded paths are untouched.
- Mutant controls in scratch copies, each caught by named tests: `VER` tightened to literal `1.0` (18 failures, 20 errors); a schema `const` edited (facet pin); `runtime_package` removed from the example (canonical-form pins); a reader mutant requiring `runtime_package` (omission pin); `runtime_package` dropped from the schema `required` (facet and required-list pins); an example `schema_version` changed to `1.1` (literal-`1.0` pin). The unmutated copy passes.

Blobs: `README.md` `aaac09311df95cb6c32922be2c96925ca7576644`, `docs/ARCHITECTURE_CONFORMANCE.md` `30521e421b92c9a5fd7f10fcd83cb854c0e20f4c`, `tests/test_contract_compatibility.py` `788e08e3b6a42adabb69a45ee1a8e729747a80b7`.

## Custody

Author: Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`, delegated for Dylan Roy. Head frozen at `ede7eb11` pending the separate nonauthor review and a distinct integration lane; the author does not mark ready or merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
