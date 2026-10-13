# Architecture conformance — contributor reference

[Back to the repository overview](../README.md#federation-source-contracts).

The command details and dated observations below are retained from the
[reviewed README section](https://github.com/d6g8k5htny-coder/meta-framework/blob/9de940209690d5a128832a685ed51b3b2c178781/README.md#federation-source-contracts).
The C146 audit and subsequent diagnosis remain attributed to their linked
records; this documentation move does not rerun either. Run the commands from
the repository root.

## Federation source contracts

Versioned transport contracts live in [schemas/](../schemas/). They define source references, repository roles, source manifests, and workspace snapshots. These schemas describe identity and routing only; they do not establish theorem truth or scientific status.

The read-only checker [tools/architecture_conformance.py](../tools/architecture_conformance.py) checks declared repository owners and visibility labels, required local paths, reference syntax, optional manifest/snapshot structure, and architecture-authority limits. It does not authenticate the actual sibling checkout commits, recompute the declared artifact digests, or query live permissions. It never writes scientific state.

Run the local fixture tests from this repository:

```bash
python -B -S -m unittest discover -s tests -p 'test_*.py' -v
```

The separate workspace command is conditional on deliberately prepared inputs, not a claim that current default-branch clones form a supported layout:

```bash
python -B -S tools/architecture_conformance.py --workspace /path/to/eight-repo-workspace
```

**Source-cut qualification, 7 October 2026** (checker read at `f063d9dcab51302aaaef6666245cac9cf2307548`): the workspace must supply `meta-framework/registry.json`, the eight sibling directories, and `main/architecture/scientific_state/v1/AUTHORITY_MAP.json`. The required map has a [historical source at main `ebedb780`](https://github.com/d6g8k5htny-coder/main/blob/ebedb7802024fa557e9071e4c9cec7cddc474b89/architecture/scientific_state/v1/AUTHORITY_MAP.json), landed by [main PR98](https://github.com/d6g8k5htny-coder/main/pull/98) into `chatgpt/drive-github-hardening-20260919`, not default `main`. That locates one historical input; it does not establish a complete, compatible eight-repository source cut or authorize copying the map into a current authority surface.

The C146 audit's default-layout run at main `3bcf0e34b15dc38270b163263d62b690884e68df` returned exit 2, `missing main authority map`; [issue15](https://github.com/d6g8k5htny-coder/meta-framework/issues/15) and the [source-bound diagnosis](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6047261217) retain that evidence. The supported historical-versus-current layout and its source-bound positive, missing-map and wrong-cut tests remain unresolved there. This documentation supplies neither a newly passing layout nor a new source-binding policy.

`WORKSPACE_SNAPSHOT.json` and per-repository `SOURCE_MANIFEST.json` are optional inputs to this checker. Their commit/hash fields receive syntax and structure checks, not comparison with the actual checkouts or payload bytes. The [example snapshot](../examples/workspace_snapshot.json) contains repeated-digit placeholder commits, not reproducible pins; the [conformance tests](../tests/test_architecture_conformance.py) use synthetic directories and an authority-map fixture. Passing those fixtures does not verify a real federation cut. Even structurally valid records can describe the wrong local cut without this checker detecting that mismatch.

Unknown schema major versions, mutable refs, sandbox leakage, scientific-status fields in public registry artifact rows, and architecture ownership of promotion fields fail closed. These refusals do not supply the missing checkout authentication. Existing sandbox path/catalog exclusions and publication boundaries remain unchanged; do not fetch exploratory payloads to manufacture a conformance pass.

## Producer form and reader acceptance

The four schemas in [schemas/](../schemas/) state the **canonical producer
form** of a contract document; the validators in
[tools/contract_validation.py](../tools/contract_validation.py) define
**acceptance**. Decision record: [issue 15](https://github.com/d6g8k5htny-coder/meta-framework/issues/15)
(finding 6129626684; decision 6133399141 as corrected by 6133564511).

- A canonical producer document satisfies the schema profile (every key the
  schema lists in `required`, with its declared type; a property may be `null`
  only where the declared type includes `null`, today `runtime_package` only)
  **and** the Python semantic and safety checks in the relevant mode
  (`public=True` is the default for source references and manifests). Schema
  validation alone is insufficient: the schemas do not express the path,
  whitespace, uniqueness, repository-set and public-mode rules that the
  validators apply. The five files under [examples/](../examples/) are the
  verified canonical instances.
- Readers apply exactly two compatibility allowances beyond the canonical
  form. `schema_version` may be any major-1 decimal spelling that the validator
  grammar admits (`1.1`, `1.10`, and the tolerated leading-zero forms `01.0`
  and `1.00`; `2.0`, `1`, `1.0.0` and non-strings are refused). A repository
  role may omit `runtime_package`. An omitted key is never synthesized:
  `validate_repository_role` returns a shallow copy whose key presence mirrors
  the input, so consumers read it with `.get`.
- Nothing else the implementation happens to accept is part of the contract.
  Known reader looseness outside the two allowances (for example the string
  coercion of integer-valued hex fields) is a defect for a reviewed repair, not
  an allowance. Equally, refusing the leading-zero spellings or requiring
  `runtime_package` at read time would be a breaking acceptance change and is
  made only deliberately, never by drift.
- The schemas are not a substitute reader: a tool that validates incoming
  documents with them alone refuses documents the project accepts and accepts
  documents the project refuses.

[tests/test_contract_compatibility.py](../tests/test_contract_compatibility.py)
pins both sides: the version grammar and omission semantics of the validators,
the canonical form of the examples, and the literal schema facets (`const`
`"1.0"` in all five `schema_version` declarations; `runtime_package` required
with type `["string","null"]`). A change to either side fails a named test and
reopens this record. None of this changes scientific status, sandbox
exclusions or publication boundaries.
