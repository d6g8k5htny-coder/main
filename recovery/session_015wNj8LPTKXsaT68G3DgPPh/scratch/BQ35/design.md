## BQ-28 initial design COMPLETE / own RELEASE — preserve raw rows, validate endpoint identities

Dylan Roy — delegated AI architecture/engineering work. Actual OpenAI / GPT-6 Astra Pro, same current invocation that posted acknowledgement6028941682, `research-loop-bq28-20261007T0130Z`; platform session UNKNOWN, source/coordination-exposed, credit0, scientific effect NONE. Token `BQ28-R1-20261007T0119Z-28b5`, confirmed6029021566. I read the confirmation and refreshed changes since its contention boundary before material work. Initial design completed before the02:00UTC checkpoint/02:45 end. This is NOT a patch, source-review verdict, complete collector replay or integration receipt.

### Exact inputs and actual execution

Frozen main `cddbb7f6cf3f57f3b495277f7da148287e027b19`:
- `tools/release_snapshot.py`:14144B, Git blob `8cba82a59231248ab597627d9caee2ee7431eed7`, SHA256 `9f12296f070f95f0d28b8f3ce6926c949657e8a67347dae04077485f6cd51d3f`.
- `tests/test_release_snapshot.py`:12336B, Git blob `3694a85a842283e611f977d2b48cd6b8d02ab8dc`, SHA256 `0c4ef3b4988be52ae9201ccd0468413416a13cd2054212d2145fc55b4f896c78`.
Both complete native UTF-8 bodies were transferred to scratch and independently Git-framed-hashed before import and after execution; unchanged. Release guide blob47c2e5820095360f2c0230893086ee8e144ec6ef was read as reference. BQ-18 result6028533315 remains GROK-H01's original repeated/omitted-ID observation, reused here, not my discovery.

Actual commands: `python -B -S /mnt/data/bq28/design_probe.py` and `python -B -O -S /mnt/data/bq28/design_probe.py`, CPython3.13.5, at01:46:11UTC October7, both exit0. Each exercised **44 disposable design-model cases** plus **four unchanged tests** (`test_missing_key_not_empty`, `test_error_not_empty`, `test_wrong_page_type`, `test_keyed_bad_total_count`), four OK/zero failures/errors. The44 observations agree exactly between modes. This is a separate proposed validation model composed with unchanged `pages()`/`pr_number()`/`write_json()`, NOT a modified collector. `collect()` and Client were not invoked; no real API collection/network/credential use or archive build. No full21-method suite, repository CI or successor GREEN is claimed.

### Complete caller → proposed identity map

AST and full-source reading identify one production `pages()` invocation inside `collect.read`, reached by these **eight** paginated call sites. Identity scope is each repository/resource collection separately, not a global set.

| Source line / resource | JSON wrapper key | Object identity |
|---|---|---|
|225 `/pulls?state=open`|none|positive integer `number`, scoped to this repository and already used by `pr_number()` / `pr-{n}` destinations|
|247 `/pulls/{n}/files`|none|exact nonempty string `filename`|
|248 `/pulls/{n}/reviews`|none|positive exact-integer `id`|
|249 `/issues/{n}/comments`|none|positive exact-integer `id`|
|250 `/pulls/{n}/comments`|none|positive exact-integer `id`|
|251 `/commits/{sha}/check-runs`|`check_runs`|positive exact-integer `id`|
|253 `/actions/runs?head_sha={sha}`|`workflow_runs`|positive exact-integer `id`|
|259 `/rulesets`|none|positive exact-integer `id`|

PR-number identity deliberately follows the existing repository-local routing contract; it does not authenticate the independent global `id` field. Changed-file identity is NOT its blob SHA, previous filename, normalized path or payload equality. Equal blobs at different filenames remain two legitimate rows. Key sets reset between collections; the same numeric ID in different resources is not a cross-resource collision.

### Minimal two-file successor design — approval still required

**Prefer post-write validation through the existing `read(..., validate=...)` hook**, rather than silently deduplicating or inserting an early pagination exception that discards successfully returned raw rows. The present ordering is `pages → write_json → validate → consume`; it already catches validation exceptions into the resource-specific `SUMMARY.errors`, returns None for the unusable collection and continues other resources/repositories. Add a small identity checker and wire explicit policies at all eight call sites, using only `tools/release_snapshot.py` and `tests/test_release_snapshot.py`.

Preserve `pages()` retrieval/count-drift/short-page/safety-cap behavior and existing diagnostics, Client/allowlist/redirect behavior, archive/fresh-output rules, output names/schema and status-authority boundaries. Generic `pages()` remains a raw pagination primitive, NOT a semantic-completeness certificate. The new guarantee would be at collector acceptance, after complete retrieval and before consuming the rows. A pagination/network failure before return keeps its current behavior; this design does not invent retained HTTP pages or an atomic snapshot.

**Malformed-key policy:** id/filename collections require object rows and the named key; IDs are `type(value) is int` and >0, not bool/float/string; filenames are nonempty strings compared verbatim, without case folding, stripping, Unicode/path normalization or reusing archive-path restrictions. These metadata filenames are not extracted/executed. Refuse duplicate identities even if their payloads are equal; report identity key and both observed positions. No first/last-wins choice.

**Preserve the special existing PR error path:** malformed/missing/nonpositive PR numbers remain the later `pr_number()` loop's per-row errors, with valid following PRs still processed. The uniqueness check considers only numbers that satisfy that SAME function; it must not recategorize malformed rows as valid evidence or silently delete them. Repeated valid PR numbers refuse the ambiguous collection before any repeated per-PR writes. The existing loop still makes `complete:false` for malformed rows. Tests must cover an invalid row followed by a valid one so a new generic validator cannot accidentally suppress the established partial-progress behavior.

**Raw-evidence meaning:** keep the original decoded rows, order and fields in the existing JSON file before identity refusal; do not drop duplicate rows, rewrite values or replace them with an empty successful list. The existing serializer is not original HTTP wire bytes, and wrapper/page/header preservation is not promised. `SUMMARY.errors` remains authoritative for completeness; a file's presence or matching hash alone is not success.

### Discriminating observations / required successor tests

The scratch model refused duplicates within/across pages, including repeated IDs with changed payloads and repeated filenames with differing blobs; accepted same-blob/different-filename rows and exact distinct lexical names; retained all raw rows and unmodified inputs. Bool/string/float/zero/negative/missing IDs and malformed filename rows were refused. Bad PR-number rows were explicitly deferred to the unchanged per-row error path, with the following valid PR retained. A101-row all-unique set that substitutes an unexpected102 for an unseen101 still passes: **uniqueness cannot establish an unknown expected set or atomicity**.

A later candidate needs parent RED/successor GREEN on actual mocked-collector acceptance for every one of the eight routes, retained duplicate/malformed raw rows, exact resource error, `complete:false`/CLI nonzero and later-resource/repository progress; canonical and equal-blob filename positives; invalid-then-valid PR behavior; unchanged pagination count/cap/permission tests. Assert all eight routes have explicit identity policies. Reason-specific sensitivity should catch a bypassed route, silent dedup and filename→SHA substitution. Run the complete applicable unchanged+new suite in both modes on the frozen candidate, then distinct eligible nonauthor review and fresh guarded integration. No author self-approval/merge.

Scratch probe identity:9882B/SHA256 `678e1e81c259354bc238862a3063911c7b50d428b21216e5548ee9ca7ad8875d`; normal report17752B/SHA256 `5de3ab3cd892641de3414a6df6d91a71cccf2d0b1550e0a9735afe766622e9a6`. Local paths are not shared repository carriers. The source map, policies, observations and limits above are the public deliverable.

**Own RELEASE of BQ-28 initial read-only design now.** No source/test/branch/PR/CI/cloud/settings/scientific record was changed. Coordinator: please give the design an explicit disposition before assigning any two-file author phase; I am available for that distinct phase but hold no writer claim. I am immediately rereading the live queue for the next eligible nonconflicting task, not extending this scope by inference.
