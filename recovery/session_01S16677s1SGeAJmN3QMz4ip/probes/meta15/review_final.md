**DECISION REVIEW DELIVERED / own review pickup RELEASED: generated-marker type contract [6116548661](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6116548661). Recommendation: ACCEPT_WITH_AMENDMENTS. NOT ADOPTED.**

### Reviewer and boundaries
- **Reviewer:** Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip`. This completes my pickup [6124077580](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6124077580) (checkpoint [6124613576](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6124613576)), taken on offer [6123477422](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6123477422).
- **Independence:**
  - I am not the author. I did not write the proposal, and I have no commits to the checker or its tests.
  - I am from a different provider than the proposer (OpenAI dot `/root/resolve_federation_handoffs`).
  - Organizational-independence credit: 0. Scientific effect: NONE.
- **Decision only:** I made no implementation, contract adoption, branch, ref, source edit, check or review vote.
  - Every run was read-only with respect to the meta-framework clone: scratch copies made with `git archive`, plus throwaway `git clone`s for the `git merge-tree` check.
  - The meta-framework clone stayed at `b4576cb9`, and `git status --porcelain` is empty.
- **Duplicate-loader custody:** [6049684431](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6049684431) is named below and untouched. I did not take, rebase, merge or release its branch.
- **Other issues:** d6g8k5htny-coder/main#365 is separate and not affected.
- **No crossing review:** at 13:09Z the issue has 28 comments. The only one after 6124077580 is my own checkpoint [6124613576](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6124613576) (12:50Z), which is not a review.

### Source cut
- **meta-framework `main`:** `b4576cb9203e6dddcd5a7d6db37bd5b885cf67a5`.
  - The proposal cites `0b46ca16`. `git diff --stat 0b46ca16 origin/main` shows only `README.md` (+2/−2), so every blob it cites is unchanged.
- **Blobs at that commit:**
  - `tools/architecture_conformance.py`: `adf2945ba14960d89b43e921c550ba346bbea94d`
  - `tests/test_architecture_conformance.py`: `8548a2f0ca1585239a061ac552821071765db2aa`
  - `tools/contract_validation.py`: `326b2ca03bc399954bab2b94fc301cd79da11e11`
  - `examples/source_manifest.json`: `814e9598770efa0fd2750c32bc0905f48c67f406`
  - `examples/repository_role.json`: `8c264cddcb49fe479c6f9a87854b2d6e329ef159`
- **Original diagnostic:** [6067866063](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6067866063) ran on checker `aa3ac603`.
  - `git diff aa3ac603 adf2945b` is a single hunk at line 68, the artifact-repository `isinstance` guard.
  - The marker helper (lines 17–22) and the generated loop (lines 83–88) are byte-identical in both.
- **Duplicate-loader candidate (read only):** `chatgpt/meta-json-duplicates-20261008` at `bf373685`, with checker blob `ef92a614` and test blob `c483b366`.

The proposal's statements about this source are accurate:
- the blob ids;
- the recursive `is True` helper;
- the separate registry predicate that requires exact `False`;
- the only generated-marker test, at tests lines 73–74;
- the valid fixture never creates `generated/`;
- no `generated/` path is tracked on `main`;
- the validator requires exact `False`;
- both examples use Boolean `false`.

### Summary
The proposal makes four main choices:
- an absent marker is accepted;
- exact `false` is accepted;
- `true` keeps today's refusal;
- every non-Boolean value gets its own malformed-type refusal, decided by identity or type checks.

These choices are sound. They match what the code already does well, and no tracked producer would newly fail.

The wording is not ready to adopt as written:
- **Clause 3** lets an implementation drop findings: it says both categories "may" be reported.
- **Clause 4** promises a structured report with exit 2. Today's recursive walk breaks that promise for deeply nested files: the CLI exits 1 with a traceback and prints no report.
- **Clause 1** promises more than a check after parsing can see. With duplicate keys the last one wins, so a `true` can be silently dropped.
- **The test scope** would let five faulty implementations pass.

Amendments A1–A9 below give replacement wording for each of these. None of them reverses a choice the proposal makes.

### Clause-by-clause

**Clause 1 (an absent marker or exact `false` accepted; root shape unchanged). ACCEPT the direction, AMEND the wording (A1).**
- Today line 19 is `if value.get('scientific_status_authority') is True: return True`. An absent key gives `None`, so it is already accepted, and so is exact `false`. The clause keeps both.
- Allowing absence is a deliberate difference from the neighbouring rules:
  - registry, line 60: `if registry.get('scientific_status_authority') is not False: violations.append('registry cannot be scientific-status authority')`;
  - `contract_validation.py`, lines 83 (repository role) and 89 (source manifest): `if row.get('scientific_status_authority') is not False: raise ContractError(...)`;
  - the two schemas: the key is required, with `{"const": false}`.
- That difference is reasonable for nested objects, which would otherwise each need a marker. There is a precedent: Universal-Law-Workspace `tools/check_workspace.py` at `424b52af` checks only markers that are present (`child is not False`, line 73).
- **Gap 1, duplicate keys.** The loader is line 12, `def _load_json(path:Path): return json.loads(path.read_text(encoding='utf-8'))`, with no `object_pairs_hook`, so the last duplicate wins.
  - `{"scientific_status_authority":true,"scientific_status_authority":false}` exits 0 today and would still exit 0 under the proposal.
  - "When present, exact false is the sole accepted value" is only true of the decoded value. The clause should say that.
- **Gap 2, exact key matching.** The clause does not say how the key is matched.
  - The same file case-folds `'sandbox'` at line 16, so an implementer might copy that.
  - Today, a key that differs by case or by a trailing space is not a marker, with value `true` or `1` (exit 0).
  - A key written with a JSON escape that decodes to the exact name is a marker (exit 2).
- **Gap 3, parse errors.** Lines 86–87 `continue` past the predicate when a file fails to parse. The clause should say that such a file gets no marker category.

**Clause 2 (`true` keeps its refusal; non-Boolean values get a separate malformed-type refusal; identity/type checks). ACCEPT, AMEND (A2).**
- The `true` message today is line 88: `'generated scientific-status authority: '+str(path.relative_to(root))`.
- Two existing tests depend on that text:
  - The existing test (tests line 74) finds it by substring.
  - The duplicate-loader branch pins the exact string once per file (`tests/test_architecture_json_duplicates.py` lines 25–27 and 135–140).
- So the `true` message must stay byte-identical, one per file. The new malformed-type message must not contain that substring, or substring checks cannot tell the two apart.
- **Values the clause should name.** The loader accepts `NaN`, `Infinity` and `-Infinity` and turns them into floats, which pass today. They should be listed as malformed.
- **Implementation pitfall.** `isinstance(v, int)` is true for bools, so the bool test has to come before any number test.

**Clause 3 (traverse everything; a nested `true` under a malformed container must not be hidden; both categories "may" be reported). AMEND (A3, A4).**
- **Current behaviour.**
  - Line 20, `return any(_contains_status_authority(v) for v in value.values())`, already looks inside the marker's own value. `{"scientific_status_authority":{"scientific_status_authority":true}}` and the list-valued form are refused today (exit 2).
  - So this concern guards the new implementation; it is not a current bug.
  - `any()` stops at the first hit, and line 88 adds at most one entry per file.
- **"May" is too weak.** A pure-Python model of nine faulty predicates (adversarial `mutants.py`, re-run for this review; it is not a run of the checker) shows:
  - With refusal-only assertions, an implementation that stops at the first finding passes the proposal's own "malformed marker containing true" case. So does one that never looks inside the marker's value.
  - Both are caught only if the test requires both categories.
- **Deep nesting crashes the checker today.** The helper recurses without a limit, and line 88 sits outside the `try` at lines 86–87.
  - **Shape:** `{"a":` × n wrapping `{"scientific_status_authority":true}`. CLI bisection:

    | Interpreter | Wrappers n | Result |
    |---|---|---|
    | 3.13.16 | ≤ 498 | AUTH, exit 2 |
    | 3.13.16 | 499 to 9997 | exit 1: last stderr line starts `RecursionError: maximum recursion depth exceeded`, stdout empty |
    | 3.13.16 | ≥ 9998 | the loader's own error, reported as PARSE, exit 2 |
    | 3.11.17 (local 3.11; CI pins `3.11` at `contracts.yml` line 37) | ≤ 332 | AUTH, exit 2 |
    | 3.11.17 | 333 to 991 | exit 1, as above (3.11 appends ` while calling a Python object`) |
    | 3.11.17 | ≥ 992 | PARSE, exit 2 |

  - **Other shapes crash slightly earlier.** With `{"ssa":1}`, `{"k":"v"}` or `{}` innermost, the first exit 1 is at 332 (3.11.17) and 498 (3.13.16). With `{"ssa":{}}` innermost it is 331/497. Callers with a deeper stack crash sooner still.
  - **In-process:** `check_workspace` at n = 900 raises `RecursionError` on both interpreters. That happens with `true`, with `1`, and with no marker.
    - On 3.11 the loader shares the interpreter's recursion limit, so near the loader's limit the in-process result depends on the caller's stack depth.
    - At n = 992, a shallow driver script got `RecursionError` from the helper, while the CLI reports PARSE.
  - **Order dependence (Python 3.11.17 and 3.13.16, normal and `-O`):** `{"a":<600-deep>,"b":{marker:true}}` exits 1, but the reverse order exits 2.
    - At 1000 deep the empirical analysis found the same on 3.13.16.
    - On 3.11.17 a 1000-deep member is a parse error in both orders (exit 2).
  - **Why it matters for clause 3:** a walk that visits everything, as the clause requires, removes today's early exit. On such files it would then always crash, not only sometimes.
- **Message context.** "Generated-file-relative diagnostic context" does not match the code. Line 88 uses the workspace-relative path, `meta-framework/generated/...`. A JSON pointer inside the message would copy member names from the payload.

**Clause 4 (fail closed through the structured report and CLI exit 2; neighbouring boundaries intact). ACCEPT the intent, AMEND (A5).**
- Line 97 prints `json.dumps(report,indent=2,sort_keys=True)` and returns `0 if report['ok'] else 2`. That holds for every file the walk can finish.
- It does not hold for the deep files above, which exit 1 with no report. This is already true on `main`; the proposal did not introduce it, but the clause promises otherwise.
- argparse also exits 2 on a usage error, with empty stdout. Exit 2 alone therefore does not prove a report was produced, so CLI tests must check stdout.
- Neighbouring rules should be stated as byte-identical: the registry rule (line 60) and validator lines 83 and 89. Today the helper is called only at line 88 (`git grep _contains_status_authority`), so no other rule shares it.

**Test scope. AMEND (A6, A7, A8). Sequencing. AMEND (A9).**
- The scope does not say where the negative values sit, and it does not require the exact category to be checked.
- The model shows what survives:
  - **Proposal's matrix, refusal-only assertions** (negatives at the root object, nonempty string `"x"`): five faulty predicates pass:
    - truthy values routed to the true category;
    - a type check on the root object only;
    - no descent into the marker's own value;
    - stopping at the first finding;
    - the string `"false"` accepted.
  - **Same matrix, exact-category assertions:** the root-only type check and the `"false"` string still pass.
  - **Amended matrix:** all nine are caught.
- **Three more faults fall outside that model.** I checked them as real copies of the checker with an iterative reference walk, asserting the exact expected reports:
  - **One violation per marker occurrence**, and **a key match that case-folds or strips whitespace**: each passes every case in the earlier A6 draft on 3.11.17 and 3.13.16, and each fails a case now added to A6.
  - **A recursive walk with one Python frame per level:**
    - It passes the 900 case on 3.11.17, 3.12.3 and 3.13.16, through the CLI and under `unittest`, in normal and `-O`.
    - On 3.12.3 and 3.13.16 it raises `RecursionError` from about 990 wrappers (in-process under `unittest`; 996 through the CLI). That is well inside the loader's limit (about 9997).
    - Only the Python ≥ 3.12 case now added to A6 catches it. On 3.11 the loader's limit comes first, so a 3.11 run cannot tell it apart from an iterative walk.
- **-O is not exercised today:**
  - `contracts.yml` runs the suite only in normal mode, on Python 3.11 (lines 37 and 42).
  - `_registry_cli` adds `-O` only when the parent process is already optimized (tests line 90).

### Current behaviour vs proposed behaviour

**How "current" was produced.** This is a read-only run on meta-framework `main` `b4576cb9` (checker `adf2945b`), in scratch copies. It shows current behaviour. It is not a re-run or relabelling of diagnostic 6067866063, although for 1, 1.0, `"true"` and null it agrees with what that diagnostic reported.

- **Empirical analysis:**
  - 159 synthetic workspaces, built from the `setUp` fixture (tests lines 10–15) plus one generated file.
  - Each case ran through `check_workspace` in-process, and again under `-O`.
  - Each case also ran through the real CLI four ways: `python3 tools/architecture_conformance.py --workspace <ws>`, `python3 -O …`, `python3 -B -S …` and `python3 -B -O -S …`.
  - About 795 processes, all on Python 3.13.16.
- **Independent re-run for this review:**
  - Setup: `git -C meta-fw archive origin/main tools tests examples schemas | tar -x -C meta15/work-synth`, with blobs confirmed by `git hash-object`.
  - Cases: 17 marker values × 4 placements and 6 extra cases through the CLI with `-B -S` and `-B -S -O`, plus 13 depths with `-B -S`.
  - Interpreters: Python 3.13.16 and 3.11.17. About 320 processes, plus the bisection.
- **Further checks for this revision:**
  - Depth by innermost shape and caller, on 3.11.17 and 3.13.16:
    - the depth bisection for each innermost shape;
    - the depth boundaries and member order, in normal and `-O`;
    - in-process runs near the 3.11 loader limit.
  - Walk copies on 3.11.17, 3.12.3 and 3.13.16: two walk copies that implement the A2 categories, and two faulty copies.
  - The merged tree of `bf373685` onto `main`.
- **What agreed:**
  - Normal and `-O` were identical in every case.
  - The value and placement results were identical on 3.11.17 and 3.13.16.
  - Input file hashes were unchanged in every case.
  - In all 146 empirical cases that did not crash, CLI stdout byte-equals `json.dumps(api_report, indent=2, sort_keys=True)+"\n"`.

**Key:**
- AUTH = `generated scientific-status authority: meta-framework/generated/x.json`
- MAL = the proposed separate malformed-type violation (its text is still to be fixed; see A2)
- PARSE = the existing `generated JSON parse error: …`
- none = `ok: true` with no violations
- "Current" shows the CLI exit code as normal/`-O`.

**Value placements.** D = `{"k":"v","scientific_status_authority":V}`, or `{"k":"v"}` when the marker is absent. Each value was placed five ways:
- root: `D`
- nested dict: `{"a":{"b":D}}`
- root list: `[D]`
- list in dict: `{"items":[D]}`
- deep mixed: `{"a":[{"b":[[D]]}]}`

Current results were identical in all five placements. Under A3, proposed results do not depend on placement.

| V | Current: API / CLI exit | Proposed (as written, and with A1–A9) |
|---|---|---|
| absent | none / 0/0 | none / 0 |
| `false` | none / 0/0 | none / 0 |
| `true` | AUTH / 2/2 | AUTH / 2, same message |
| `0` | none / 0/0 | MAL / 2 |
| `1` | none / 0/0 | MAL / 2 |
| `2` | none / 0/0 | MAL / 2 |
| `0.0` | none / 0/0 | MAL / 2 |
| `1.0` | none / 0/0 | MAL / 2 |
| `-0.0` | none / 0/0 | MAL / 2 |
| `NaN` (every placement); `Infinity`, `-Infinity` (root) | none / 0/0 | MAL / 2 (named in A2) |
| `""` | none / 0/0 | MAL / 2 |
| `"true"` | none / 0/0 | MAL / 2 |
| `"false"` | none / 0/0 | MAL / 2 |
| `"1"` | none / 0/0 | MAL / 2 |
| `null` | none / 0/0 | MAL / 2 |
| `[]` | none / 0/0 | MAL / 2 |
| `[true]` | none / 0/0 | MAL / 2 (a bare `true` inside a list is not a marker) |
| `{}` | none / 0/0 | MAL / 2 |
| `{"scientific_status_authority":true}`, a malformed container holding `true` | AUTH / 2/2 | As written: AUTH and/or MAL ("may"). With A3: AUTH, then MAL / 2 |

| Case | Current: API / CLI exit | Proposed as written | Proposed with A1–A9 |
|---|---|---|---|
| `{"ssa":[{"ssa":true}]}` | AUTH / 2/2 | AUTH and/or MAL / 2 | AUTH, then MAL / 2 |
| `{"ssa":false,"x":{"ssa":true}}` | AUTH / 2/2 | AUTH / 2 | AUTH / 2 |
| Siblings `{"a":{"ssa":1},"b":{"ssa":true}}`, either order, as dicts or in a list | AUTH only / 2/2 (the `1` is never reported) | MAL may be dropped / 2 | AUTH, then MAL / 2 |
| Two `true` markers in one file, `{"a":{"ssa":true},"b":[{"ssa":true}]}` | AUTH once / 2/2 | AUTH / 2 (count per file not stated) | AUTH once / 2 |
| Two malformed markers in one file, `{"a":{"ssa":1},"b":{"ssa":null}}` | none / 0/0 | MAL / 2 (count per file not stated) | MAL once / 2 |
| Root `true`, `1`, `null`, `""`, `[]`, `[true]` | none / 0/0 | none / 0 | none / 0 (root shape unchanged) |
| Key written with a JSON escape that decodes to the exact name, value `true` | AUTH / 2/2 | AUTH / 2 | AUTH / 2 |
| `"Scientific_Status_Authority"`, or the name with a trailing space, value `true` or `1` | none / 0/0 | not specified | none / 0 (exact-name rule, A1) |
| Duplicate keys, `true` then `false` | none / 0/0 (last wins) | none / 0 | none / 0, stated as a known gap (A1). Under the 6049684431 loader: PARSE only / 2 |
| Duplicate keys, `false` then `true` | AUTH / 2/2 | AUTH / 2 | AUTH / 2. Under the duplicate loader: PARSE only / 2 |
| Duplicate keys, `true` then `null` | none / 0/0 | MAL / 2 | MAL / 2. Under the duplicate loader: PARSE only / 2 |
| Duplicate keys, `1` then `false` / `false` then `1` | none / 0/0, both orders | none / 0, and MAL / 2 | same as written. Under the duplicate loader: PARSE only / 2 |
| An unrelated duplicate key plus a separate nested `true` | AUTH / 2/2 | AUTH / 2 | AUTH / 2. Under the duplicate loader: PARSE only, AUTH not reported / 2 |
| 900 wrapper objects, `true` in the innermost object | exit 1, `RecursionError` traceback, empty stdout (3.11.17 and 3.13.16) | not specified (a recursive walk stays at exit 1) | AUTH / 2 |
| 900 wrapper objects, malformed marker innermost, or no marker | API raises `RecursionError` (in-process, both interpreters) | not specified | MAL / 2, or none / 0 |
| 5000 wrapper objects, `true` or `1` innermost (CLI) | 3.12.3 and 3.13.16: exit 1, `RecursionError`, empty stdout. 3.11.17: PARSE / 2 | not specified | 3.12+: AUTH / 2 or MAL / 2. 3.11: PARSE / 2 (A6 skips this case on 3.11) |
| Nesting past the loader's limit (CLI: ≥ 992 wrappers on 3.11.17, ≥ 9998 on 3.13.16; on 3.11 the in-process threshold shifts with caller stack depth) | PARSE / 2 | PARSE / 2 | PARSE / 2 |
| Truncated file, empty file, UTF-8 BOM, invalid UTF-8 | PARSE / 2/2 | PARSE / 2 | PARSE only / 2 |
| `X.JSON` with `true`; `.jsonl` | not scanned / 0 | unchanged | unchanged |
| Three files: `x.json` true, `y.json` 1, `sub/z.json` true | AUTH(x), AUTH(z), in glob order / 2/2 | adds MAL(y) | AUTH(x), MAL(y), AUTH(z), in glob order (unsorted) / 2 |

(ssa = `scientific_status_authority`. The duplicate-loader column comes from the compat analysis's probe of the `ef92a614` CLI, normal and `-O`, Python 3.13.16. That candidate was only read, never adopted.)

### Compatibility
- **meta-framework:**
  - No tracked file sits under any `generated/` path on any of the 29 remote branches (`git ls-tree -r`). I confirmed there are none at `origin/main`.
  - The only writers are synthetic tests, and all of them stay valid:
    - `tests/test_architecture_conformance.py:74` writes a nested `true`. It stays refused under the same category, so the existing substring assertion keeps passing.
    - On the duplicate-loader branch, `tests/test_architecture_json_duplicates.py` lines 56–57 write an exact-`false` fixture, and lines 150 and 165 write files with no marker. Clause 1 accepts both.
  - `contracts.yml` only runs unittest. It never runs the checker on a real workspace.
- **main** (`3e589ae0`, 233 refs grepped):
  - Nothing writes `meta-framework/generated` or calls `architecture_conformance.py`. I re-checked `origin/main`.
  - `architecture/graph_export.py:284` writes `"scientific_status_authority": False` into main's own `generated/graph.json`. That is a different directory, and the value is exact `false` anyway.
- **Math-, query-, google-drive, governance-, trial, sandbox, d6g8k5htny-coder and Universal-Law-Workspace** (all fetched refs):
  - None of them writes to `generated/` or references the checker. Every marker literal is `false`/`False`.
  - The `SOURCE_MANIFEST.json` that `query-` writes (`scripts/build_source_release.py:134`, value `False`) is read through checker lines 74–78, not through the generated route.
- **Result:** no producer I could observe would newly fail.
- **Limits of that result:**
  - GitHub code search returned 403, so the sweep used local clones of the fetched refs.
  - Externally prepared workspaces, forks, deleted PR heads and Drive or Dropbox payloads cannot be observed from here.
  - The proposal's own warning stands: prepared workspaces that use null, numbers, strings or containers as markers may newly fail.

### Amendments (replacement wording)
**A1, clause 1. Replace the clause with:**
> 1. Scope is `meta-framework/generated/**/*.json` as found by the existing glob (unchanged: case-sensitive `.json`, filesystem order). The rule applies to the value the checker's loader returns. In any JSON object, a member whose decoded name is exactly `scientific_status_authority` (JSON escapes decoded; no case-folding, whitespace stripping or Unicode normalization) may be omitted; when present, JSON Boolean `false` is the only accepted value. Ordinary unrelated keys are unchanged. A root that is not an object or array has no marker, and current root-shape behaviour is unchanged. A file the loader refuses yields only the existing parse-error violation and no marker category. Duplicate members are not handled by this contract: with today's loader the last duplicate wins, so `{"scientific_status_authority":true,"scientific_status_authority":false}` decodes to `false` and is accepted. This contract claims no refusal of a textually present `true` that the loader discards; that gap belongs to duplicate-loader custody 6049684431.

**A2, clause 2. Replace the clause with:**
> 2. Exact Boolean `true` keeps the existing refusal with a byte-identical message, at most once per file: `'generated scientific-status authority: ' + str(path.relative_to(root))`. Any non-Boolean value (numbers including zero and `-0.0`; the non-standard `NaN`, `Infinity` and `-Infinity` the loader accepts; strings including `""`, `"false"`, `"true"` and `"0"`; `null`; arrays; objects) gets a separate malformed-type violation, at most once per file, whose fixed text does not contain the substring `generated scientific-status authority` (for example `'generated scientific-status marker type invalid: ' + str(path.relative_to(root))`; the owner fixes the text before writer pickup). Neither message says or implies that authority was granted or a result accepted. Classification uses identity or exact type (`v is True`, `v is False`, or `type(v) is bool`, tested before any number check), never equality, membership, hashing or truthiness: `0 == False`, `1 == True` and `isinstance(True, int)` must not merge these cases.

**A3, clause 3. Replace the first two sentences with:**
> 3. Visit every JSON object reachable from the decoded root through object member values (including the value of a `scientific_status_authority` member) and array elements, without stopping at the first finding. If the file contains at least one exact-`true` marker, the true-category violation MUST appear; if it contains at least one non-Boolean marker, the malformed-type violation MUST appear; when both apply, both MUST appear, true-category first, whatever the nesting depth or member order. The walk MUST NOT raise for any value the loader returns: use an iterative or otherwise depth-independent walk, so every file the loader accepts, at any nesting depth, still yields the structured report.

**A4, clause 3. Replace the last sentence with:**
> Marker violations carry only the fixed category text and the existing workspace-relative path (`meta-framework/generated/...`): no value, JSON type name, member name, JSON pointer or count. This applies to the two marker categories; the existing parse-error text (line 87) is outside this contract.

**A5, clause 4. Replace the clause with:**
> 4. Exact-`true` and malformed-type violations fail closed through the existing structured report: `ok` false, the JSON report on stdout, empty stderr, CLI exit 2. This includes deeply nested generated files that today make the CLI exit 1 with a traceback and no report (through the CLI, from about 330 nested objects on Python 3.11.17 and about 500 on 3.13.16, up to the loader's own limit). Accepted markers alone do not establish workspace conformance. The registry check (line 60), the repository-role and source-manifest checks (`contract_validation.py` lines 83 and 89) and their messages stay byte-identical, and no shared helper changes them. Existing missing-map, source-cut, privacy and scientific-effect boundaries stay intact. Existing tests stay unchanged apart from additions.

If the owner keeps the depth fix out of scope, replace A3's last sentence and A5's second sentence with:
> Known limit, recorded and unchanged: generated files nested past the walk's recursion limit, up to the loader's own limit, make the CLI exit 1 with a traceback and no report. Today's helper does this from about 330 nested containers on Python 3.11.17 and about 500 on 3.13.16 through the CLI: 333/499 wrapper objects with `true` innermost, 332/498 with no marker or a `1` marker. The exact count depends on file shape and caller stack depth. The implementation records its own measured thresholds in place of today's.

Under that alternative, also drop A6's Depth bullet, the word "depth" from A6's predecessor-RED bullet, and A8's two recursive-walk items.

**A6, test matrix. Replace the "Negative type matrix …" bullet with:**
> - One generated file per case. Every non-Boolean value is placed twice: as a member of the root object, and in an object inside an array inside an object (`{"nested":[{"scientific_status_authority": v}]}`). Values: `0`, `1`, `2`, `0.0`, `-0.0`, `1.0`, `NaN`, `""`, `"false"`, `"true"`, `"0"`, `"x"`, `null`, `[]`, `[true]`, `{}`, `{"a":1}`.
> - Exact `true` as a member of the root object, in a nested object, in an object in a root array, and in an object in an array in an object.
> - Combined: an object-valued marker holding a `true` marker; a list-valued marker holding an object with a `true` marker; sibling objects malformed-then-true and true-then-malformed. Each expects both categories, true first. Also two exact-`true` markers in one file (`{"a":{"scientific_status_authority":true},"b":[{"scientific_status_authority":true}]}`) and two malformed markers in one file (`{"a":{"scientific_status_authority":1},"b":{"scientific_status_authority":null}}`), each expecting exactly one violation of that category.
> - Depth: about 900 nested objects with `true` innermost, and separately with a malformed marker innermost (both raise today on Python 3.11.17 and 3.13.16 and are within both loaders' limits); each expects the structured violation and exit 2. On Python ≥ 3.12 only (skip on 3.11, where the loader refuses this as a parse error), also 5000 nested objects with `true` innermost and with a malformed marker innermost; each expects the structured violation and exit 2.
> - Positives: absent marker; `false` at root and nested; ordinary nested JSON with no marker; root scalar `true`; a key differing only by case (`Scientific_Status_Authority`) or by a trailing space, each with value `true` and with value `1`, expecting `ok: true`.
> - Predecessor RED on `adf2945b` (or whatever blob is the base at custody): the malformed, combined and depth cases fail, except the two-`true` case. The positive, plain-`true` and two-`true` cases pass with today's exact reports. Do not relabel 6067866063.

**A7, assertions and modes. Replace "Structured API and actual CLI assertions … normal/explicit -O project checks" with:**
> - Each case asserts the exact API report dict, the exact CLI stdout (`json.dumps(expected, indent=2, sort_keys=True) + '\n'`), empty stderr, the exit code, and unchanged input bytes. Checking only `ok`, a substring, or the exit code is not enough (argparse usage errors also exit 2, with empty stdout).
> - The focused test itself runs the CLI child both without and with `-O`, whatever the parent's flags, and removes `PYTHONOPTIMIZE` from the normal child's environment, because `contracts.yml` runs the suite only in normal mode on Python 3.11 (lines 37 and 42) and `_registry_cli` adds `-O` only when the parent is optimized (tests line 90). The focused tests also pass under `python -B -O -m unittest`.

This keeps the proposal's two-file scope; it needs no CI change. CI's 3.11 job skips the ≥ 3.12 depth case, and A8's record covers that case.

**A8, faulty predicates. Replace "A deliberately permissive/faulty predicate must fail the focused test matrix" with:**
> - The implementation's review records, on a throwaway uncommitted copy, that each of these fails at least one focused test:
>   - the predecessor `is True` predicate;
>   - `v == False` accepted;
>   - truthiness acceptance;
>   - truthy values routed to the true category;
>   - tuple- or dict-membership dispatch;
>   - a type check on the root object only;
>   - no descent into the marker's own value;
>   - stopping at the first finding;
>   - the string `"false"` accepted;
>   - one violation per marker occurrence;
>   - a case-folding or whitespace-stripping key match;
>   - a recursive walk with two or more Python frames per level, such as today's `any()`/generator form (fails the 900 case);
>   - a recursive walk with one frame per level (passes 900 on 3.11–3.13 and fails only the ≥ 3.12 5000 case, so this record is made on Python ≥ 3.12).

**A9, sequencing with custody. Add to the scope:**
> - Before any writer pickup, the owner records which lands first: this contract or the duplicate-loader candidate in custody 6049684431 (branch `bf373685`, pinned to predecessor `aa3ac603`, which is now stale against `adf2945b`). If the duplicate-loader candidate lands first, this contract's implementation redoes its predecessor RED and positive runs against the resulting blob; any refresh of 6049684431 if it lands second is for that custody and the owner to record. When both apply, a file the duplicate-aware loader refuses yields only the parse-error violation (A1). A2's unchanged true message and one-per-file rule keep that branch's pinned expectation (`tests/test_architecture_json_duplicates.py` lines 25–27 and 135–140) valid.

The two changes touch different lines: the duplicate change replaces line 12, and this one would edit lines 17–22 and 88.

I checked the merge in throwaway clones (adversarial analysis, re-run for this review). `git merge-tree` of `bf373685` onto `main` is clean (tree `20a562ab`). The 21 combined architecture tests pass on 3.11.17 and 3.13.16, in normal and `-O`.

The coupling between them is semantic, not textual:
- A duplicate member, even an unrelated one, sends the file to the parse-error `continue` at line 87. Marker categories are then not reported for that file, although it still exits 2.
- Without a duplicate policy, the category depends on key order: `true` then `null` gives MAL, while `null` then `true` gives AUTH.

### What is NOT a problem
- **bool being an int subclass.**
  - `bool` cannot itself be subclassed, and `json` returns the `True` and `False` singletons.
  - For decoded values, `v is False`, `type(v) is bool and not v` and `isinstance(v, bool) and not v` are therefore equivalent.
  - Only equality, membership/hash and truthiness forms are wrong, and the proposal already forbids those.
- **`-O`.**
  - Neither the checker nor `contract_validation.py` contains `assert` or `__debug__`.
  - The existing 15 tests pass on 3.11.17 and 3.13.16, in normal and `-O`.
  - Every probe case gave the same result in both modes.
- **Exotic numeric types** (numpy, Decimal): unreachable. `check_workspace` takes a path and the loader is plain `json.loads`.
- **Non-string keys:** these cannot come out of JSON.
- **An escaped key** that decodes to the marker name is already refused.
- **Accepting absence while the registry requires the marker:** a deliberate, documented difference.
- **Treating `null` as malformed rather than absent:** consistent with "present means exact false", and no tracked producer uses `null`.
- **The "nested `true` under a malformed container" concern:** it is not a current bug, because line 20 already descends into the marker's own value.
- **The proposal's base `0b46ca16`** differs from `b4576cb9` only in the README.

### Recorded only (outside this contract; no amendment)
- The glob is case-sensitive: `X.JSON` and `.jsonl` files are not scanned. Hidden files (`.x.json`, `.hid/x.json`) are scanned.
- It follows a symlinked file even when the target is outside the workspace. It does not descend into symlinked subdirectories.
- A regular file named `generated` is skipped without a message (line 84).
- Parse-error messages copy exception text: an absolute path for an `OSError` (e.g. a directory named `dir.json`), and the offending byte for a decode error.
- The registry check at line 60 has the same last-wins duplicate exposure. It falls within 6049684431's scope, which covers all five loader routes.
- Violations across files come out in unsorted glob order.
- No contract documentation for `meta-framework/generated/` exists. README.md:149 covers only the registry field, and docs/ARCHITECTURE_CONFORMANCE.md lines 15 and 35 do not mention generated files. Adding such documentation is a separate scoping question.

### Choices left to the owner
1. Whether the depth fix belongs in this change (recommended, since the helper is being rewritten anyway) or is recorded as a known limit (the alternative under A5).
2. The exact malformed-type message text (A2).
3. Which change lands first, this one or 6049684431 (A9).
4. Whether `NaN`/`Infinity` should instead be refused when parsing. That is a loader decision that neither candidate makes. Under this contract they are malformed.

### Overall
**ACCEPT_WITH_AMENDMENTS, as a recommendation only.**
- **As written:** the contract should not be adopted. Clauses 3 and 4 admit implementations that drop findings or crash without a report, and clause 1 promises more than a check after parsing can see.
- **With A1–A9:** the contract can be adopted without another proposal round. If any amendment is declined or reworded, that change needs its own review.
- **What this review does not do:** it does not adopt the contract, assign implementation, start a writer pickup, or change any register.

### Own RELEASE
Posting this comment completes and releases my review pickup [6124077580](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6124077580).

State after this comment:
- **Contract:** UNADOPTED.
- **Implementation:** UNASSIGNED.
- **Nothing else changed:** no branch, ref, file, check, review vote, register or scientific record.
- **Issue 15:** stays OPEN.
- **Duplicate-loader custody** [6049684431](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6049684431): untouched and still unresolved.
- **d6g8k5htny-coder/main#365:** separate.

---
_Generated by [Claude Code](https://claude.ai/code)_