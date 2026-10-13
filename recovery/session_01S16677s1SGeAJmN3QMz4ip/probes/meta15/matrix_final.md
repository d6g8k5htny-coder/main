**Current behaviour vs proposed behaviour: generated-JSON `scientific_status_authority` marker**

**How "current" was produced.** This is a read-only run on meta-framework `main` `b4576cb9` (checker `adf2945b`, tests `8548a2f0`), in scratch copies made with `git archive`. It shows current behaviour. It is not a re-run or relabelling of diagnostic 6067866063, although for 1, 1.0, `"true"` and null it agrees with what that diagnostic reported.

- **Empirical analysis:**
  - 159 synthetic workspaces, built from the `setUp` fixture (tests lines 10–15) plus one generated file.
  - Each case ran through `check_workspace` in-process, and again under `-O`.
  - Each case also ran through the real CLI four ways: `python3 tools/architecture_conformance.py --workspace <ws>`, `python3 -O …`, `python3 -B -S …` and `python3 -B -O -S …`.
  - About 795 processes, all on Python 3.13.16.
- **Independent re-run for this review:**
  - Cases: 17 marker values × 4 placements and 6 extra cases through the CLI with `-B -S` and `-B -S -O`, plus 13 depths with `-B -S`.
  - Interpreters: Python 3.13.16 and 3.11.17. About 320 processes, plus the bisection.
- **Further checks for this revision:**
  - Depth, on 3.11.17 and 3.13.16:
    - the depth bisection for each innermost shape;
    - the depth boundaries and member order, in normal and `-O`;
    - in-process runs near the 3.11 loader limit.
  - The added A6 cases (two markers in one file; case- or space-variant keys with `true` and `1`), run through the API and the CLI in normal and `-O`.
  - The 5000-deep row, on 3.11.17, 3.12.3 and 3.13.16.
- **What agreed:**
  - Normal and `-O` were identical in every case.
  - The value and placement results were identical on 3.11.17 and 3.13.16.
  - Input file hashes were unchanged in every case.
  - In all 146 empirical cases that did not crash, CLI stdout byte-equals `json.dumps(api_report, indent=2, sort_keys=True)+"\n"`.

**Key:**
- AUTH = `generated scientific-status authority: meta-framework/generated/x.json`
- MAL = the proposed separate malformed-type violation (its text is still to be fixed)
- PARSE = the existing `generated JSON parse error: …`
- none = `ok: true` with no violations
- "Current" shows the CLI exit code as normal/`-O`.

**Value placements.** D = `{"k":"v","scientific_status_authority":V}`, or `{"k":"v"}` when the marker is absent. Each value was placed five ways:
- root: `D`
- nested dict: `{"a":{"b":D}}`
- root list: `[D]`
- list in dict: `{"items":[D]}`
- deep mixed: `{"a":[{"b":[[D]]}]}`

Current results were identical in all five placements.

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
| `NaN` (every placement); `Infinity`, `-Infinity` (root) | none / 0/0 | MAL / 2 |
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
| Root `true`, `1`, `null`, `""`, `[]`, `[true]` | none / 0/0 | none / 0 | none / 0 |
| Key written with a JSON escape that decodes to the exact name, value `true` | AUTH / 2/2 | AUTH / 2 | AUTH / 2 |
| `"Scientific_Status_Authority"`, or the name with a trailing space, value `true` or `1` | none / 0/0 | not specified | none / 0 |
| Duplicate keys, `true` then `false` | none / 0/0 (last wins) | none / 0 | none / 0, a stated known gap. Under the 6049684431 loader: PARSE only / 2 |
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

ssa = `scientific_status_authority`.

**Depth thresholds** (CLI with `-B -S`; shape `{"a":` × n wrapping `{"scientific_status_authority":true}`):

| Interpreter | n | Result |
|---|---|---|
| 3.13.16 | ≤ 498 | AUTH, exit 2 |
| 3.13.16 | 499 to 9997 | exit 1: last stderr line starts `RecursionError: maximum recursion depth exceeded`, stdout empty |
| 3.13.16 | ≥ 9998 | PARSE, exit 2 |
| 3.11.17 (local 3.11; CI pins `3.11` at `contracts.yml` line 37) | ≤ 332 | AUTH, exit 2 |
| 3.11.17 | 333 to 991 | exit 1, as above (3.11 appends ` while calling a Python object`) |
| 3.11.17 | ≥ 992 | PARSE, exit 2 |

Other innermost shapes crash one or two wrappers earlier:
- With `{"ssa":1}`, `{"k":"v"}` or `{}` innermost, the first exit 1 is at 332 (3.11.17) and 498 (3.13.16).
- With `{"ssa":{}}` innermost it is 331/497.
- In-process callers with a deeper stack crash sooner still.

The duplicate-loader column comes from the compat analysis's probe of the `ef92a614` CLI, normal and `-O`, Python 3.13.16. That candidate was only read, never adopted.