# Federation drift crawler

CBZ-0 addendum tool: the dynamic federation drift crawler. It uses the standard library
only. Its gate is G7 (pins consistent), extended to interface consistency.

```
python3 -B -S tools/cbz_drift/crawl.py --seed d6g8k5htny-coder/Universal-Law-Workspace \
  --search DIR [--search DIR2] [--clone-into DIR] [--fetch] [--at pins|tips] \
  [--override OWNER/NAME=REV] [--seed-ref REV] [--json REPORT]
```

Exit status: 0 no BLOCK finding, 1 at least one BLOCK, 2 usage or internal error.

## What it discovers

Only the seed is named on the command line. The crawler reads the seed's
`.gitmodules`, gitlinks, `pins.json` and `workspace/repositories.json`, then:

- each registry named under `canonical_inputs` (today the meta-framework
  `registry.json`), resolving artifact `repository` keys through the registry's own map;
- every Lean package (`lakefile.lean` / `lakefile.toml`) in every reached repository;
- every `lake-manifest.json` dependency on a repository of the federation's owners
  (those of the seed and of every repository the records list). Others, such as
  mathlib, are counted as external edges and are not crawled.

It does this as a breadth-first fixed point: every reached repository's own records,
canonical inputs and lake manifests are read in turn until nothing new appears, and
every reached repository is then validated and scanned once. A repository is read at
one ref: an override, else the first pin met in breadth-first order, else its default
tip. With `--fetch`, each repository is refreshed from origin first, so a default tip
is the remote's current one.

So a rename, addition or removal in those records changes the crawl with no code
change. `--search` directories are scanned for clones by their `origin` URL. With
`--clone-into`, a discovered repository that no search directory holds is cloned, so
a CI job needs only the seed. A repository it still cannot read is a BLOCK finding,
never a skip. The one exception is a repository its own records mark opt-in
(`update = none` / `opt_in`), which is INFO.

## What it checks

The rule: **two records that claim the same identity must agree (BLOCK); a record may
lag a moving tip (INFO).**

| Code | Level | Meaning |
|---|---|---|
| `PIN-SOURCES` | BLOCK | gitlink, `.gitmodules`, `pins.json` and `repositories.json` disagree, or one of them lacks a repository the others list |
| `PIN-MISSING` | BLOCK | a pinned or recorded commit is absent (`--fetch` consults origin) |
| `PIN-OFFLINE` | BLOCK | a pin is not an ancestor of its repository's default tip; on a shallow clone it says ancestry cannot be decided |
| `PIN-STALE` | INFO | a pin is behind the default tip |
| `UNRESOLVED` | BLOCK | a discovered repository cannot be read |
| `REGISTRY-BYTES` | BLOCK | a registry artifact's size or SHA-256 differs at its recorded commit, or the path is absent |
| `REGISTRY-EVOLVED` | INFO | the artifact differs, or is gone, at the crawled ref |
| `INPUT-OBSERVED` | INFO | a canonical input's recorded blob differs at the crawled ref |
| `LEAN-DRIFT` | BLOCK | one fully qualified Lean declaration has different interfaces in two crawled locations |
| `LEAN-SYNTAX` | BLOCK | a Lean file cannot be lexed, or its `namespace`/`section`/`end` do not balance |
| `LAKE-DEP` | INFO | a lake dependency on a crawled repository pins a present, different rev (an absent, empty or non-string rev is `PIN-MISSING`; a git package with no repository URL is `MALFORMED`) |
| `PIN-DIVERGENT` | INFO | two reached repositories pin one repository differently; the first in breadth-first order is read |
| `FETCH-FAILED` | BLOCK | `--fetch` could not refresh a repository, so its tips are not current |
| `TOOLCHAIN` | INFO | Lean packages that share declarations use different `lean-toolchain`s |
| `MALFORMED` | BLOCK | a manifest exists but cannot be parsed |

`--at pins` (the default) reads each pinned repository at its pin, which is the
federation snapshot. `--at tips` reads default tips. `--override` puts a candidate head
in place of one repository's ref: that is how a PR is tested against the rest of the
federation (see `.github/workflows/cbz-federation-drift.yml`).

## Domain and limits (B-006)

`lean_decls.py` extracts declarations written literally in source:
- `theorem`, `lemma`, `def`, `abbrev`, named `instance`, `axiom`, `opaque`, `structure`,
  `class` and `inductive`;
- with their modifiers, qualified by `namespace` / `section`.

The compared **interface** is:
- for a theorem, lemma, axiom or opaque: its statement (proofs are irrelevant, and
  opaque values are hidden);
- for anything definition-like: statement plus value, because the value is part of
  its meaning.

The interface also carries its **context**: the `open`, `variable`, `universe`, `export`,
`include` and `omit` commands in scope (file level, enclosing `namespace` and `section`
blocks, and any `open … in` or `variable … in` prefix of the declaration). So the same
statement text under `variable (x : Nat)` and under `variable (x : Int)`, or under
`open Real` and `open NNReal`, is drift. A variable the statement does not use still
counts, which errs toward reporting drift. Notation, macros and `set_option` are not
tracked. Inside a declaration, `open … in` and `set_option … in` stay in its value
unless they start a line that is not indented deeper than the declaration, and
`mutual … end` is a scope.

Comments are removed and whitespace is normalized. Equal interfaces mean equal source
text, not equal elaborated terms. The extractor does not elaborate, expand notation or
macros, or see generated declarations. `private` names are file-local and are never
compared. Two copies that differ only by an alpha-renamed binder are reported as drift;
that is the safe direction, since named arguments are part of the interface. Kernel
closure and axiom dependencies stay with G2–G4.

## Verification

`tests/test_cbz_drift.py` runs every check against a synthetic git federation built in
a temporary directory, covering both positive and negative cases. That includes:
- topology changes (added, removed and registry-only repositories, and lake edges);
- shallow clones;
- clone-on-demand;
- candidate-head overrides.

`ScannerMutationTests` applies 58 source mutants. Each one must be killed by its own
designated test, failing by assertion (`FAILED (failures=N)` with no errors). Every
mutant must apply exactly once and must compile, and the unmutated copy must pass
every designated test, as the harness control.

```
cd tests && python3 -B -m unittest test_cbz_drift && python3 -B -O -m unittest test_cbz_drift
```
