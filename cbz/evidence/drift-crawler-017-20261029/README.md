# Drift crawler: evidence and proposed registration

Built by Anthropic Claude, session Claude017 (`session_017Mi3hx…`), under the owner's CBZ-0
addendum ("Intent-locked reinforcements": a dynamic federation drift crawler).
- Pickup: main#684/6510531192.
- Credit 0. Scientific effect NONE.
- Under B-003, this directory and `tools/cbz_drift/` are the only things written here.
  `GOVERNANCE.md`, `TOOLING.md` and the gate matrix belong to Root. The proposed text
  for them is below.

## Observed runs (2026-10-29, CPython 3.11.15)

The tool bytes and every command are in this directory. Exit status is in `exits.txt`.

| Run | Seed | Crawled | BLOCK | INFO | Lean files / declarations / names shared across locations | Registry artifacts verified |
|---|---|---|---|---|---|---|
| `crawl-pins.json`, local clones | ULW `a4b01682` | 9 repositories | **0** | `PIN-STALE` 4, `REGISTRY-EVOLVED` 31 | 166 / 3,030 / 479 | 592 of 592 |
| `crawl-tips.json`, default tips | ULW `a4b01682` | 9 | **0** | `PIN-STALE` 4, `REGISTRY-EVOLVED` 32 | 172 / 3,039 / 482 | 592 of 592 |
| `crawl-cold-pins.json`, only the seed cloned, the rest found and cloned by the crawler | ULW `a4b01682` | 9 | **0** | as for pins | as for pins | 592 of 592 |

- **Cold start.** The cold-start report has the same statistics, refs and findings as the
  local run at pins. The crawler needs nothing but the seed.
- **What the 0 means.** No two records that claim the same identity disagree. At the pins
  and at the tips, every Lean name declared in two or more places (479 at the pins, 482 at
  the tips) has one interface. Almost all of them are statement packages copied between Math- packages
  (`frontiers/source_*`, `companions/torus_filtration`, `frontiers/compact_morse_elder_adapter`,
  `specifications/lean_statements_20261010` and others). Three are within `main`'s
  `cbz/evidence` kernel replays.
- **Pins.** The four stale pins (`governance-`, `main`, `Math-`, `meta-framework`) are
  behind their tips and on the default line. This crawler reports staleness as INFO.
  Currency is what ULW's own `verify_pins.py --require-current` enforces.
- **Local fetch.** Several local clones were shallow. `--fetch` unshallowed them, which is
  additive only. Without it, the crawler reports BLOCK `PIN-OFFLINE … shallow clone` and
  does not guess ancestry.

## Tool verification

- Two interpreter modes (`python3 -B` and `-B -O`): 43 tests OK, about 42 s each.
- **35 mutants, all killed**, each by its designated test failing by assertion. Every
  designated test passes on the unmutated copy (the harness control).
- The 344 tests in `main`'s workflow-reading modules give the same 3 errors with and
  without this change: `test_formal_workflow_shell` and two `test_periodic_pilot_custody*`
  load errors. All three are environmental in this container.

## Proposed text for Root (B-003)

**GOVERNANCE.md, Tool registry.** Add this entry:

> `tools/cbz_drift/crawl.py`: federation drift crawler (G7). Discovers the federation from
> the workspace seed's records and fails on disagreement between records of one
> identity: pins, registry bytes, and the source interfaces of Lean declarations shared
> across locations. Lexical domain and limits: `tools/cbz_drift/README.md`. It certifies
> agreement of recorded identities and source text only; G2–G4 remain authoritative for
> kernel closure.

**TOOLING.md.** Add this row:

| Tool | Gate | Command | Verification |
|---|---|---|---|
| `tools/cbz_drift/crawl.py` | G7 | `crawl.py --seed d6g8k5htny-coder/Universal-Law-Workspace --search fed --clone-into fed --fetch --at pins` (and `--at tips`) | `tests/test_cbz_drift.py`, both modes; 35 designated-assertion mutants (G6) |

**Gate matrix §10, G7.** Proposed wording: "PASS when `cbz-federation-drift` exits 0 at
`--at pins` in the final run, with every discovered repository resolved. A BLOCK finding
fails G7 and blocks integration of the PR that introduces it." The workflow already runs
on every PR to `main` that touches Lean, lake files or the tool. It overrides `main`
with the candidate head and crawls the rest of the federation at its pins and tips.
