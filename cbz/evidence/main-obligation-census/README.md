# Census of main's proof obligations

CBZ-0 Phase 1 discovery for criterion 3, unit `CBZ0-MAIN-OBLIGATION-CENSUS-01SHCRXV` on issue #684. Read at `main`
`81acc060`. It gives draft rows for Root's `PROOF_INDEX.md` (governance B-003) and no dispositions.

| File | What it is |
|---|---|
| `CENSUS.md` | 20 prose and archival rows, plus the 31 Lean targets. Start here. |
| `census.json` | The same rows, machine-readable, with the computed links. |
| `build_census.py` | Holds the manual reading and renders both files. It checks every citation against Git objects at the revision. |
| `test_build_census.py` | Adverse tests: a wrong blob, an out-of-range or inverted line range, a misplaced anchor and a missing path must each fail. |
| `extract.py`, `digest.py` | The extraction aids used for the reading: headings, status lines, statement and non-claim lines, and review verdict lines. |

## What it establishes

- **Mechanically checked.** For every row, the path exists at `81acc060` with the recorded blob, each cited line range lies inside the file, and each cited anchor text occurs in its range. The links between rows are computed from each document's own relative links.
- **Manual and not mechanical.** Each claim restates the document's own words, and the statement and limit ranges locate where it says so. This is not a new mathematical review.
- **The draft dispositions follow governance.** Prose proofs are QUARANTINED, because prose review cannot establish kernel closure. The 31 Lean targets are CLOSED candidates, pending G3, G4 and G2 observed in the final run.

## Findings for the integrator

1. **`main` has no `PROOF_INDEX.md`.** These 21 rows would be its first content for `main`.
2. **Status disagreement (G8).** `formal/sources/side24_v1/PROOF.md` line 4 says "nonauthor review open". `STATUS.md`'s D3 row records that review on main #65.
3. **Doc drift (G8).** `STATUS.md` is dated 2026-09-29 and predates every row from OBL-MAIN-02 to OBL-MAIN-15.
4. **The link graph has a cycle.** OBL-MAIN-10 and OBL-MAIN-13 link each other, so links are references and are not a strict dependency order.
5. **OBL-MAIN-20 is an open HOLD.** The LB-RATE constant `c = 0.9144` must not be indexed as CLOSED, and its `history/` verifiers are float-based (01XJMJ's A3 census).

## Rerun

```sh
python3 -B -I build_census.py --git <clone> --out .   # exit 1 if any citation fails at REV
python3 -B -I test_build_census.py                    # 6 tests
```
