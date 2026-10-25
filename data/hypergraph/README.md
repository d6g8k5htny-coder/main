# Math artifact hypergraph — organization only, NOT A RESULT

credit 0 · OBL OPEN · no flags. This indexes existing records; it makes no correctness claim.

Built with Hypergraph-DB 0.4.0 by `tools/hypergraph/math_hypergraph.py`. Each coverage-ledger row is
**one hyperedge** whose `roles` map labels members as unit / source / review / impl / node / lane / record.

## Inputs (all in-repo)
- `math_coverage_ledger_20261023.md` — RETRO-MATH-BIND-02 ledger with the 24–25 October Main membership corrections; the original 211/133 and subsequent 213/131 and 214/130 snapshots are retained in its history.
- `../../docs/site/dependency-source/GRAPH.json` — 49 graph nodes (#527).
- `graph_binding_report_20261023.json` — #527 binding report snapshot (21 bound nodes).

## Outputs
`math_hypergraph.json` (vertices + hyperedges), `query_results.json` (gaps, reach example, shared impls),
`bindings_hypergraph.png` (CI artifact only: the hyperedges that touch a graph node).

## Rebuild / query
Locally (from the repo root, Python 3.11/3.12):
```
pip install -r tools/hypergraph/requirements.txt
python -m tools.hypergraph.math_hypergraph build            # regenerate these JSON files (main only)
python -m tools.hypergraph.math_hypergraph build --check    # CI gate: exit 1 if stale
python -m tools.hypergraph.math_hypergraph reach main:node:math.d1-component.congruence-erratum 2
python -m tools.hypergraph.math_hypergraph gaps
python -m tools.hypergraph.math_hypergraph validate
python -m tools.hypergraph.math_hypergraph shared-impl 5
python -m tools.hypergraph.math_hypergraph export /tmp/hg-bundle --repos all --png
pytest tools/hypergraph
```
On GitHub: `.github/workflows/math-hypergraph.yml` runs the same steps on PRs/pushes touching these inputs or via
workflow_dispatch, with read-only checkouts of public Math- and query-, and uploads the all-repo bundle + PNG as the
`math-hypergraph` artifact. Vertex keys are repo-prefixed (`main:`, `Math-:`, `query-:`). See `tools/hypergraph/README.md`.
The PNG is a build artifact (optional, `--png`), not committed.

## Gap counts after the zeta membership correction (2026-10-25)
344 ledger rows / 1296 vertices / 344 hyperedges. Sourced containers 215 (129 missing). Implemented 296 (48 missing),
of which 30 are placeholder-only ("in PR head" / "per census", no artifact). Bound units 14.
28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit. Role problems 0.
Rows 307 and 310 now include the 56 tracked Markdown members under the existing
periodic-H0 and universality containers at Main
`32366ac0d4e2d1464eaac51866b9dce2128970b9`. These are proofs, reviews, notes,
protocols and historical reports, not 56 new theorem units or verified
proof-to-code relationships. That supplement had 1290 vertices and 213 sourced /
131 missing containers. The original 2026-10-23 graph had 1234 vertices and
211 sourced / 133 missing containers.

The later row-312 correction consumes the existing `formal/manifest.json`
source-to-target map at Main `9c03589b1a3f0c1fd6befd09abe6b95c99cff7d7`.
It adds the two pinned SIDE24 source copies and the omitted `Ledger.lean`
implementation beside the existing ConeMoments/Endpoints modules. The map
declares 27 proof-source arithmetic targets plus four published-number
endpoint targets; these do not establish the coefficient enclosure or parent
theorem. Statement alignment remains PENDING_INDEPENDENT_REVIEW; no Lean or
formal gate ran. Row 313 is import-only and retains MISSING source. That
correction produced 1293 vertices and 214/130 sourced/missing containers while
preserving all row identities and the other 343 raw rows.

Row 311's later zeta correction consumes committed README/CLAIMS and the
existing MANIFEST at Main `6c6c6b0ff649d5ac0413344c00b6cdbe95344f82`.
README's Proposition 1 is the primary proved-note carrier; CLAIMS is an audit.
The explicitly linked `test_exact.py` is a bounded exact algebraic-control
consumer beside the two existing numerical collision programs. This is not
verification of the full analytic proposition, every 46-script package claim,
a limiting exponent or RH progress. It reuses [Main548's](https://github.com/d6g8k5htny-coder/main/pull/548)
historical GB-8n proposal/pins and ENGINEERING_CONTROL/unbound meaning; no new
node or theorem unit is introduced. All original vertices and memberships,
343 other raw rows and all node/bound/status cells remain unchanged. Only row
311's source/implementation membership changes; no zeta program was executed.

This is a bounded membership correction, not complete Main/history/cloud
coverage. The exact-sample shrinking-bin comment sources have separately
recorded [archive custody in Main591](https://github.com/d6g8k5htny-coder/main/pull/591#issuecomment-6398496409);
this index change supplies no mathematical or execution qualification for
them. Separate storage crosswalks retain their own coverage/privacy limits;
this supplement ingests no Dropbox-only source bytes. No scientific acceptance
or original review outcome changes.
