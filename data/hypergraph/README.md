# Math artifact hypergraph — organization only, NOT A RESULT

credit 0 · OBL OPEN · no flags. This indexes existing records; it makes no correctness claim.

Built with Hypergraph-DB 0.4.0 by `tools/hypergraph/math_hypergraph.py`. Each coverage-ledger row is
**one hyperedge** whose `roles` map labels members as unit / source / review / impl / node / lane / record.

## Inputs (all in-repo)
- `math_coverage_ledger_20261023.md` — RETRO-MATH-BIND-02 ledger with the 24 October Main source-membership supplement; the original 211/133 snapshot is retained in its history.
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

## Gap counts after the Main membership supplement (2026-10-24)
344 ledger rows / 1290 vertices / 344 hyperedges. Sourced containers 213 (131 missing). Implemented 296 (48 missing),
of which 30 are placeholder-only ("in PR head" / "per census", no artifact). Bound units 14.
28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit. Role problems 0.
Rows 307 and 310 now include the 56 tracked Markdown members under the existing
periodic-H0 and universality containers at Main
`32366ac0d4e2d1464eaac51866b9dce2128970b9`. These are proofs, reviews, notes,
protocols and historical reports, not 56 new theorem units or verified
proof-to-code relationships. The original 2026-10-23 graph had 1234 vertices and
211 sourced / 133 missing containers. Its 344 row identities and all
implementation/node/bound/status cells are preserved.

This is a bounded membership correction, not complete Main/history/cloud
coverage. The exact-sample shrinking-bin note's declared complete comment proof
remains a separate custody boundary. A project-wide Dropbox crawl is not
established; connector availability differs by actor. No scientific acceptance
or original review outcome changes.
