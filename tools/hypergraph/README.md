# tools/hypergraph — repo-agnostic math artifact hypergraph (organization only, NOT A RESULT)

credit 0 · OBL OPEN · no flags. Indexes existing records across repos; makes no correctness claim.

Built on Hypergraph-DB (`hypergraph-db==0.4.0`). Each coverage-ledger row is one `binding` hyperedge; each
artifact directory in a scanned repo is one `artifact` hyperedge. Every vertex is keyed `<repo>:<type>:<id>`
(`main:`, `Math-:`, `query-:`), so a unit named in main's ledger and the same directory scanned in Math- share one vertex.

## Inputs config — `inputs.json`
Lists each repo: `checkout` (path relative to this repo's root), and either main's `ledger`/`graph`/`report`
or an `artifacts` map of role -> fnmatch globs (`source`, `review`, `formal`, `impl`, `spec`, `test`) with `unit_depth`.
`default_repos` is what `build`/`--check` use; `--repos all` adds every repo whose checkout is present.
Absent checkouts (private or not cloned) are skipped with `skip <repo>: ...` on stderr.

## Run locally (Python 3.10+; CI uses 3.12, tested on 3.11 and 3.12)
```
python -m venv .venv && . .venv/bin/activate
pip install -r tools/hypergraph/requirements.txt          # all pins exact
pytest tools/hypergraph
python -m tools.hypergraph.math_hypergraph build           # rewrite data/hypergraph/*.json (default repos)
python -m tools.hypergraph.math_hypergraph build --check   # exit 1 if committed outputs are stale
python -m tools.hypergraph.math_hypergraph validate | gaps | repos | shared-impl 5
python -m tools.hypergraph.math_hypergraph reach main:node:math.d1-component.congruence-erratum 2
# other repos (optional, read-only):
git clone --depth 1 https://github.com/d6g8k5htny-coder/Math- external/Math-
git clone --depth 1 https://github.com/d6g8k5htny-coder/query- external/query-
python -m tools.hypergraph.math_hypergraph export /tmp/hg-bundle --repos all --png   # portable bundle
```
`--png` needs Pillow (pinned); without it the PNG is skipped with a message. All paths resolve from the repo root
(`HG_ROOT`, else `git rev-parse --show-toplevel`, else this file's location); nothing machine-specific.

## GitHub Actions — `.github/workflows/math-hypergraph.yml`
Runs on pull requests and pushes to `main` that touch the hypergraph tool, its data, GRAPH.json or the ledger, and on
manual `workflow_dispatch` (Actions tab → "Math hypergraph" → Run workflow). Steps: checkout (plus read-only
checkouts of public Math- and query- into `external/`, skipped if unavailable), Python 3.12 with pip cache,
pinned install, `pytest`, `build --check`, `validate`, `gaps`, then an all-repo `export --png` uploaded as the
`math-hypergraph` artifact (JSON, PNG, README, MANIFEST). `contents: read`, no secrets, 10-minute timeout, not a required check.
If `build --check` fails, run `build` locally and commit the regenerated `data/hypergraph/*.json`.

## Export bundle
`export DIR` writes `math_hypergraph.json`, `query_results.json`, `README.md` (format + loader snippet) and
`MANIFEST.json` (sha256 per file) — self-contained, movable anywhere.
