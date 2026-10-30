# Integration kit for Root (B-003)

Everything here is a proposal. Nothing in this directory is active until Root moves it into place.
Root also decides whether G7 runs with `--strict`.

| File | What it is | How to apply it |
|---|---|---|
| `root-695.patch` | Adds the tool's `GOVERNANCE.md` tool-registry entry and its `TOOLING.md` row, in #695's own format | `git apply root-695.patch` at #695 head `82d9e480`; checked with `git apply --check` |
| `cbz-federation-references.yml` | The G7 merge gate. On every pull request, push to `main`, nightly and on demand, it crawls the federation with the tested commit as main's candidate (`--repo main=main --at main=HEAD`). Every other repository is cloned anonymously from its `pins.json` URL; all are public. | Copy into `.github/workflows/`. To make it a required context, add `federation-references` to ruleset `23798639`. |
| `cbz-reference-crawler-self-test.yml` | The tool's own G0/G6 evidence, hosted: the tests in both interpreter modes, then the mutation check. Path-filtered to the tool, its tests and the workflow, because the mutation run takes about 25 minutes. | Copy into `.github/workflows/`. |

**The merge gate was simulated before proposing it.** It was run on #721's own head
`f91bfb2e`, using this repository's working checkout as main's mirror:
- result: 28,639 references and 1 blocking finding, the known Math- `rn_status` record;
- this PR's own evidence files add only two INFO `FOREIGN_REPOSITORY` rows: the mathlib reference
  echoed in the summaries.

**What a green gate means.** Once Math-#548 lands, this gate exits 0 on the current federation:
the readback at Math- `44a583c4` gives 0 blocking at the pins and at the tips. It then fails any
pull request that records a broken reference or breaks a live reference into main.
