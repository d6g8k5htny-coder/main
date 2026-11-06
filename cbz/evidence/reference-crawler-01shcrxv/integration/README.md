# Integration kit for Root (B-003)

Everything here is a proposal. Nothing in this directory is active until Root moves it into place.

| File | What it is | How to apply it |
|---|---|---|
| `root-695.patch` | Adds the tool's `GOVERNANCE.md` tool-registry entry and its `TOOLING.md` row, in #695's own format | `git apply root-695.patch` at #695 head `82d9e480`; checked with `git apply --check` |
| `cbz-federation-references.yml` | The G7 merge gate. It runs on every pull request, on each push to `main`, nightly and on demand. It crawls the federation with the tested commit as main's candidate (`--repo main=main --at main=HEAD`), in strict mode with the reviewed acknowledgements (`--strict --acknowledged tools/cbz/reference_crawler_acknowledged.json`). Every other repository is cloned anonymously from its `pins.json` URL; all are public. | Copy into `.github/workflows/`. To make it a required context, add `federation-references` to ruleset `23798639`. |
| `cbz-reference-crawler-self-test.yml` | The tool's own G0/G6 evidence, hosted: the tests in both interpreter modes, then the mutation check. Path-filtered to the tool, its tests and the workflow, because the mutation run takes about 35 minutes. | Copy into `.github/workflows/`. |

**Why the gate runs strict.** The owner's addendum requires G7 to block merge synchronization
when it detects drift.
- **Without `--strict`,** interface drift is INFO.
- **With `--strict` alone,** every historical citation whose cited file later changed or was
  removed would block forever. A dated report or an archived snapshot cites a file at a commit
  on purpose.
- **The acknowledgement file** lists those citing files, each with a reason that can be checked
  against the file. The gate then blocks on every other drift.
- **Changes are visible.** A pull request that adds an acknowledgement shows it in its diff, with
  its reason.

The evidence README's section "Interface drift under `--strict`" gives the counts and every run.

**What a green gate means.** Once Math-#548 lands, this gate exits 0 on the current federation.
The readback at #548's current head `542cadbc`, and at its earlier head `44a583c4`, gives 0
blocking at the pins and at the tips, in strict mode with the acknowledgements. From then on it
fails any pull request that does any of these:
- records a broken reference;
- breaks a live reference into main;
- changes or removes an interface that an unacknowledged record cites.

**Simulation on this pull request.** The gate's own command is run on each pushed head of #721,
with this repository's checkout as main's mirror. Each result is posted on #721 for its exact head.
