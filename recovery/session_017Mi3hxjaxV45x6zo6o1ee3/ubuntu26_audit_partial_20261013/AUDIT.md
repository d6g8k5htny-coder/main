# Ubuntu 26 runner audit, 2026-10-13: PARTIAL

**Status: PARTIAL.** This audit is checkpointed and stopped on the coordinator's
instruction (main #307, comment 6151543585). It is not complete. The eight Lean
workflows listed below were **not tested**.

Scientific effect: NONE. Organizational-independence credit: 0. Performer:
Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Information only:
no workflow was changed, dispatched or rerun. Advisory: main #307, comment 6151438160.

## Trigger

GitHub's runner notice on check runs in this project (for example Math- check
run 114537992336): "The ubuntu-latest label will migrate to Ubuntu 26 beginning
October 19, 2026."

## Cut

Workflow files were read from each repository's default branch at about
17:30Z on 13 October. The default-branch heads at the time of this checkpoint
(about 17:43Z) were:

| Repository | Head |
|---|---|
| main | `f036dab9abdf782da9edbcccfd69dafbf47e867c` |
| Math- | `b68d88b3993a5e2f2d590635a555ed963b97fbf0` |
| trial | `3738c859d519d2dbf1e388fe48fe0ae60c9487a3` |
| meta-framework | `1549bec4ac65bfe886bfb5ab502d7b8adaf0baaa` |
| Universal-Law-Workspace | `a152b9d1119272db914978dc58af46d7926151e5` |
| google-drive | `ad08584b3ea9d533af10113e85169f471fa8ec93` |
| governance- | `5f7ce82949438ba8996811e8a2e0d139f7b7532b` |
| query- | `961a7d3a1f04bddaf62e722cedf625a91f0a4347` |
| sandbox | `dce04a4174835107126ab4f36c199a2dc320f964` |
| d6g8k5htny-coder | `3ba2f7cf6fdf55177bf3f50978be8e852b508b25` |

## Results

185 workflows use `runs-on: ubuntu-latest`. The per-file list is in
`WORKFLOWS_UBUNTU_LATEST.tsv`.

| Class | Count | Result |
|---|---:|---|
| `actions/setup-python` | 175 | **Checked.** Every pinned version has an Ubuntu 26.04 x64 build in the `actions/python-versions` manifest at `49858b91`. That covers `3.11.16`, and `3.11` and `3.12`, which now resolve to `3.11.17` and `3.12.15` (`PYTHON_VERSIONS_CHECK.json`). |
| Runner `python3` without `setup-python` | 5 | **Partly checked.** None of the scripts imports a removed standard-library module. Two were run with Python 3.14.8 as stand-in: main `test_public_intake.py` (71 tests OK in `-B -S` and in `-B -O -S`) and google-drive `verify_replicas.py` (12 of 12; unit tests OK in normal and `-O` mode). The three trial workflows were checked statically only. |
| Lean or elan | 8 | **NOT TESTED.** |

### The eight untested Lean workflows

| Repository | Workflow |
|---|---|
| Math- | `.github/workflows/cap-first-exit-lean.yml` |
| Math- | `.github/workflows/d2-atom-integral.yml` |
| Math- | `.github/workflows/d2-moment-bridge.yml` |
| Math- | `.github/workflows/d2-square-schur.yml` |
| Math- | `.github/workflows/lean-specification-proofs.yml` |
| Math- | `.github/workflows/lean-specifications.yml` |
| Math- | `.github/workflows/torus-filtration.yml` |
| trial | `.github/workflows/ci.yml` |

## Not covered

- The eight Lean workflows above.
- Any other dependency on the operating system or image contents.
- Workflows on branches other than the default branch.
- Runs on the actual Ubuntu 26 image.

## Resume conditions

Only on the coordinator's or the owner's explicit word. The next step would be
to test the eight Lean workflows' toolchain setup on an Ubuntu 26.04 image, or
to watch their first runs after 19 October. Pinning `runs-on: ubuntu-24.04` is
each workflow owner's decision, and it changes reviewed workflow bytes.
