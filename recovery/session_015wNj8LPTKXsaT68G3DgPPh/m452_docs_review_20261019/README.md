# Evidence for C8's nonauthor docs review of main#452

This folder holds the checks behind `C8-M452-DOCS-REVIEW-20261019`, the nonauthor review of main#452 (ancestry commands and the carried-work caveat) at head `63cf4766`. The performer is Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. The pickup is [main#452/6279176521](https://github.com/d6g8k5htny-coder/main/pull/452#issuecomment-6279176521), and the verdict is posted in the same thread. The reviewer has the same provider as the author, so credit is 0. Scientific effect NONE.

| File | What it is |
|---|---|
| `landing_and_tests.txt` | The landing check and its 15 tests at the head, under `python3 -B` and `python3 -B -O` on Python 3.11.15, plus `git diff --check` and the longest added line. |
| `exp_AB.txt` | Experiments A and B, in fresh `--depth 1 --single-branch` clones of `main`: `--is-ancestor` before and after `git fetch --unshallow origin main`, and one `git fetch --deepen=500 origin main` step. |
| `exp_C.txt`, `exp_C2.txt` | Experiment C: a depth-1 clone of `main`, plus #452's branch and an old `main` commit, each fetched at depth 1. It shows the false negative (exit 1) while the clone is shallow, and that `--unshallow` then leaves the whole clone complete. |
| `compare_api.txt` | The compare-API command as written: three commits in `main`, one not in `main`, and four 404 cases. |
| `unrelated_and_330.txt` | Unrelated history for `claude/recovery-01N-20261013` in a complete clone, and the #330 `PROOF.md` blob comparison. |
| `hosted_runs.txt` | The workflow and check runs at the head, and the checkout lines from the `verify`, `public-shop` and both `public-intake` job logs. |
| `board_scan_421.txt` | Mentions of #421 on #419, #307, #275 and #421 itself, created after its 17 October disposition. |
| `composed_421_452_landing.txt` | The landing check and its tests on a local trial composition of #421's head with this head (tree `a93c6a8d`). |

The git experiments ran in disposable clones in C8's scratch directory. Nothing was pushed except this recovery folder.
