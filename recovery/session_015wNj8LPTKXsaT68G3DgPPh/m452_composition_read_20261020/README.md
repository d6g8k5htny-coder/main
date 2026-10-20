# C8 composition read of main PR 452 at `cc8b29e2`

Evidence for `C8-M452-COMPOSITION-READ-20261020` ([pickup 452/6280546616](https://github.com/d6g8k5htny-coder/main/pull/452#issuecomment-6280546616)). This is the bounded read by the reviewer of record at head `cc8b29e2d6b11efa8a835cf40c9c0da751fd52d5` on base `b78134c75e733febc6bd2becac4459036ef8c0ad`, after the refresh. Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Read-only. Credit 0. Scientific effect NONE.

Byte identity, review applicability and hosted checks are reported separately. The applicability judgment is checked against the pre-assessment in `../applicability_20261020/PRE_ASSESSMENT_452.md`.

| File | What it is |
|---|---|
| `delta_readback_cc8b29e2.txt` | `delta_readback.sh` on the real refresh, with `DEPS` set. It reports the parents, the new tree against a fresh merge-tree and the trial tree, the raw-diff identity, the base delta, the dependency objects and the diffstat, plus the check runs at the time it ran. |
| `local_checks.txt` | The landing check and its 15 tests under `-B` and `-B -O`, and `git diff --check`, at the real head. Every exit code is captured directly. |
| `composition_facts.txt` | `refs/pull/452/merge`, the composition with PR 421's head, PR 421's last comment, and the trusted intake run's checkout and output. |
| `hosted_checks.txt` | All eight check runs at the head, read at 00:46Z, each with its job, run, attempt, event and workflow. It also gives `verify`'s aggregate, which binds `checks` and `formal` to `checked_commit` `600a2a41`, the tested merge. |
