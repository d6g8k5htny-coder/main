# C8 composition read of main PR 451 at `057b8325`

Evidence for `C8-M451-COMPOSITION-READ-20261020` ([pickup 451/6280786151](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6280786151)). Dylan requested the read in C8's session at about 00:55Z. It is the bounded read by R1's reviewer at head `057b832543f2d67667da028104642c5823fd5e50` on base `8f7c2a498709ed8e8b89699560a1d9ab5a789903`. Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Read-only. Credit 0. Scientific effect NONE.

Byte identity, review applicability and hosted checks are reported separately. The applicability judgment is checked against `../applicability_20261020/PRE_ASSESSMENT_451.md` and `../applicability_20261020/preview_451_onto_8f7c2a49.txt`.

| File | What it is |
|---|---|
| `delta_readback_057b8325.txt` | `delta_readback.sh` on the real head, with `DEPS` set. |
| `local_checks.txt` | The landing check and the navigation check under `-B` and `-B -O`, R1's classifier on the record and packet at this head, and `git diff --check`, with every exit code captured directly. |
| `classify_log_057b8325.txt` | The full classifier output at this head. Apart from path prefixes, it is identical to `../m451_r1_review_20261019/classify_log.txt`. |
| `composition_facts.txt` | `refs/pull/451/merge`, the two refresh commits, the trusted intake run's output and the classifier comparison. |
