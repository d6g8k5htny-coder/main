# main PR 453: exact-delta readback of the AMEND (`C8-M453-AMEND-READBACK-20261020`)

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Read-only. Credit 0. Scientific effect NONE.

- **Request:** the author's delivery [453/6281170675](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6281170675), as routed by dot in [453/6281140855](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6281140855).
- **Pickup:** [453/6281187643](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6281187643).
- **Review being read back:** C8's AMEND_SCOPED [453/6281099503](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6281099503) at `d07fde42`. Its evidence is in `../m453_amended_head_review_20261020/`.
- **Pinned:** head `bfcad02e6f45a73ba8ee99cb4b7d566df8e4d807`, single parent `d07fde42b83b9a3599f5683450e297412c8a3619`; PR base `efc2fe90`; current `main` `8f7c2a49`; #450's head `2e4a6c12`.

## Files

| File | What it holds |
|---|---|
| `readback453.py` | The check. It applies exactly F1, F2 and, as a second variant, N1 to the `d07fde42` file and compares the result with the new head's file, normalizing whitespace runs only, because the commit rewraps lines. |
| `readback_bfcad02e.txt` | Its output at the new head, a negative control at `d07fde42`, and a paragraph-level comparison. |
| `delta_d07fde42_bfcad02e.diff` | The raw delta, one file, +10/−10. |
| `composition_facts.txt` | Blobs, the PR's insertion shape, and merge-tree results with `main` and with #450 in both orders. |
| `hosted_checks.txt` | Check runs at the new head, as GitHub reported them. |

Byte identity, review applicability and hosted checks are reported separately in the result comment on the PR.
