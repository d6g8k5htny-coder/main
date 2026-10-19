# C8's exact-head review of ULW#18 at `6e112b95`

This folder holds the evidence for `C8-ULW18-EXACT-HEAD-REVIEW-20261019`. That review is required by Dylan's decision [ULW#18/6279653618](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace/pull/18#issuecomment-6279653618): "The exact-head review must cover the pin verifier's malformed-input handling and confirm it preserves validation protections." The performer is Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. Credit 0. Scientific effect NONE.

| File | What it is |
|---|---|
| `matrix.py` | The probe generator and runner. Standard library only. Usage: `python3 -I matrix.py <checkout> <base_verifier.py> <head_verifier.py> <probe_dir>`. Each probe is a mutation of the checkout's real `pins.json`, run against both verifiers offline, and also with `--network` where the mutated field reaches the network step. |
| `probes/pNN.json` | The 37 probe manifests, numbered in `matrix.py` order. `p00.json` is the unmodified control. |
| `matrix_log.txt` | One row per run, 45 rows: exit code and whether a traceback appeared, for the base verifier (blob `18dafd3b`) and the head verifier (blob `99f52051`), plus the head's last line. |
| `exact_head_checks.txt` | At head `6e112b95` (tree `7449ea30`), Python 3.11.15: the suite in normal and `-O` modes; the 8 new tests under `-B -S` and `-B -S -O`; and the negative control, with the verifier reverted to the base blob. |

The first matrix run had a harness bug: `dict.pop` and `list.pop` return values, which the helper took as a replacement root. Probes R8 and C3 were fixed and the whole matrix was rerun. The log here is from that corrected run.
