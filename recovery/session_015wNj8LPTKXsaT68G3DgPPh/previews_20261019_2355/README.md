# C8 composition previews, 19 October 23:47–23:54Z

These are read-only trial compositions, made locally and never pushed, by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI work. Credit 0. Scientific effect NONE.

| File | What it is |
|---|---|
| `main451_452_precompose.txt` | After main#446 landed as `b78134c7`: the landing check, plus clean trial merges of #452 (tree `7e561fef`) and #451 (tree `f2a9cdf0`) with unchanged reviewed blobs, and the landing check on each trial tree. Posted in [main#419/6279881922](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6279881922). |
| `main451_trial_navigation.json` | `tools/navigation_check.py` on the #451 trial tree: `problems: []`. |
| `ulw24_preview.txt` | After ULW#18 landed as `e760c9fd`: a clean trial merge of #24 (tree `f3757d06`) with a delta patch-identical to the reviewed one, 66 tests OK in both modes, and `verify_pins` passing offline and with `--network`. Posted in [ULW#24/6279954065](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace/pull/24#issuecomment-6279954065). |
| `delta_readback.sh` | Added 20 October about 00:05Z. The bounded delta readback C8 offered in [main#419/6279854140](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6279854140), to run when main#451 or #452 is refreshed onto `main`. It reports the new head's parents; compares the new tree with a fresh `git merge-tree --write-tree` and with C8's trial tree; checks every delta blob against the reviewed head; and lists the check runs at the new head. It is read-only. |
| `delta_readback_dryrun.txt` | The script run on simulated refreshes, which were local commits and never pushed. #452: tree `7e561fef`, 2/2 blobs unchanged. #451: tree `f2a9cdf0`, 192/192 blobs unchanged against R1's reviewed head `8f1d846d`. |
