# R2 replay venue: different-provider replay of the zeta fold slice C package

**Status: venue only.** No replay has run in this record yet. The packet README's
§9 slice C line stays NOT_PERFORMED. Scientific effect NONE.

## Why this exists

- **R2 of main#451** is the full replay of the 59 README §8 commands of
  `experiments/zeta_fold_exploration` by a provider other than Anthropic.
  - It is split into chunks in the [ledger](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281117591).
  - The coordinator's open offer is [main#451 comment 6281445636](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281445636).
- **A Cursor cloud agent starts from a pull request's head branch.** main#451's
  branch was deleted after its merge, so a request there failed at startup
  ([6281525014](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281525014)).
  This draft gives one verified non-Anthropic execution worker a place to run
  the replay and record it.
- **Owner approval.** Dylan Roy chose "Allow a venue PR" in Claude Code session
  `session_017Mi3hxjaxV45x6zo6o1ee3` on 20 October 2026 (about 02:25Z), with
  these conditions, verbatim:

  > Use one verified non-Anthropic execution worker for R2, check existing
  > claims first, and preserve all 59 commands, environment details, output
  > hashes and discrepancies. Keep README §9 unchanged until the complete replay
  > is reviewed. No merge or scientific acceptance is authorized by the venue
  > approval.

## Source

- **Packet:** `experiments/zeta_fold_exploration`, tree
  `bca06db86581b2ff878fa321c4040c7b8f7bd55d`. It is the same at `0a7e5145` (the
  ledger's commit), at `73a3cfcd` (the main#451 merge) and at `5cab7663` (this
  branch's base).
- **Expected outputs:**
  - the 66 committed files under the packet's `results/`;
  - the 9 raw-sample hashes in the packet's `MANIFEST.json`.
- **Commands, output mapping and timing masks:**
  `reviews/zeta_fold_slice_c_rerun_20261019/`, specifically `scripts/commands.sh`,
  `scripts/driver.py` and `comparison/timing_mask_rules.txt`.

## What the finished record holds

1. **Performer:** provider, model and session.
2. **Environment:** Python, numpy and mpmath versions, BLAS, CPU count and
   thread pins, plus every deviation from the pinned environment.
3. **Jobs:** all 59, each with its exact command, exit code and wall time.
4. **Committed output files:** for each, the SHA-256 of the replay output and
   its class:
   - IDENTICAL;
   - TIMING-ONLY, meaning equal after the masks;
   - DIFFERENT, with the diff.
5. **Raw samples:** SHA-256 and bytes, compared with `MANIFEST.json`.
6. **Residual trails:** for `indep_ht_c910.out` and `indep_ht_c3435.out`,
   compared with both the committed copies and the same-provider copies.
7. **Discrepancies:** every one, explained or not.

## Limits

- Partial chunks are progress records, not R2.
- A run by an Anthropic performer earns no R2 credit.
- This venue authorizes no merge and no scientific acceptance.
- README §9 changes only after the complete replay is reviewed, in a separately
  reviewed change.
