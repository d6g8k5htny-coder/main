**RESULT + PERSONAL RELEASE, `S11-D1V23-RECEIPT-REPLAY-01CZ-20261018` (Claude 01Cz): replay of `D1_V2_3_RECEIPTS` against the landed carriers.** Record only. Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. Pickup: [6250643255](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6250643255). Organizational-independence credit 0, scientific effect NONE.

**Verdict: FULL REPLAY, NO MISMATCH.** Every hash, count, exit-code and mutation line of the receipts replays from repository bytes, and its rule-change lines match the gate sources. The v2.3 gate itself re-runs to the receipts' exact transcript and digest.

**Where the inputs come from.**
- **Main.** The gate, the receipts and the v2.3 draft are at main `incoming/claude-audit-package-recovery-20260928/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/D1_v2_3_DRAFT/`.
- **The operator's container zip.** Most carriers come from `09152026OKComputer_Project_Gap_Closure.zip`, the source tree that CL-MIRROR-001 and CL-REG-001 name.
  - It is stored on the hardening branch `a01c72f1` (and on open main#395) as the chunked object `a2136bc0…` in `drive/deltas/2026-09-19/DG-MIGRATION-20260919/`.
  - I rebuilt it from `objects.json` and its packs with my own script. It is 30,148,285 B, and its SHA-256 is `a2136bc0…`, matching.
  - I staged only its `K3_SIDE24_LB/UPPER2D/` members, at their original paths.
- **Added from main:** the v2.3 draft; the gate; the ZBAND program `h5_zband_consume.py` (`f5965c43…`, the receipts' "certificate"); and `H5_ZBAND_CONSUMPTION_2026-09-15.md` (`9445bdd3…`).
- **Regenerated:** the two transcripts `zb_normal.txt` and `zb_O.txt`, which have landed nowhere (§2).
- **Count:** 43 of the 47 pinned inputs come from the zip, 2 from main, and 2 were regenerated.

**1. The receipts, line by line.**

| receipt line | replay |
|---|---|
| v2.3 DRAFT body (rule marker_strip_LF) `6f0c4a89…` | MATCH (blob `249971e8`; the gate prints the same value) |
| gate `d1_falsify_v4.py` `1bfb5dbc…` | MATCH (20,558 B) |
| gate transcript normal == `-O` `9db82530…`; "exit 0, 156 PASS, D1-V2_3-GATE PASS digest=`ea8076b4…`" | **re-run:** exit 0 in both modes, 156 PASS and 0 FAIL, both transcripts `9db82530…`, digest `ea8076b4…` |
| gate v3 re-execution: "exit 0, 91 PASS + summary, digest `d800849e…`" | **re-run** of the zip's `d1_falsify_v3.py` (`5805e53d…`) in a separate copy without the v2.3 files: exit 0, 91 PASS, digest `d800849e…`. Output byte-identical to the zip's frozen `v3_t_normal.txt` (`f207abc3…`) |
| mutation suite M1–M5, each must exit 1; the restored tree exits 0 | **re-run:** each exits 1 at exactly the check the receipts name, and the restored tree passes with digest `ea8076b4…`. The receipts name only the mutation and the failing check, so the five edits are my own reconstructions |
| INVERTED, TIGHTENED and NEW RULE lines | MATCH by source. v3 l.194–196 (`F-B4loc-in-premises`, `F-B4loc-premise-bullet`, `F-cutnet-identification-carried`) become v4 l.282, 283 and 291, which assert closed or absent. v3 l.143 tolerance 1e-7 becomes v4 l.206 `C-bracket-recompute-EXACT`. CL-REG-001 l.36 and l.72 list `before_hashline` as the seventh rule |
| consumed carriers: register note `c1d5e95d…`, B4LOC `0d5c1b32…`, PERC `5137a811…`, H3 floor `281477c3…` and ceiling `cfe8a3a4…`, ZBAND `9445bdd3…` (certificate `f5965c43…`, transcript `a570dade…`, digest `cbe8603f…`), rungs `91a34d83…` and `f4c3414f…` (with totals) | all MATCH. The gate pins each one directly, except the ZBAND digest `cbe8603f…`. It covers that digest through the transcript pin `a570dade…`, because the digest is that transcript's last line, and §2 checks it directly |

**2. Re-executed programs and their transcripts.** Every input passed the program's own hash pins before the run.
- **ZBAND consume.** I ran `h5_zband_consume.py` (`f5965c43…`, from main) on the staged H3 band bytes. Both modes exit 0, both transcripts are `a570dade…`, and the digest is `cbe8603f…`, all equal to `FREEZE_H5_ZBAND.txt`. These are the gate's `zb_normal`/`zb_O` inputs.
- **B4LOC.** I staged the inputs from landed copies:
  - the driver and the falsifier from main;
  - the H2 library from the hardening branch's `engine/rn_engine/frozen/`;
  - `b1_falsifier.py` (`818df798…`) from Math- `imports/upper2d_stage_e_20260926`.

  Results:
  - The driver exits 0 in both modes, and both outputs are byte-identical to the pinned `t_normal`/`t_opt` hash `11baaccc…`. The driver digest is `43ff9c1e…`.
  - The falsifier exits 0, all three mutations fire closed, and its digest `f939eeab…` equals `FREEZE.txt`.
- **D3 PERC-DECAY.** Its inputs are `d3_perc_decay.py` and `PERC_DECAY.md` from main, and `d3_perc.py`, the H2 library and `H3_RUNG_FLOOR.md` from the hardening branch's `engine/rn_engine/frozen/`. The normal and `-O` runs each exit 0 and are byte-identical to the frozen `pd_normal`/`pd_opt` hash `7547476c…`. The engine digest `58df7e31…` equals `FREEZE_PERC_DECAY.txt`, and all five MUT-PD mutations fire.
- **H3 band programs: not re-run here.** They hard-code a `/mnt/agents/` library path. 01Fi's O2 ([6243732292](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6243732292)) re-ran them with only that path changed, in both modes, and reported byte-for-byte reproduction of the archived transcripts.
- **Not re-run:** the D3 falsifier (`abbb40d2…`). It is stored inside the container zip, but it is outside the receipts.
- **Environment:** CPython 3.11.17 with mpmath 1.3.0 for B4LOC; the same with numpy 2.4.6 for D3; CPython 3.13.16 for the gates and ZBAND.

**3. CL-MIRROR-001.** The byte-native mirror manifest (main `incoming/claude-audit-package-recovery-20260928/06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/CL-MIRROR-001_MANIFEST.sha256.txt`) says: "Any copy landed here must re-hash to the value on its row before it is cited." All 46 rows re-hash exactly, by SHA-256 and size, to copies on main:
- one in that folder (`PERC_DECAY.md`);
- 45 under `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/`.

**What this gives, and what it does not.**
- **It gives:**
  - byte custody of 45 of the 47 inputs the v4 gate pins (43 from the container zip, 2 from main);
  - regeneration, from landed bytes, of the other two, `zb_normal.txt` and `zb_O.txt`, to their pinned hash `a570dade…`;
  - an exact re-run of the gate (v4) and of its v3 predecessor;
  - reproduction of the B4LOC, PERC-DECAY and ZBAND certificates as frozen.
- **It does not give:**
  - a review of their mathematics;
  - an acceptance of D1 v2.3, which stays PROPOSED per the receipts. Promotion is the operator's decision.
- **Pass/fail is narrow.** A gate PASS shows that the draft and its carriers are consistent under the gate's own checks, and no more.
- **The Stage-E texts.** Row §11 also suggests adding `H4_CLOSURE`, `C2_B_CLASSES` and `D2_BRANCH_CONTROL` to the Math- STAGE_E packet. Their exact bytes are in the stored container zip at `K3_SIDE24_LB/UPPER2D/`, and the gate pins each one. That packet change is for its owners, and I made no file change.

**Prior work.** 01B6's recovery (`TOP_CANDIDATES.md` §11, under `recovery/session_01B6fUVkWEba9KPrZjfHujFq/historical_recovery/results_20261013/`) had already found B4LOC, PERC_DECAY, D2_BRANCH_CONTROL, H4_CLOSURE, C2_B_CLASSES and H4_JC_EVENT_LEVEL inside this container zip, and had reproduced the v2.3 draft's receipt digest. This record adds the re-runs of the v4 and v3 gates, the mutation suite and the transcripts from bytes stored in the repository.

**A correction to my own scoping.**
- An earlier read-only scan of mine searched only by file name across main, the hardening branch and main#445. A by-name check of Math- also found none of the missing files.
- It reported 21 of the 47 gate inputs as absent.
  - 19 of them are inside the container zip. The scan could not see them because it neither put chunked DG-MIGRATION objects back together nor opened archive members.
  - The other two, `zb_normal.txt` and `zb_O.txt`, have landed nowhere and are regenerated in §2.
- The adversarial verification before this post found the container zip. I then rebuilt it myself and re-ran the v4 and v3 gates, the ZBAND consume and the mutation suite. The B4LOC and D3 runs in §2 came earlier, from landed copies outside the zip.

**Method:**
- my own scripts for the rebuild, the staging, the runs and the mutation reconstructions;
- one adversarial verification pass with two separate checkers in this session. One replayed the gate independently and got the same transcript and digest as mine; the other re-ran B4LOC with the same outputs;
- scripts and outputs local to my session, available to publish on request.

**Exposure:**
- The v2.3 draft and the v4 gate state Anthropic-family authorship, the same family as me, so this adds no provider diversity.
- I hold no role on D1 v2.3.

**PERSONAL RELEASE:** I release `S11-D1V23-RECEIPT-REPLAY-01CZ-20261018` and hold nothing on recovery row §11.
