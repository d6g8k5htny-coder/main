

### M. Round-4 audit, 2026-10-06 01:54–02:00Z (same session)

**M1. Merges since section L.** Math-#315 `99ba2dfe` at 01:49Z (C8 integrator; parents `a19f1076` + `5c24fea3`; four landed blobs `5790aaa0` / `20ca26bd` / `c0b1d5e9` / `07a23d35` equal the reviewed head `5e027afc`, verified by me; tip checks 3/3). Math- tip `99ba2dfe`; main tip unchanged at `3e3a2df1`.

**M2. Open queue and owners at 02:00Z.** #319 (Astra IBA2-012 count-weighted transfer; Cursor PASS_SCOPED 6007644377; author release asked a Claude lane to integrate; C8 picked it up at 01:55:49Z, 6007735247, with its tip lock 6007735654; my own pickup script read that pickup first and stopped without writing, so no duplicate). #316 (Astra follow-on to #305; Codex integration pickup 6007601305 waits on the tip-lock release after #319). #317 (Astra Lean companion on the D2 moment equality cases that AUD-309-REVIEW-STRICTNESS-01 raised; draft, head still moving with the `probReal_univ` fix, dedicated check compiling; the author asked a Claude lane to read it and I am lined up for that read once the head is frozen and green, 317/6007753131, not yet a pickup). #318 (Astra draft, LM006 tilted endpoint-law bounds). #320 (Astra's isolated repair for #307's replay wrapper, test-first RED checkpoint stacked on `b157df1b`; the two replay jobs fail by design at this checkpoint; Astra acknowledged my contract handoff, 307/6007660860). #321 (Astra's six-file repair for issue #314, the same crash-as-rejection class in the D5/C6 workflows; Cursor `bc-ad3a60f7` reviewing; the author found a whitespace-prefix defect in its own helper and holds the amendment until the read closes). #312 (Sol draft; three findings open). #307 (QS lane, now active again: it published addendum A3.8 with controls, 6007704303/6007706590, and a Cursor Slice 2 read is picked up; #307 itself waits on #320). Nothing in this queue waits on me.

**M3. Pattern worth naming.** Three independent workflows carried the same engineering defect, a replay gate that accepted any exit 1 as a mutant rejection: #302's (fixed in `890b9ea`, regression #303), the periodic-jet packet's (#311), the QS chain's (#320 pending) and the D5/C6 trio's (#321 pending). Every fix follows the pattern first used on #302: named failure contracts, empty-stderr baselines, strict JSON, and a real-subprocess regression file under `tests/`. Owner action: none; the lanes are converging on it without a directive.

**M4. Other surfaces.** Drive: the register `GP-REG-032-v1.2` was written again at 01:42:58Z, coincident with C8's #313 landing receipt (01:42:52Z) and #315 pickup (01:42:55Z), consistent with claim-ledger rows as in L4; no other Drive object changed. Dropbox: still nothing newer than 2026-09-28. Consensus: quota exhausted.

**M5. Tooling.** No new refusal this round; the #319 script stopped at its own ownership check, before any write.

**M6. Timers.** One check-in of mine at 02:53Z, prompt refreshed to this state; no loop.
