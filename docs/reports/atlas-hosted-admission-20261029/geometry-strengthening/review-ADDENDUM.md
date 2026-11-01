# A1 follow-up — exact current helper

OpenAI / GPT-6 / Codex, read-only technical follow-up requested by the parent. Prior Atlas/formal notes and original A1 finding are explicit exposure; not blind or human, organizational-independence credit0, scientific effect NONE. No approval gate or source/integration ownership is created.

**A1 disposition: FIX VERIFIED for helper SHA256 `f1552c954e711cfcb8281723f2579425df16af2aea7ee9dcde5cd8f23684ab02`.** Runner SHA256 remains `ced49ece18704510b8e4e70936fc04fd43f471cb6973d1fd0dccc37791b863a2`. At pickup, HEAD was `420bc28134ad495257a167af31b88748a48c95c8`, tree `816991105ecc425c415563eeb0f5d59065e6fac8`; the helper was the sole modified tracked file. This review qualifies the exact working-tree helper bytes, not an invented committed helper revision. The old `ab39e802…` finding and evidence remain unchanged.

The source-owner diff in `_restore_geometry_and_pick` captures prior view/panel, selects Graph only when the prior view is not Graph, and records `browse_navigation` and `graph_selection_dispatched`. Thus the already visible Graph path reaches geometry assertions without the selector-triggered full render that masked the old defect. Intentional List-to-Graph and narrow Evidence-to-Browse journeys remain operational and visibly labelled in the report. The change directly addresses A1's demonstrated cause.

Completed parent integrated reports and their exact source bindings were read and hashed:

| Report | Cases | Passed | Observed rejection |
|---|---:|---:|---|
|old helper + no-reflow|45|45|None: retained false-admission baseline|
|fixed helper + correct source|45|45|None|
|fixed helper + no-reflow|45|42|Desktop positive, module390 and noGL1200 fail specifically on overlapping cards|
|fixed helper + E1|45|43|Both positive paths fail original focused SUMMARY preservation|
|fixed helper + blocked Three|45|43|Both positive paths fail actual-rendering requirement|

All four current reports bind the exact f155 helper and ced49 runner. They contain no top-level execution error; expected rejected rows carry the stated errors. The parent reported correct process0 and no-reflow process1. I inspected the wrapper's `raise SystemExit(m.main())` propagation and the actual report/log outcomes; I did **not** independently rerun every full45 process. This distinction preserves actual execution attribution.

Personal targeted check: the same in-memory no-reflow HTTP response mutation, with no checkout/fixture file edits, now yields exactly3 failed and3 passed helper cases. All three failures are the expected geometry cases, with `graph_selection_dispatched:false` and the overlap assertion as the reason. The old personal control had admitted all6. See `followup-no-reflow/report.json`. A fresh normal-mode run of the13 current contract/seed/preservation tests passed on this sidecar's default Python3.12. Parent normal/optimized13-test logs were separately read and hashed. The thirteen include five report-preservation contract tests; I make no separate independent certification of the remote c47 commit identity.

No remaining Critical, Important or Minor source concern was identified in this bounded fix. Remaining coverage/exposure limits:

- Narrow positive recovery deliberately navigates Evidence→Browse, and List fallback deliberately selects Graph. Those actions may render. Their passing geometry proves the recorded navigation/recovery journey, not untouched immediate geometry while a hidden graph remains hidden. Current route fields distinguish this correctly.
- Publish/bind the final committed helper and updated report to these exact source hashes; old ab39 reports must remain the old phase. A working-tree hash is not a final commit/CI identity.
- Python3.11 restoration/TAR modes, remote c47, hardware/browser/AT qualification and global CBZ gates are outside this follow-up. No claim concerning them follows from the helper or unit results. I used the available default3.12 for personal checks.
- The expected browser rejection controls are bounded experiments, not a proof against arbitrary malicious code that forges every oracle.

`ADDENDUM-EVIDENCE.json` records all report and binding hashes, failed-case reasons, source/diff identity and attribution. This personal follow-up is complete and released; no source edits, install, remote mutation, new global tool, new required gate, or foreign release was performed. Parent source publication and other work continue under their own authority.
