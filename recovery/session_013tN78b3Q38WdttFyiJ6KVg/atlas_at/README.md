# Orca speech on the current Atlas, baseline against a status-wrapper variant (main#388)

This is evidence for pickup [388/6404030858](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6404030858), `ATLAS-AT-ORCA-CURRENT-CLAUDE54-20261025`. Credit 0. Scientific effect NONE. The served pages are `docs/site` at `main` `93ad7a463348743d08a6db2c21b4e4f01b1942d0`; they are not copied here. The two scratch variants are recorded as `variant.diff` and `variant2_select.diff`.

- **`atlas_at.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-25T00:00:00Z, owner 0, xz -6. It holds:
  - `pickup.md`, `checkpoint.md` and `delivery.md`. The archived `delivery.md` still has the evidence line as a placeholder; the posted text is read back beside this README once posted.
  - `variant.diff` and `variant2_select.diff`;
  - `scripts/`: the executor's harness (`stack.sh`, `stack2.sh` with the shared lock, `drive.mjs`, `xkey.py`, `extract.py`, `summarize.py`, the task files);
  - `evidence/`: the runs BASE1, BASE2, VAR1, VAR2 and SEL1. Each has Orca's debug log, the speech-dispatcher log, the steps, DOM and AT-SPI probes, server and CDP request logs, the proxy status before and after, and `summary.json`/`summary.md`;
  - `rep/`: the independent replicator's own scripts and its runs REP_BASE1 and REP_VAR1, plus three pilot runs (not counted), and its report `rep_result.json`;
  - `replicator_brief.md`, `critic_brief.md` and `critic_result.json`;
  - `SUMMARY_NOTES.md`: the critic's label corrections to the generated summaries.
- **`CONTENTS_SHA256`** gives the sha256 of every file inside the archive.
- **`LOG_HASHES.tsv`** gives the length and sha256 of every Orca, speech-dispatcher, Chromium and server log.

Paths inside these files are left as recorded.
