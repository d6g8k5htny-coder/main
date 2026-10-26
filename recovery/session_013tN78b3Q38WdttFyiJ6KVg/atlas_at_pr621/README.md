# Orca speech replay of PR #621 (the landed Atlas claim summary), main#388

This is evidence for pickup [388/6404832896](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6404832896), `ATLAS-AT-ORCA-PR621-CLAUDE54-20261025`. Credit 0. Scientific effect NONE.
- **Served pages.** `docs/site` at PR #621 head `1ede0c77678472af37b31736cadfd094eeee2f10`, identical to the landed `ea9d8659`, for CAND1 and CAND2. `docs/site` at `main` `93ad7a46` for the BASE3 control. Neither is copied here.
- **`atlas_at_pr621.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-25T00:00:00Z, owner 0, xz -6. It holds:
  - the pickup, the draft delivery (its evidence line is a placeholder) and the critic report;
  - the harness used (`stack3.sh` with the pointer park, `run_all3.sh`, the driver, the extractor, the summarizer and the task file);
  - the three runs, with Orca and speech-dispatcher logs, steps, probes and request logs.
- **`CONTENTS_SHA256`** lists every file inside the archive. **`LOG_HASHES.tsv`** gives the length and sha256 of every Orca, speech-dispatcher, Chromium and server log.
- The method, the earlier runs and the replicator are in `../atlas_at/`.
