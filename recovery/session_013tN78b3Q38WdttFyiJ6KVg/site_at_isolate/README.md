# SITE screen-reader cause isolation on minimal pages (main#539)

This is evidence for pickup [539/6402450999](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6402450999), `SITE-AT-ISOLATE-CLAUDE54-20261025`. Credit 0. Scientific effect NONE.

The pages were written from scratch for this test; they contain no SITE bundle bytes. They were run under Orca 46.1 with Chromium 141, and Chromium had no route to the proxy.

- **`site_at_isolate.tar.xz`** (sha256 `d04bfa801449d845375bdf8036dcc3bda4eec7d400be75bebf3706cf9630b415`) is one deterministic archive: GNU tar, sorted names, mtime 2026-10-25T00:00:00Z, owner 0, xz -6. It holds:
  - `pages/`: the minimal test pages;
  - `scripts/`: the harness;
  - `evidence/`: the runs SELECT1–4 and STATUS1–4, each with Orca's debug log, the speech-dispatcher log, the steps, AT-SPI probes, server and CDP request logs, and the proxy status before and after;
  - `src_notes.md`: the Orca 46.1 source reading;
  - the stage results.
- **`LOG_HASHES.tsv`** gives the length and sha256 of every original Orca and speech-dispatcher log.
