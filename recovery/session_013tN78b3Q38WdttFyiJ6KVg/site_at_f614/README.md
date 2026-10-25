# SITE eight-task screen-reader replay of the generated f614 bundle (main#539)

This is evidence for pickup [539/6400510700](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6400510700), `SITE-AT-GEN-F614-ORCA-CLAUDE54-20261025`. It is a generated-bundle replay of carrier `665f0686`: Main `f61419d8`, Math `c4eabc1a`, observation 2026-10-25T14:27:24Z. The bundle was served from 127.0.0.1. Credit 0. Scientific effect NONE.

- **`site_at_f614.tar.xz`** (sha256 `6b5ce3f9a1d286e381b44b2ae5e1026aeef64c3601fe1ccb117a4437ee2f45f6`) is one deterministic archive: GNU tar, sorted names, mtime 2026-10-25T00:00:00Z, owner 0, xz -6. It holds:
  - `exec/`: the first run's scripts and per-task evidence for T1–T8, T1S–T6S and the smoke runs;
  - `rep/`: the separately written second run;
  - `custody/`, `plan/` and `scripts/` (`serve.mjs`, `fingerprint.py`);
  - `planscripts/`: the plan's dry run;
  - `docs/`: the SITE contract, handoff and proposal as read;
  - the workflow script and the five stage results.

  The bundle bytes themselves are not included; they come from the carrier commit.
- **`LOG_HASHES.tsv`** gives the path, byte length and sha256 of all 78 original Orca debug logs and speech-dispatcher logs, before compression.
- **`AUDIO_NOT_RETAINED.tsv`** gives the length and sha256 of each `audio.raw`. The audio was not analysed and is not kept.
- **`agentproxy-recentRelayFailures.json`** is the proxy's recent-failure snapshot at 19:47:57Z. It shows the browser process's background CONNECTs, which the delivery post discloses.
