# C8 recovery delta, 13 October 2026, about 20:30Z

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), for Dylan Roy. Scientific effect NONE.

This adds to the C8 recovery copy made at 15:03Z (commit `eceaf907`; see `../MANIFEST.json` and `../README.md`, both unchanged). It answers the owner's standing request [main#307/6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988): still-local work is preserved so that an interruption does not force anyone to re-derive it.

## Files

| Path | Bytes | SHA-256 | What it is |
|---|---:|---|---|
| `site_run_readback.py` | 4,358 | `1705d6c5c358b9b64b3294d6c7f35ae75c8c995f57cbcbb1103039559cf3b90a` | C8's independent exact-run readback for the current-site workflow |
| `validation/38432795042/READBACK.json` | 4,236 | `3f3cc88fae9dffdeb98c243157578e41d335c3aaa257324fc59572530b9d14ee` | The script's validation output on the first production run |
| `watch.sh` | 3,232 | `a459fff57a78faa7d0d56efac239e376d1d219500f839074367874bda3672acc` | C8's read-only watcher, version 11 |

`SHA256SUMS` in this directory lists the same hashes.

### `site_run_readback.py`

- **Usage:** `python3 -I site_run_readback.py RUN_ID [OUTDIR]`. It needs `gh api` read access to `d6g8k5htny-coder/main`.
- **What it reads:**
  - the one announced run;
  - the receipts branch `codex/site-publication-receipts`: its head, three parents and `latest.json`;
  - the three public files: `snapshots/index.json`, `current-meta.json` and `current.json` under `https://d6g8k5htny-coder.github.io/main/site/`.
- **What it checks:** each public file's bytes and SHA-256 against the receipt. It writes `READBACK.json` and copies of the three files into `OUTDIR`.
- **What it never does:** list runs, dispatch, rerun or write anything remote. The sole observer, `01a11d37`, owns run discovery.

### `validation/38432795042/READBACK.json`

- Run `38432795042` was an ordinary `push` run that concluded `success`. The receipt head at the time was `142a06aa`.
- All three public files matched the receipt. The read ran from 18:32:00Z to 18:32:03Z.
- This is evidence that the script works. It is not the current publication: later runs (for example push run `38455779711` for main#390) have since published.

### `watch.sh`

- It polls the coordination threads and C8's open items every 60 seconds, read-only.
- Its comment patterns and state paths are specific to this session.

## Open obligations at this checkpoint

1. **Site.** When the sole observer `01a11d37` announces the first genuine `event=schedule` run of `.github/workflows/current-site.yml`, run `site_run_readback.py` on it and post the result on [main#387](https://github.com/d6g8k5htny-coder/main/pull/387). As of this checkpoint, no scheduled run has been announced.
2. **Math-#454** (C8 is the author). Head `d517287c` has a non-Claude delta review PASS ([5508838177](https://github.com/d6g8k5htny-coder/Math-/pull/454#pullrequestreview-5508838177)). It waits for a new, distinct non-Claude integrator. After it merges, C8 does the author landing check. C8 never merges it.

## Done since 15:03Z, all recorded on GitHub

- DOC358 readback: PASS ([main#358 review 5508651060](https://github.com/d6g8k5htny-coder/main/pull/358#pullrequestreview-5508651060)).
- BQ38 r3: delivered in [main#275/6153450094](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153450094), and the readback passed ([6153678987](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153678987)).
- Math-#371: merged as `9844f785` on the owner's action-time approval.
- Math-#454 amendment: `d517287c`.

The session's drafts of those posts are not copied here, because the posted versions are the record.

## Excluded

- No credentials, personal data or private account or host information.
- No copies of public site files or repository files, which are recoverable from their own sources.
