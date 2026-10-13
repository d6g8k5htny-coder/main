# Recovery checkpoint: session `session_01B6fUVkWEba9KPrZjfHujFq` (Claude 01B6)

This answers the owner's recovery request on main #307
([6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988),
[6136210973](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136210973)),
following the implementation decision
[6136231457](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136231457).
It publishes this session's still-local project tooling, so that a session or account
interruption does not force anyone to re-derive it.

- **Scientific effect:** NONE. Nothing here is a proof, a review verdict or a status change.
- **Organizational-independence credit:** 0.
- **Retention:** this branch is a payload branch. Please keep it while any work relies on it,
  until a verified durable replacement holds the same bytes and the index is updated
  (per [6136476552](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136476552)).
  Nothing on it is for merge.

## What is here

| group | files | what it is | stage |
| --- | --- | --- | --- |
| `historical_recovery/tools/` | `build_corpus.py`, `supplement.py`, `cs.py` | Builder and search helper for a local, provenance-indexed text corpus. The corpus covers every blob reachable from every ref and object in 11 local project mirrors, plus the hardening Drive custody store (DG-MIGRATION-20260919) and the Dropbox text exports. `cs.py` prints source, object id, SHA-256, kind, bytes and names for each hit. | working tool, built 2026-09-28 |
| `historical_recovery/gaps.json` | 1 | Inventory of 155 open gaps (G01–G155) merged from three inventories: `Math-` at `a1fd679`, `main` at `2eaf083` and hardening at `a01c72f`. Each row has id, cluster, lane, name, aliases, statement, current status as the sources state it, status sources, cited artifacts and missing carriers. | snapshot of 2026-09-28; statuses are transcriptions, not verdicts |
| `historical_recovery/workflows/` | `historical-math-recovery.js`, `current-gap-inventory.js` | Multi-agent workflow scripts. The gap-inventory run completed and produced `gaps.json`. The historical search run (15 search groups, an adversarial verify per strong fragment, then a synthesis) never ran: every agent hit a usage limit on 28 September. | inventory done; search NOT_RUN |
| `grok_held_survey_20261013/` | `fetch.py`, `classify.py`, `after.py` | The read-only scripts behind the Grok-held pickup survey [307/6154355553](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6154355553). | delivered and released |

`MANIFEST.json` lists every file with its byte count, SHA-256, provenance, stage,
supersession and next step. `SHA256SUMS` covers every file in this directory except itself.

## Running the tools

- **Paths:** the scripts hard-code this session's scratch directory and import
  `tools/dropbox_reconcile.py` from a `main` checkout at `/home/user/main`. Edit the path
  constants at the top of each script before running them elsewhere.
- **Dependencies:** `supplement.py` needs `pypdf` and blocks the optional `cryptography`
  import.
- **Example search:**
  `python3 -B cs.py -e 'REGEX' [-i] [-F] [--src git|custody|dropbox|pdf] [--lines K]`.
  Use `--show FILE --range A:B` to print text, and `--prov FILE` to print provenance.

## Deliberately not included

These are recorded by identity only. They are private, regenerable, or already on GitHub.

- **The corpus itself** (about 6.1 GB of extracted text, 91,276 index rows; `index.tsv`
  SHA-256 `aa32b745efdda2ba7b50bd76827806c803e2365b38c0af362c611b1a3033d07d`).
  - It holds text from the owner's private Drive and Dropbox archives, so it stays in this
    session only.
  - It can be rebuilt with the tools above from the mirrors and archives.
  - Any lane may ask Claude 01B6 to search it while this session lives. Answers come back as
    provenance only.
- **`gaps_raw.json`** (the three raw inventories behind `gaps.json`; SHA-256
  `b0086bcf62179ab99a27f0fa7af4e09385e95eb7253dd50c512fe0ab25ff995f`). It carries about 200
  Drive file identifiers. The project has neutralized public custody locators, so the file
  stays out. It can be regenerated from the three pinned commits.
- **Dropbox reconciliation working data, Drive search results and the 27 September
  privacy-redaction tooling.** These are private data, or work the owner asked to stop.
- **Already on GitHub, byte-identical:**
  - the Grok uploads review, Parts 1–4: `main` commits `544fafbf` and `8d0061e5`;
  - the formal-enforcement evidence bundle: main #192;
  - `tools/dropbox_reconcile.py`: main #184;
  - the OP-PRIVACY clarifications: main #189.

## Next step

The historical search (`historical-math-recovery.js`) is unrun. Whoever resumes it must follow
the original task's no-invention rule:
- label any rebuilt argument "RECONSTRUCTED CANDIDATE — NOT ORIGINAL CARRIER";
- change no scientific status.

It should be routed on #307 before anyone starts it.
