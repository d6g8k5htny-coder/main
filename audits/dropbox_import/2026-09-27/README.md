# Dropbox import — 2026-09-27

Provenance record of project material that existed only in Dylan's Dropbox.
**Scientific effect: NONE.** Nothing here promotes a claim or a status. Files keep
the claims their authors made at the time, and current Git and Drive carriers
govern present status. Treat every file as historical input for review.

This import follows the owner's standing rule
[OP-PRIVACY-20260927](../../../governance/OP-PRIVACY-20260927.md): only material
related to the project is published here, and nothing personal is.

## What was compared

The Dropbox was listed through the Dropbox connector (5,105 files). Each file
was matched against every blob reachable from any ref of the ten account
repositories and against the Drive source map (snapshot 2026-09-17), using
`tools/dropbox_reconcile.py`:

| tier | files | size |
|---|---:|---:|
| `GIT_EXACT` | 244 | 2.9 MB |
| `DRIVE_EXACT` | 277 | 3.0 MB |
| `GIT_NAME_SIZE` | 909 | 35.6 MB |
| `DRIVE_NAME_SIZE` | 2,529 | 46.4 MB |
| `DUP_IN_DROPBOX` | 275 | 5.6 MB |
| `SKIP_SOFTWARE_BUNDLE` | 235 | 30.7 MB |
| `SKIP_CACHE` | 83 | 2.6 MB |
| `SKIP_RUN_DEBRIS` | 180 | 59.2 MB |
| `MISSING` | 373 | 119.1 MB |

Byte-exact tiers use the Dropbox `content_hash` (SHA-256 over the SHA-256
digests of 4 MiB blocks). Name+size tiers are probable matches, not verified
bytes.

## What happened to the 373 unmatched files

| outcome | files |
|---|---:|
| Published here, byte-identical to the Dropbox original | 93 |
| Published here as extracted text of a PDF or Word file | 14 |
| Project material that could not be imported here (listed in `NOT_IMPORTED.csv`) | 26 |
| Kept private: personal information, keyed share links or confidential review material | 65 |
| Kept private: the owner's other research, not part of this project | 157 |
| Kept private: relation to the project not confirmed | 6 |
| Not imported: other archives, binaries or unreadable files | 9 |
| Empty | 3 |

Only published files are named here. Files kept private are not listed, because
their names could themselves reveal private or unrelated material. They remain in
the owner's Dropbox, and a full manifest is in the owner's private Drive.

- `MANIFEST.csv`: one row per published file, with the Dropbox path, size,
  Dropbox content hash, disposition, archive path, SHA-256 of the archived bytes,
  a suggested home and a one-line summary.
  - `PUBLISHED_EXACT_ORIGINAL`: `files/<Dropbox path>` is byte-identical to the
    Dropbox file. The connector returned its text, and those bytes were accepted
    only because they reproduce the file's Dropbox `content_hash`.
  - `PUBLISHED_EXTRACTED_TEXT`: `files/<Dropbox path>.txt` is the connector's
    extracted text of a PDF or Word file, not the original. Layout, figures and
    equations may be lost.
  - `suggested_home`: 11 files are marked `Math-imports-candidate`: a
    mathematical proof, calculation or code that a Math- reviewer should look at.
    The rest are program history, operations and reviews.
- `NOT_IMPORTED.csv`: project files known to exist only in Dropbox that this
  container could not import. They are archives, figures, and logs or maps over
  256 KiB, and the reason is given per row. Many archives' contents are already
  here as their extracted loose files.

Paths are relative to the Dropbox root, except that `Github/` is the shared folder
at `2026/Research 07-13-26/Research topology 07-14-26/Github`.

## How files were chosen

1. **Privacy.** Each recovered text went to an agent instructed to read it in full
   and propose publish, hold or skip. Two independent adversarial verifiers then
   re-read every publish proposal. One checked personal data, secrets and keyed
   links; the other checked third-party copyright and provenance. Regex scans for
   credentials, emails, phone numbers, keyed URLs and GPS pairs ran first.
2. **Relation to the project.** A second pass classified each privacy-cleared
   file against the project's scope ([README](../../../README.md),
   [research index](../../../docs/RESEARCH_INDEX.md)). A skeptic re-checked every
   "related" call, and only files both agreed on are published.
3. **Final scan.** The exact published bytes were scanned for the personal values
   found in held files, and for credential, email, phone, social-profile and
   share-link patterns. No match remained.

Agents were instructed to read in full; that is an instruction, not something this
record can prove. Any doubt meant the file was kept private.

## Limits

- The Drive side is the 2026-09-17 source-map snapshot.
- Name+size matches were not re-verified byte for byte.
- The container could not download from Dropbox (`*.dl.dropboxusercontent.com` and
  `www.dropbox.com` were refused by its network policy). Only text the connector
  could extract was recovered.
- Automated review can miss things. If anything here should not be public, remove
  it; the originals remain in Dropbox.

Generated 2026-09-27 by Anthropic / Claude (Claude Code); the Dropbox was treated as
read-only.
