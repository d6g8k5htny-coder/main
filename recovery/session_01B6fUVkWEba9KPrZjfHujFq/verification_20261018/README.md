# Verification scripts behind three 18 October board records (Claude 01B6)

This folder holds the scripts behind three of this session's public records. Until now they existed only in the session's scratch directory. With them, any agent can re-derive the values listed under **Expected output** from git objects alone, without this session's conversation. **Not scripted** below lists the main posted values these scripts do not produce. This follows Dylan's standing direction on living project knowledge, recorded at [main#419/6321304701](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6321304701).

- **Scientific effect:** NONE. These are byte, index and quote checks. They review no mathematics and change no status.
- **Organizational-independence credit:** 0.
- **Retention:** part of this payload branch. Not for merge.

## What each script backs

| script | backs | what it checks |
| --- | --- | --- |
| `g90/check.py`, `g90/idx.py` | [main#419/6244603305](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6244603305) (§17/G90 locators) | Opens the DG-MIGRATION pack `55e00c96…` and the custody object `1e11b246…`, a single chunk inside it. It then finds the zip member whose SHA-256 is `a9083363…` and prints that member's index, size and CRC. |
| `rung/rebuild.py` | [main#419/6245774604](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6245774604) (H3 addendum, rung-floor paragraph) | Rebuilds custody object `a2136bc0…` from its 29 chunks in 8 DG-MIGRATION packs. Each chunk and the whole object are checked against their SHA-256. It then extracts `h3_rung_floor.py` and `h3_band_ceil.py` and prints their sizes and hashes. |
| `h3scan/scan.py` | [main#419/6243792389](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6243792389) (H3 consumer inventory) | Runs `git grep` for the 7 H3 decimals from O2 and records the direction of each quote. |
| `h3scan/scan2.py` | [main#419/6245774604](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6245774604) (H3 addendum) | Runs `git grep` for the values O7 affects and for the 4-digit band table. The output is raw candidate hits, before filtering. |
| `h3scan/filter2.py` | the same addendum | Keeps a hit only if it equals a published H3 value, rounded half-up or truncated at the quote's own number of decimals. This drops the 64 unrelated quotes. |
| `h3scan/summarize.py`, `reproduce.sh` | — | Helpers added on 21 October. `summarize.py` prints row counts, distinct files per commit and a digest. `reproduce.sh` runs everything. |

**Provenance**
- **Original scripts.** `check.py`, `idx.py`, `rebuild.py`, `scan.py` and `scan2.py` are byte-identical to the scripts that produced the posted values on 18 October. `scan.py` has three unused variables (`loc`, `txt`, `path`); they stay so the file remains identical.
- **The 18 October scan runs.** They used the branch refs `origin/main`, Math- `origin/main` and `origin/chatgpt/drive-github-hardening-20260919`. The pinned commits (main `6fce9b8c` and `3edacfe5`, Math- `4e722922` and `b6fc0e64`, hardening `a01c72f1`) are the cuts the posts name. At the pinned commits the rows are the same, once the tree label is normalised.
- **`filter2.py`.** The filtering step on 18 October was done ad hoc. `filter2.py`, written on 21 October, reconstructs it. Run on the saved 18 October raw hits, its output is byte-identical to the saved 18 October filtered output (230 rows). Run on a fresh scan, it gives the same rows.

## How to run

```
reproduce.sh MAIN_GIT MATH_GIT OUT_DIR
```

- **`MAIN_GIT`:** a plain full clone of `d6g8k5htny-coder/main` (`git clone https://github.com/d6g8k5htny-coder/main.git`). It must contain `6fce9b8c`, `3edacfe5` and the hardening head `a01c72f1937875b94bfd78e183a6dbbdb2b04da4` (branch `chatgpt/drive-github-hardening-20260919`).
  - A shallow clone lacks the old commits.
  - A `--filter=blob:none` clone works, but `git grep` then fetches about 3,000 blobs one at a time. Prefetch them first with `git rev-list --objects --missing=print a01c72f1^{tree} | sed -n 's/^?//p' | git fetch origin --filter=blob:none --stdin`.
- **`MATH_GIT`:** a clone of `d6g8k5htny-coder/Math-` containing `4e722922` and `b6fc0e64`. A plain `git clone` is enough; no API is used.
- **`OUT_DIR`:** a new, empty directory. Relative paths are fine for all three arguments.

**What `reproduce.sh` does**
- It stops if any of the five commits is missing. The scans themselves ignore `git grep` failures, so a missing commit would otherwise look like "no quotes".
- It pins git's `core.quotePath` to its default (`true`), because the scan rows store git's quoted path text.
- It needs only Python 3 and its standard library, and reads only through `git show` and `git grep`. It runs Python with `-I`, because the scripts open zip archives taken from git.

A full run takes about 10 seconds on a full clone.

## Expected output

These are from a fresh run at 17:14Z on 21 October 2026. A separate checker agent, of the same model family, repeated them later that day on a fresh full clone.

- **Inputs:**
  - `objects.json` (blob `851b4b14`): SHA-256 `f9f41b2955bfc9ab7a0a11bf199ae5b382ab621605d0f6183d80cd0aeb66fc4b`.
  - The pack (blob `f55d426e`): SHA-256 `55e00c9634b5771d95075e67445d6f5850ab215e2fdca38c7e5a0d834c6d14be`.
- **`g90/check.py`:**
  - The chunk is 178,117 bytes, SHA-256 `1e11b246…`.
  - The custody zip has 105 members.
  - It prints `MATCH index0=26 index1=27 size=4655` for member `Prize_Research_P14_Fractional_20260917/proofs/04_LIMITATIONS_AND_HIERARCHY.md`.
- **`g90/idx.py`:** 0 directories and 105 files. The member is at index 26 (zero-based) in every ordering, with CRC `aac4948b` and file size 4,655 bytes.
- **`rung/rebuild.py`:**
  - It prints `rebuilt 30148285 sha256 ok`.
  - `h3_band_ceil.py`: 37,950 bytes, SHA-256 `b97c5428a20e5dd33f3e53653d20447f1e27515dcc7dc7a7c2004ebe30029452`.
  - `h3_rung_floor.py`: 16,960 bytes, SHA-256 `094d382451ff37e8d3123e7203c743f359cb0af1c2c12cf336adf74fc0b25d8f`.

| step | rows | files per commit | digest (with `core.quotePath=true`) |
| --- | ---: | --- | --- |
| `scan.py` | 126 | `6fce9b8c` 19, `a01c72f1` 14; Math- `4e722922` none | `0d15459207e723d5e407b285b6fa6324269eb8207df2ca90eee26982e83be95c` |
| `scan2.py` | 294 | `3edacfe5` 26, `a01c72f1` 19, Math- `b6fc0e64` 15 | `171168b41ff953ab23641ade81cad8da2c61c1a0187688c4b5432884a3be9465` |
| `filter2.py` | 230 | `3edacfe5` 24, `a01c72f1` 19; Math- none | `1262ca6767a015cde44c6cb9d3a33479febd7fc328812279b075634cdf441183` |

**What the scan rows show**
- The inventory's "19 files on main, 14 on hardening, Math-: no quotes" is the `scan.py` row.
- The addendum's "Math-: no quotes", after it "excludes 64 quotes of unrelated numbers", is the `filter2.py` row (294 − 230 = 64).
- The "none" results for Math- rest on `reproduce.sh`'s check that the commits exist.

## Not scripted

These main posted values came from reading, ad hoc computation, a checker agent or this session's private corpus. No script here produces them.

- **6244603305:**
  - the mirror blob `a56e129f` and the `REV-P15-D.md` locator (`0f384c00`);
  - the §E line numbers;
  - the git blob ids and byte sizes of the pack and `objects.json`. Only their SHA-256 is printed above;
  - the correction that `m81` and `m228` are numbers in this session's private search corpus. That cannot be re-derived from git.
- **6243792389:**
  - which quoting files are singled out, and their blob ids and line numbers;
  - the live-versus-archived classification;
  - the pin and consistency verdicts (`h5_zband_consume` l.88, the `lpw_constant_v4` token checks, `d1_falsify_v4` l.252), and the reading of 3.66435028046302 in v4 as a rejected mutant;
  - the exact-fraction LPW v4 values of c, the 2.9e−25 overstatement, and the match with `RECEIPT.json`.
- **6245774604:**
  - which records are singled out, how each is characterised, and their blob ids and line numbers;
  - the 10 files new since the inventory and the count of 32 already in it. Both follow from comparing the `scan.py` and `filter2.py` file lists, but no script compares them;
  - the 29-chunk, 8-pack count. `rebuild.py` checks every chunk but prints only the totals;
  - the hash pins (`FREEZE_H5_ZBAND.txt`, `d1_falsify_v4.py.txt` line 71), found by reading outside the scans;
  - the live records that rely on the band floor without quoting it (`README.md` line 128, `claims/graph.json` lines 17 and 36, `engine/carriers/MANIFEST.json` line 179), found by a same-family checker agent;
  - the rung-floor code-pattern finding (no `matmax`/`vecmax`; lines 203–207, 227–228 and 246 of `h3_rung_floor.py`). `rebuild.py` extracts the file but does not check the pattern.

## Limits

- **`scan.py`:** matches the truncated 6-decimal prefix of the 7 O2 values. It misses quotes of them with 5 or fewer decimals, and rounded quotes whose rounding changes the first 6 decimals (3.662829, 3.673606, 3.6736060, 2.581742).
- **`scan2.py`:** covers 4-decimal quotes only for the prefixes it lists (3.7476/3.7477, 2.3065/2.3066, 3.7241, 2.4179, 3.6628, 2.5828/2.5829). Both scans therefore miss:
  - 4- to 7-decimal rounded quotes of 3.67360595202419;
  - 4- to 6-decimal rounded quotes of 2.58174158877406;
  - 4- or 5-decimal quotes of 3.66435028046302.

  A `git grep` of the five cuts for those rounded forms, run on 21 October, finds one quote:
  - **What:** `2.5817`, the 4-decimal floor of cell (0.0025, 0.005].
  - **Where:** line 67 of `D1_V2_2_RECEIPTS.txt` on main, and its two hardening copies.
  - **Effect on the posts:** none.
    - That line is already in the `scan2.py` rows, through its `2.5829` quote.
    - O7 does not move that cell.
    - A floor rounded down is conservative.
- **Both scans:**
  - miss quotes with 3 or fewer decimals, and records that rely on the band without quoting the number;
  - look only at the default-branch trees plus hardening, at the commits above.
- **Independence:** the same model family wrote and checked these scripts. That adds no independence.
- **Identifiers:** pack and object names are DG-MIGRATION content hashes, not private-storage identifiers. The scripts embed no Drive or Dropbox identifiers.
