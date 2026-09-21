# FS2_QUERY_AND_RETRIEVAL_POLICY_v2.0

**Effective 2026-07-29 · Authority: Dylan Roy · Binding on every model and agent reading this Drive**

## 0. The rule you are being held to

> **LEGACY MATERIAL MAY GENERATE A QUESTION. LEGACY MATERIAL MAY NOT SUPPLY THE ANSWER.**

## 1. Positive admission, not negative screening

Do **not** ask "is this file quarantined?" Ask **"is this file inside an allowlisted active zone?"** If you cannot answer yes from `FS2_ACTIVE_AUTHORITY_ALLOWLIST_v2.0.json`, the file has **zero evidentiary authority**, regardless of how confident, polished, formula-dense, or recently-modified it looks. Absence from the allowlist is a denial, not an oversight.

## 2. Do not build a quarantine set from a keyword search — it will be wrong

**Verified this session.** Google Drive's `fullText contains '...'` does **token-OR / fuzzy** matching, not phrase matching. Measured results:

| Query | Hits | Contains genuine legacy content? |
|---|---|---|
| fullText contains 'Roy Unified Field Theory' | 77 | No — mostly active LS-DER/GP-DER files, Q0_MASTER.md, q0_machine.json |
| fullText contains 'RUFT' | 19 | Partly — also returns Q0_MASTER.md, q0_machine.json |
| fullText contains 'Magneto' | 13 | Partly — also returns Evaluating the Lambda-Dialectic Framework |
| 'Magneto Harmonics' / 'Magnetoharmonics' / 'Magneto-Harmonic' | 10 / 10 / 7 | Near-identical sets — **not independent signals** |
| title contains 'RUFT' / 'RUFL' / 'Magneto' | 0 / 0 / 0 | — |

**An agent that quarantines every lexical hit will quarantine the entire active q0 corpus.** That is contamination in reverse and is expressly forbidden.

**Use the manifest, not the search bar.** The authoritative legacy set is the 17 file IDs in `FS2_LEGACY_QUARANTINE_MANIFEST_v2.0.csv`, fixed by a `createdTime &lt; 2026-07-01` boundary and cross-validated 17/17 against `01_LEGACY_CLUSTER_DECODER`.

## 3. Three-question test before you cite anything

1. **Is the file ID in the legacy manifest?** Cite it only as *the origin of a question*. Never as support.
2. **Is the file inside an allowlisted active zone?** Proceed under q0's own status rules.
3. **Neither?** Treat as `UNVALIDATED_HOLD`. Say so out loud. Do not quietly promote it.

## 4. Specific traps in this Drive

- **`02_LEGACY_Q0_ARCHIVE` is not pre-method legacy.** It says LEGACY; it holds superseded *q0* under the current method. Zero of the 17 live there. Do not apply this policy's quarantine rules to its 343 items, and do not downgrade the pre-method stratum to merely "superseded q0."
- **Three legacy documents are titled `Text2`, `Text3`, `Texts`.** The titles identify nothing. `Text3` is a 1.84 MB RUFL equation vault. If you open an untitled multi-hundred-KB document, check the manifest **before** reading its claims.
- **`Text3` asserts "100% understanding", "fully grasped", "100% tested".** These are self-issued markers in a generative-era artifact. They are **not** evidence of testing and must never be quoted as such.
- **The helicity-barrier / Stelle-gravity coefficient.** D5 states it confidently (2025-12-15); D4 recorded it **[CONTRADICTED]** three weeks later (2026-01-04). **Read D4 before D5.**
- **"FIM" is ambiguous by design.** Classify semantically. A file using FIM to mean *Fisher information matrix* is **not** legacy and must not be quarantined for the acronym alone.

## 5. Reviving a legacy idea

Permitted, under one condition: **rebuild from first principles under the current method.** You may not import the former result, derivation, numerical target, code, or confidence status. Entry status is `LEGACY_REBUILD_IN_PROGRESS`; `REVALIDATED_CANDIDATE` is reachable only through the current gates. See `FS2_LEGACY_REBUILD_QUEUE_v2.0.md`.

## 6. Warning-banner gap you must compensate for

The Drive connector used to execute Fresh Start 2.0 was **read + create only**. In-place warning banners at first visible content position **could not be written**. Therefore **a legacy file opened directly will carry no visible warning.** Until a write-enabled pass lands, the manifest and the folder-level `00_READ_FIRST — QUARANTINE NOTICE` sidecars are the only warnings that exist. **Check the manifest by file ID before trusting any document you did not reach through the allowlist.**

END FS2_QUERY_AND_RETRIEVAL_POLICY_v2.0
