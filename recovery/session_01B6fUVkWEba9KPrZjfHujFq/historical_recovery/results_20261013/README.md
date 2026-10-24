# Historical recovery results, 13 October 2026 (session 01B6)

This is the search that the session's retention README listed as NOT_RUN. It ran on 13 October 2026, from 22:07Z to 23:45Z.

- **Scientific effect:** NONE.
- **Status changes:** none. Nothing here promotes, demotes, closes or reopens any theorem, lemma, register row or claim.

## What was done

1. **Search.** Nine read-only search agents covered all 155 gaps of the 28 September inventory (`../gaps.json`), in the groups listed in `method/groups.json`, under `method/RULES.md` and `method/AGENT_BRIEF.md`. They searched the provenance-indexed corpus, GitHub, and live Drive and Dropbox, all read-only.
   - **The corpus:** every blob of every ref and object in the project mirrors, the DG-MIGRATION-20260919 custody store and the recovered Dropbox texts.
   - **Gap coverage:** the original script's groups left out G101 and G103. A ninth agent covered them.
2. **Verification.** Nine adversarial verifiers (`method/VERIFY_BRIEF.md`) re-opened every fragment whose stated confidence was EXACT CARRIER, BYTE-IDENTICAL COPY, VERIFIED TRANSCRIPTION or STRONG HISTORICAL REFERENCE. They checked each quote verbatim, recomputed the SHA-256, set the chronology, and searched for later amendments, rejections or supersessions.
   - **Result:** 272 verdicts. 228 CONFIRMED, 44 DOWNGRADED, 0 REFUTED.
3. **Merge.** `method/merge.py` builds one row per gap.
   - **Gap class:** the search lane's class, changed only where a verifier explicitly reclassified the gap: G14, G46, G47, G60, G98 and G136. Each change is noted in the matrix.
4. **Synthesis.** A writer agent drafted the five files below (`method/SYNTH_BRIEF.md`). They were then screened for private data with `method/screen.py` and read by this session before publication.

## Files

| file | content |
| --- | --- |
| `RECOVERY_MATRIX.md` | One row per gap G01–G155: status on 28 September, class A–J, priority, best verified evidence with locator, what is missing, next action, and post-snapshot changes. |
| `TOP_CANDIDATES.md` | 18 verified recoveries ranked by expected impact, with locator, SHA-256, verdict, caveats and reintegration needs. |
| `DEPENDENCY_GRAPH.md` | Trees for the D1 parent, D5, D0 RN-UNIF, the SIDE24 3D package, P0.1/P0.2 and the lower bound. |
| `CUSTODY_CANDIDATES.md` | 185 CONFIRMED fragments at EXACT CARRIER, BYTE-IDENTICAL COPY or VERIFIED TRANSCRIPTION level, each with where it lives. |
| `FINAL_ASSESSMENT.md` | The four shares, the blocker most likely already solved, the blocker that appears open, the five artifacts most worth locating, and the limits. |
| `assessment_counts.json` | The counts behind the shares. |
| `EXCLUDED_RAW_IDENTITIES.json` | Byte counts and SHA-256 of the raw search and verify JSON and of `matrix.json`. These are not published, because they carry Drive and Dropbox identifiers and local paths. |

## Shares (gap class after verification, as of the 28 September list)

| share | classes | gaps |
| --- | --- | --- |
| recovery or custody problem | B, F, G, H, I | 38 / 155 |
| partially solved before | C | 73 / 155 |
| open, or reconstructable only | A, J | 31 / 155 |
| superseded or refuted | D, E | 13 / 155 |

No gap is class B. No case was found where a complete solution once existed and its carrier is now missing.

## Limits

- **Search and verification.** Language-model agents did both. The verifiers are the same provider as the searchers, and agreement between models is not evidence of correctness.
- **Self-audit.** Some audited records were written by this session, for example the 27 September upload review, PART3/PART4.
- **Unexamined items.**
  - Two large Dropbox zips added on 28 September were not examined: this environment's network policy refused the Dropbox content-download host. See `FINAL_ASSESSMENT.md`.
  - Live Drive zip contents were not examined.
- **Labels.** Every reconstruction is labelled "RECONSTRUCTED CANDIDATE — NOT ORIGINAL CARRIER" and is not counted as recovered.
- **Locators.** Drive file IDs and Dropbox IDs are withheld. A Drive-only object is named by its custody SHA-256 where the custody store holds it. The custody store lives on `chatgpt/drive-github-hardening-20260919`.
