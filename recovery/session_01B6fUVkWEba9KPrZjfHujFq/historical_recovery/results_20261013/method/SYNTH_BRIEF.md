# Brief for the synthesis writer (session 01B6 historical recovery)

Inputs (read all):
- <SCRATCH>/recovery/out/matrix.json — 155 rows, one per gap. Use classification_after_verify as the gap class (it already applies the verifiers' explicit gap-level overrides; see override_note). Fragment verdicts and confidence_after_check are inside each row.
- <SCRATCH>/recovery/out/assessment_counts.json — the ONLY numbers you may use for the final assessment. Copy them exactly.
- <SCRATCH>/recovery/results/*.json (search output: crosswalk_notes, fragments) and <SCRATCH>/recovery/verify/*.json (verifier verdicts and reasons, later_contradictions, focus_notes). Prefer the verifier's text where it differs from the search.
- <SCRATCH>/recovery/RULES.md (binding: no invention; a reconstruction is labelled "RECONSTRUCTED CANDIDATE — NOT ORIGINAL CARRIER"; agreement between models is not evidence; scientific effect NONE).

Write five Markdown files into <SCRATCH>/recovery/out/:
1. RECOVERY_MATRIX.md — one table row per gap G01..G155: gap, short name, status as of 2026-09-28, class (A–J), priority, best verified evidence (one line, with locator), what is still missing, next action. Mark rows whose best evidence was DOWNGRADED. Note post-2026-09-28 GitHub records that change a row (from best_evidence / verifier notes) in a final "since snapshot" column.
2. TOP_CANDIDATES.md — the 12–20 highest-value verified recoveries, ranked by expected mathematical impact: gap, what was found, locator + sha256, verification verdict, independence/authorship caveats, what reintegration needs.
3. DEPENDENCY_GRAPH.md — indented text trees for: Theorem A / D1 parent; D5 geometry; D0 RN-UNIF; SIDE24 3D package; P0.1/P0.2; lower bound. Tag each node: recovered / reviewed / missing carrier / reconstructed candidate / genuinely open / superseded. Use only what the matrix and verdicts support.
4. CUSTODY_CANDIDATES.md — every fragment whose confidence_after_check is EXACT CARRIER, BYTE-IDENTICAL COPY or VERIFIED TRANSCRIPTION with verdict CONFIRMED: gap, artifact, locator, sha256, where it currently lives.
5. FINAL_ASSESSMENT.md — the four shares exactly as in assessment_counts.json (with numerator/denominator), the blocker most likely already solved somewhere, the blocker that appears genuinely open, the five highest-value artifacts still worth locating (include the two large Dropbox zips the network policy blocked, described without ids), and a limits section (snapshot date 2026-09-28; corpus built 2026-09-28; search by models; verification adversarial but same-provider; no scientific status changes).

Locator rules (the files will be published on GitHub):
- Allowed: git blob/commit ids, repo@commit:path, GitHub issue/PR/comment ids, custody sha256 ids with member numbers (the custody store DG-MIGRATION-20260919 lives on main branch chatgpt/drive-github-hardening-20260919), file names and sizes.
- FORBIDDEN: Google Drive file ids or URLs, Dropbox ids ("id:...") or URLs, any /tmp or scratchpad path, e-mail addresses other than none, any personal data (birth dates, family, IDs, addresses, phones, health, finances). Replace a Drive-only locator with "Drive-only (owner's private archive); custody sha256 <x>" or "Drive-only, not in custody".
- Quotes: at most 200 characters, only copied from the verified fragments.
Keep each file compact. Final reply: the five file paths and their sizes, and any row you could not fill.
