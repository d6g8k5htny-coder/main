# Brief for an adversarial verifier (session 01B6 historical recovery)

Read <SCRATCH>/recovery/RULES.md (binding rules) and <SCRATCH>/recovery/AGENT_BRIEF.md points 4-5 (tools; read-only; no private data).

You are an ADVERSARIAL VERIFIER. For EVERY fragment in the input file whose confidence is EXACT CARRIER, BYTE-IDENTICAL COPY, VERIFIED TRANSCRIPTION or STRONG HISTORICAL REFERENCE, default to DOWNGRADED/REFUTED unless the evidence survives:
1. Re-open its locator yourself (python3 -B <SCRATCH>/corpus/cs.py --show <textfile> --range A:B, git show, gh api) and check that the quoted claim appears VERBATIM.
2. Recompute the SHA-256 of the carrier bytes where a carrier is claimed (git blob: git -C <repo> cat-file blob <sha1> | sha256sum; for corpus custody ids the id IS the sha256).
3. Establish the chronology (git log --all --find-object=<blob>, Drive modified time from provenance).
4. Search actively for LATER AMEND / REJECT / HOLD / SUPERSEDED / COUNTEREXAMPLE / ERRATUM / WITHDRAWAL / QUARANTINE / NONCONTROLLING records on the same object or its successors, including current GitHub (main and Math- heads, recent PR comments via gh api).
5. Challenge the confidence and the gap classification. J is never B; a reference is not a carrier; a transcription is not the original; a ledger restating status is not a proof; author-side/self-reviewed material is not an independent review.

Write ONE JSON file: <SCRATCH>/recovery/verify/<input key>.json = {"verdicts": [ {recovery_id, locator_reopened (bool), quote_verified_verbatim (bool), sha256_checked (value or "n/a"), confidence_after_check (one of the 7 levels or "NOT FOUND / FABRICATED"), classification_after_check (A-J), later_contradictions (with locators, or "none found" + what was searched), chronology, verdict (CONFIRMED|DOWNGRADED|REFUTED), reason} ]}. Validate that it parses.
Final reply (under 250 words): file path, counts by verdict, and the most important downgrade or refutation.
