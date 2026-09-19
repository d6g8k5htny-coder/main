# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-08-01-to-08-03 — SIDE24 RATIFICATION + P0.1 POST-RATIFICATION` (ratification chain only)

Drive lane of the SIDE24 3D ratification (80 inventory items, 71.9 MB; the 3D
track, plus the 2D P0.1 §5 post-ratification audits that share the same Drive
folder). This directory mirrors the **3D ratification chain** of
`01_SIDE24_3D_RATIFICATION_AND_RECOVERY/`: the operator record `AO48-OPR-045`,
the ratification package `AO48-AUD-044`, the reconciliations `AO48-AUD-043`
and `AO48-AUD-036`, the records `AO48-REC-034` and `AO48-REC-035`, and the
folder's `README_2026-08-01.txt`. One `_MANIFEST.jsonl`, verified by
`tools/verify_manifests.py` in CI. **Mirroring is not ratification, review,
replay or endorsement.** Nothing here is composed with the 2D tracks.

## What is here

| file | identity |
|---|---|
| `AO48-OPR-045 …` (3,397 B) | **byte-exact**: SHA-256 `e48d7c27…` and byte count equal the `drive/inventory.jsonl` row (id `1MfA94SaoYpnnAs7HYLNvowG9EHIR3Opk`) and the `operator_decisions` register row |
| `AO48-AUD-044 …` (5,245 B) | byte-exact (`1qX_KJz2KreWDy7BXNmCBesUmbkJCoPBa`) |
| `AO48-AUD-043 …` (4,818 B) | byte-exact (`1wDuGvUxFnWTYe8PEIgjo6upwzjdBMV4-`) |
| `AO48-AUD-036 …` (7,137 B) | byte-exact (`1n5AkaFD6ThBoPTw15l8Zy_8ff4h-_oXA`) |
| `AO48-REC-034 …` (7,453 B) | byte-exact (`12pnNmP7Uznn8p6km2c5r74qhxlShC3MQ`) |
| `AO48-REC-035 …` (5,137 B) | byte-exact (`1XqQOv2G8Q9lFczNqlwXTdkCL0OToVJTo`) |
| `README_2026-08-01.txt` (3,131 B) | byte-exact (`1J1DMv19INJIEnwWb5yyTMO4uowNbwbvt`) |
| `AO48-AUD-033 …` | **not stored**: two raw downloads returned 10,155 bytes (`b1f49627…`) against the inventory's 10,154 B (`895adbf1…`); recorded as `HASH_MISMATCH` in the manifest. The id is in neither delta file, so which record reflects the live file is not decidable here |

Not mirrored (indexed in `drive/inventory.jsonl`): the G.9/R2 recovery annex
(22.3 MB zip, 573 members — one carrier, two byte-identical Drive copies), the
four LOCAL-ONLY checkpoint zips, the V3.4 audit-replay kit, the
`SIDE24_AUDIT_EVIDENCE_2026-08-01.zip` the folder README calls "TRUNCATED /
CORRUPT … DO NOT USE IT", and the P0.1 (2D) sub-lane.

## The status words, verbatim

* `AO48-OPR-045` — `AUTHOR: Claude Opus 4.8 (Anthropic) — AO48, relaying the
  operator`; `AUTHORITY: OPERATOR (Dylan Roy). Verbatim statement, this
  session: "I I Dylan Roy the human of this the project sign. You may record it
  as the operator decision along with any supporting changes."` By that
  authorization "the operative ratification text of AO48-AUD-044 §D is adopted
  as signed": "I, the human owner of this work, ratify the closure of RP-C and
  RP-S per the KIMI-AUD-006/006b audit chain and accept the theorem sup(1−p_r)
  ≤ Cr³ and the statement ν₃,₂₄(ℓ) = c₃,₂₄ℓ^(−1/3)(1+o(1)) at their stated
  scopes, with the carried dependencies recorded in AO48-AUD-044 §C." Scope:
  "the compact-positive-mark elder-selection estimate sup_{t,b,κ}(1 −
  p_r(t,b,κ)) ≤ C r³, uniformly on compact (b,κ) subsets, for the normalized
  periodized Bargmann–Fock field on the side-24 three-torus". Carried
  dependencies "(ratified as stated, AUD-044 §C)": the frozen V3.3 eigenfloor
  tables ("consistent with, not re-derived in, the audit"), the three absent
  V3.4 diagnostic scripts ("non-blocking"), the V3.3 Palm normalizer cr² ≤ Z_r ≤
  Cr². Reopening conditions: "an exact counterexample to any audited display;
  failure of a V3.3 eigenfloor table; a landed diagnostic script contradicting a
  corroborated claim." Firewall: "this ratification concerns the SIDE24 3D track
  only. The 2D q0 program is untouched … no sealing, no release, no cross-track
  inference." Post-ratification action 1: "update the controlling theorem
  status from HOLD to RATIFIED-AT-STATED-SCOPE".
* The register (`registers/json/operator_decisions.json`, row AO48-OPR-045,
  2026-08-02T16:10:54Z): decision `RATIFIED-AT-STATED-SCOPE`, "3,397 B; SHA-256
  e48d7c27…; q0/P0.1 explicitly unchanged."
* `AO48-AUD-044` — `AUTHORITY: none — the decision below is the operator's
  alone`; `CANONICAL IMPACT: NONE until ratified`; verification ledger §B:
  "Kimi (third family): every display in the facewise draft recomputed
  correct"; "AO48 (this line, independent): the two flagship displays
  confirmed from scratch in exact arithmetic"; "AO48 verification boundary,
  stated for the record: this line did not independently re-read the envelope
  file's full text this session".
* `AO48-AUD-043`, `AO48-AUD-036`, `AO48-REC-034`, `AO48-REC-035` — each
  `AUTHORITY: none`, `CANONICAL IMPACT: NONE`; AUD-036: "HOLD unchanged";
  REC-035 on the relayed V3.4 draft: "Direction convergent; correctness NOT
  certified here."
* `README_2026-08-01.txt` (the independent auditor, the day before
  ratification): "Candidate theorem: HOLD." — superseded for status by the
  ratification record above and by the register's status fold
  (`GP-SIDE24-CTL-001`), which "Supersedes SIDE24 HOLD/OPEN status surfaces".

## What this directory does not establish

A digest match establishes identity of bytes. The ratification is the
operator's; this repository transcribes its register word and neither reviews
nor re-derives anything in the chain. The record's author is an Anthropic line
relaying the operator, and the independent review it cites is the Kimi
third-provider family relayed through that line; no independence credit is
computed or awarded here. Nothing on this track bears on the 2D upper or lower
tracks, Theorem D1's five OPEN premises, `D3-LEMMA-RN-UNIF`, P0.1, P0.2 or
Theorem B, and the standing firewall (CLAUDE.md rule 4, `FW-2D-3D-COMPOSITION`)
forbids composing it with them.
