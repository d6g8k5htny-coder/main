# CL-REG-001-v1.0 — FROZEN-BODY EXTRACTION-RULE REGISTER (bytes-verified)

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-15 · CLASS: REG — provenance register
STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE (record-integrity only; no theorem content)
SOURCE TREE: K3_SIDE24_LB/ from `09152026OKComputer_Project_Gap_Closure.zip`
  (zip sha256 a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b, per CURRENT_STATE_DELTA)
COMPANION: `CL-REG-001_EXTRACTION_RULE_REGISTER_2026-09-15.json` (machine-readable, same content; in the bundle zip)
FALSIFICATION: any entry whose stated rule, applied to the carrier's bytes, fails to reproduce the claimed hash.

## Why this register exists

The corpus convention says a frozen-carrier body hash is "the LF-joined lines strictly between the unique exact
lines BEGIN_FROZEN_BODY / END_FROZEN_BODY … exactly one terminal LF." Applied to the tree, that sentence
resolves to **three different byte sequences** depending on trimming, and the tree also carries four other
conventions (`sepbody`, `whole`, body-excluding-hash-line, before-hashline). Every claimed hash DOES verify —
under its own carrier's rule — but a verifier applying one rule uniformly fails roughly a third of the claims.
Example (same day, adjacent lanes): `PERC_DECAY.md` verifies only under `marker_raw` (7409 B);
`D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` only under `marker_strip_LF` (10752 B); `B4LOC_DAMLINE.md` under either
(13503 B, no leading/trailing blank lines).

This register records, for every body-hash claim found in the tree, the rule that reproduces it — so the
next verifier (human, model, or gate) does not have to guess, and so the divergence can be normalized at the
next issuance.

## Rules (exact definitions)

| id | definition |
|---|---|
| `whole` | sha256 of the entire file |
| `sepbody` | all bytes before the first `"\n---\n\n"` |
| `marker_raw` | bytes strictly between `BEGIN_FROZEN_BODY\n` and `END_FROZEN_BODY`, **no trimming** |
| `marker_rstrip_LF` | marker segment, trailing newlines stripped, exactly one LF appended |
| `marker_strip_LF` | marker segment, leading **and** trailing newlines stripped, one LF appended |
| `excl_bodyhash_line` | file with the single line containing "Body hash of this document" removed |
| `body_excl_hashline` | file with the last line containing a 64-hex digest and "hash"/"sha" removed |
| `before_hashline` | all bytes before the file's own `body-sha256:` line (H5 rung certificates) |

## Register (25 claims, de-duplicated; STAGE_E/mut_D1 byte-copies omitted)

| carrier | claimed body hash | claim source | reproducing rule |
|---|---|---|---|
| UPPER2D/B1_taxonomy/B1_TAXONOMY.md | 7d7ddbec6e63… | B2B_BOUNDARY.md | marker_rstrip_LF |
| UPPER2D/H4_closure/H4_CLOSURE.md | a67d50b9a24c… | D1_ASSEMBLY_v1_2_ADDENDUM.md | marker_rstrip_LF |
| UPPER2D/B4LOC_damline/B4LOC_DAMLINE.md | 0d5c1b328437… | D1 v2.2 REGISTER_NOTE | marker_raw (= rstrip = strip; no blank edges) |
| UPPER2D/C2_B_classes/C2_B_CLASSES.md | 2aba099d630d… | D1_V2_RECEIPTS.txt | marker_raw |
| UPPER2D/D2_branch_control/D2_BRANCH_CONTROL.md | 9e95648acbe5… | D1_ASSEMBLY_v1_2_ADDENDUM.md | marker_raw |
| UPPER2D/D3_percolation/D3_PERCOLATION.md | 8e7fef6b4fb9… | D1_ASSEMBLY_v1_2_ADDENDUM.md | marker_raw |
| UPPER2D/D3_percolation/D3_REMOTE_AMENDMENT.md | 63d91cdd6364… | D1_ASSEMBLY_v2_1.md | marker_raw |
| UPPER2D/D3_percolation/D3_REMOTE_AMENDMENT_v2.md | 6796deea4bfd… | D1_ASSEMBLY_v2_2.md | marker_raw |
| UPPER2D/D3_percolation/PERC_DECAY.md | 5137a811e6be… | D1 v2.2 REGISTER_NOTE / FREEZE_PERC_DECAY.txt | **marker_raw only** |
| UPPER2D/H4_JC_repair/H4_JC_EVENT_LEVEL.md | 42ee88da984b… | D1_ASSEMBLY_v2_1.md | marker_raw |
| UPPER2D/D1_assembly/D1_ASSEMBLY.md (v1.0) | 006b8a7d0114… | D1_V2_1_RECEIPTS.txt | marker_strip_LF |
| UPPER2D/D1_assembly/D1_ASSEMBLY_v1_1_ADDENDUM.md | 634338b43a00… | D1_V2_1_RECEIPTS.txt | marker_strip_LF |
| UPPER2D/D1_assembly/D1_ASSEMBLY_v1_2_ADDENDUM.md | a7e1958c0d37… | D1_V2_1_RECEIPTS.txt | marker_strip_LF |
| UPPER2D/D1_assembly/D1_ASSEMBLY_v2_0.md | 86882dca5d52… | D1_V2_1_RECEIPTS.txt | marker_strip_LF |
| UPPER2D/D1_assembly/D1_ASSEMBLY_v2_2.md | 490ad6b2f141… | REGISTER_NOTE / PACKAGE_VERIFICATION | marker_strip_LF (verified this pass) |
| UPPER2D/D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md | c1d5e95d97e0… | its own freeze record | **marker_strip_LF only** |
| UPPER2D/BRANCH_dir/BRANCH_DIR.md | 8821c8facc7e… | FREEZE_PERC_DECAY.txt | whole |
| UPPER2D/C1_alpha_intensity/C1_ALPHA_INTENSITY.md | ca2c02b81a05… | H4_JC_repair/FREEZE.txt | whole |
| UPPER2D/H5_closure/H5_CLOSURE.md | 465c97c0553e… | D1_V2_RECEIPTS.txt | sepbody |
| UPPER2D/H5_closure/H5_CLOSURE_TIGHTENING_2026-09-15.md | 7fefa17b7483… | D1_ASSEMBLY_v2_1.md | sepbody |
| UPPER2D/STAGE_E/REVIEW_proof.md | 922f5668d733… | itself | sepbody |
| UPPER2D/H3_closure/H3_BAND_FLOOR.md | 281477c39412… | H3 MANIFEST / RETURN_06 ADDENDUM-1 | excl_bodyhash_line |
| UPPER2D/H3_closure/H3_BAND_CEIL.md | cfe8a3a49e32… | H3 MANIFEST / LPW v4 review | excl_bodyhash_line |
| W2_symbolic/W2_DERIVATION.md | 26e2d1dd3b36… | W2_ADDENDUM.md | body_excl_hashline |
| UPPER2D/H5_closure/H5_RUNG2_2026-09-15.md | 91a34d83688b… | its own `body-sha256:` line | before_hashline |
| UPPER2D/H5_closure/H5_RUNG3_2026-09-15.md | f4c3414fe8b0… | its own `body-sha256:` line | before_hashline |
| UPPER2D/LEAD_INTENSITY_DERIVATION.md | 729644f3521d… | H4_closure/FREEZE.txt ("own convention") | **UNRESOLVED** — none of 9 candidate rules reproduces it |

## Findings

1. **Seven distinct reproducing rules are live in one tree** (the seventh, `before_hashline`, surfaced when gate
   v4 consumed the rung certificates), four of them variants of the single sentence the corpus states as its
   convention. Verification outcome currently depends on which agent authored the freeze record.
2. **One claim is unreproducible** from its stated convention: `LEAD_INTENSITY_DERIVATION.md` (H4 FREEZE says
   "body (own convention)"). The hash is not wrong; the rule is undocumented. Owner: H4 lane.
3. `B4LOC_DAMLINE.md` is the only marker-rule carrier whose body is rule-agnostic, because its author left no
   blank line inside the markers. That is the safest authoring pattern and could simply be mandated.

## Recommended disposition (for the operator; nothing here is self-executing)

- At the next issuance of any freeze record: state the rule by **id from this table**, not by prose.
- Prefer `marker_raw` with the authoring rule "no blank lines adjacent to the markers" (makes all three marker
  variants coincide, as B4LOC already does).
- Add a gate ck to the D1 assembly gate family: for every consumed body hash, the (carrier, rule-id) pair must
  reproduce — the same discipline the gate already applies to whole-file hashes. (Gate v4 already applies
  rule-ids per consumed body: marker_raw / marker_rstrip_LF / marker_strip_LF / sepbody / excl_bodyhash_line /
  before_hashline.)

Verification transcript: `extraction_register_build.txt` (in the bundle zip; regenerable from the JSON companion).
