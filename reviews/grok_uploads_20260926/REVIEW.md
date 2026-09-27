# Review of the 2026-09-26 xAI/Grok uploads

Review date: 2026-09-27. Repositories: [d6g8k5htny-coder/main](https://github.com/d6g8k5htny-coder/main) and [d6g8k5htny-coder/Math-](https://github.com/d6g8k5htny-coder/Math-).

**Reviewer.** Anthropic / Claude. The account is the same operator account that every lane uses, so this review earns **zero organizational-independence credit**. It is source-exposed and not blind: it read the earlier Claude, Codex and Copilot comments on these PRs.

**Scientific effect: NONE.** No file under `STATUS.md`, `PROOF_INDEX.md`, `claims/` or `LANDING_CLAIMS` is edited. Verdicts are technical only.

**Scope.** Every upload that identifies itself as xAI/Grok on either repository, up to the cutoff 2026-09-27T00:45Z. That covers:
- `grok/…` and `incoming/grok-…` branches and PRs;
- the Grok Heavy cycle-4 packet (Harper, Benjamin, Lucas);
- the "Grok 4.7 via Cursor" nonauthor-review lane, including the reviews that `STATUS.md` cites;
- Grok-lane PR and issue comments.

**Pins.** Findings are assessed at the pinned heads listed in each part. Many of these PRs were merged on 2026-09-27 between 17:26Z and 18:08Z, after the pins. Section 5 records which findings now sit on a default branch.

This file is the summary. The evidence is in four parts:

| Part | Surfaces | Findings (B/M/m/n) |
|---|---|---|
| [PART 1](PART1_MAIN_PACKETS_AND_MATH_DRAFTS.md) | main PRs #163, #159, #161, the cycle-2 branch and issue #165; Math- PRs #80, #81, #82; Files 1–6 public-mirror check; PR #87 reading maps | 148 (4/14/87/43), 41 PLAUSIBLE |
| [PART 2](PART2_PROVENANCE_PR85_CYCLE4.md) | How Math- `d8f5505` and `5ed3b45` reached default (PRs #64, #85); PR #85 content; Math- PR #87 cycle-4 packet; Grok comments; cross-cycle consistency | 64 (1/4/43/16), 15 PLAUSIBLE |
| [PART 3](PART3_CURSOR_REVIEW_LANE.md) | Grok 4.7 via Cursor lane: 9 landed Math- reviews; Math- PRs #52, #53, #55, #69, #72, #73, #74; main #63, #67, #116 comments; main PR #128 | 101 (0/7/51/43), 25 PLAUSIBLE |
| [PART 4](PART4_STATUS_D3_D4.md) | The Cursor reviews behind `STATUS.md` D3 (main #65) and D4 (main #76) | 20 (0/0/9/11) |

B/M/m/n means blocker/major/minor/nit. The totals overlap: Part 1 §10–§11 is deduplicated against Parts 2 and 3, and several findings share one root cause. [checks/](checks/) holds two scripts an outside reader can run.

## 1. Bottom line

- **The Grok work is mostly careful arithmetic with over-stated conclusions.** Most exact identities re-derive: closed-form conditional variances, Schur complements, congruence algebra, and coefficient and rational constants. The defects concentrate in order-of-magnitude claims, "ABSENT" and "refuted" labels, custody pointers, and verdicts phrased more strongly than their derivations support.
- **Two mathematical errors matter most, and both are D5 power-counting.**
  - The cycle-4 D5 ledger (Math- PR #87, open) says `det H_S = O(k r)` and calls Math-#58's `O(r⁶q²)` an error. On the Kac–Rice event, `det H_S = Θ(r²)` and the on-axis product is `Θ(r⁶q²)`, so #58 is right (Part 2 D1–D3, X1). The same wrong power also appears in cycle-2 and in comments on #58 and #165.
  - The D5 microdisk note (Math- PR #82, now on Math- default) gives `det H_M = −12λkP²r³/Q²`. That formula comes from mixing orders: under the exact witness solve λ cancels, and the true r³ coefficient is `3k(−24kP³+3cPQ²+dQ³)/Q²`. "det H_M = 0 identically" holds only at leading order (Part 1 MD1, MD2).
- **No `STATUS.md` acceptance rests on a Grok verdict that is wrong on its own scope.** All four "ACCEPT — scoped" rows (D2, D3, D4, D6) rest on Grok 4.7 via Cursor reviews, and each re-derives. D2 is under-supported: the cap §§2–4 import was accepted by citation only, and `LANDING_CLAIMS` still says FAIL_CLOSED (Part 3 U11-F2, U11-F3).
- **Math- `d8f5505` and `5ed3b45` were not direct pushes.** They are squash merges of PR #64 (a ChatGPT-lane erratum) and PR #85 (a Grok branch), and both complied with the ruleset. PR #64 was undrafted and merged 28 s later over an unanswered Claude AMEND. The load-bearing erratum is still outside `MANIFEST.json`, so the `verify-imports` check never reads it. The record does not show who merged either PR; the Grok lane is a strong but circumstantial attribution (Part 2 §1).
- **Several flagged items landed after the pins without their fixes.**
  - Merged with some fixes: main PRs #163, #159 and #161 (intake files and some items).
  - Merged without fixes: Math- PR #82 and Math- PR #55.
  - PR #161's body line "Does not close #116." auto-closed #116 at merge (Part 1 LB3).
  - Section 5 lists what now needs an erratum or a successor.

## 2. Surfaces reviewed

| Surface | Lane (self-ID) | Pinned head | State on 2026-09-27 | Technical verdict | Where |
|---|---|---|---|---|---|
| main PR #163 `incoming/grok-session-20260926-replay` (34 files) | xAI/Grok session | `cb774284` | Merged `46c0f69` after intake files were added | AMEND. Arithmetic mostly exact. False "ABSENT from public GitHub" premise; stale PR #64 lines; (ND) mislabelled OPEN | P1 §1 |
| main PR #159 `grok/drive-lane-map-20260926` | xAI/Grok | `39161a6` | Merged `7cf3bdb` | AMEND. Navigation only; stale rows persist | P1 §2 |
| main PR #161 LB-RATE / KIMI-THM-023 HOLD review | "xAI / Grok 4.6" | `43b3ced` | Merged `dc58979` with items 1–4 fixed | HOLD disposition correct. #116 auto-closed by the body wording | P1 §3 |
| main branch `incoming/grok-cycle2-nd-d5-sard-20260926` | xAI/Grok | `8dfde10` | Unchanged; no PR | Not landable (IDENTITY.json and the cited predicate file missing). D5 power one r weak | P1 §4, P2 §5 |
| main branch `grok/drive-missing-source-hunt-20260926`; Math- `grok/congruence-erratum-on-default-20260926`, `grok/required-math-20260926` | Grok (name) | no unique commits | Stale | Delete when unused | P1 §2, P2 §1 |
| Math- PR #80 contact-kernel substitute | Grok | `a79007e` | Open draft | AMEND. All its mathematics is exact; navigation-level | P1 §5 |
| Math- PR #81 drive-hole ledger | Grok | `b28cf25` | Open draft | AMEND | P1 §6 |
| Math- PR #82 D5 microdisk NOTE/QUARTIC | Grok | `b190a4d` | **Merged to Math- default** 17:35:56Z | **REJECT as stated.** Needs an erratum (MD1–MD4) | P1 §6 |
| Math- PR #64 erratum (ChatGPT lane) → `d8f5505` | merged by the shared account | `1d9b64ba` | Merged 2026-09-26 | Algebra correct. Merge unsound: AMEND unanswered, custody gap | P2 §1 |
| Math- PR #85 → `5ed3b45` | Grok branch | `ff36667` | Merged 2026-09-26 | AMEND (minor). Mathematics exact; custody and coverage gaps | P2 §1–2 |
| Math- PR #87 cycle-4 (Harper / Benjamin / Lucas) | "Grok + Harper + Benjamin + Lucas" | `8657130` → `0fab933` | Open | AMEND; **do not merge as is** (D5 ledger blocker) | P2 §3 |
| Math- PR #88 reading map | Grok (inferred) | `2cdf62f` | Open | Minor packaging | P2 §4 |
| 9 landed Math- reviews (PR16, PR22, PR25 ×2, PR28, PR19/21 + two-scale, D2 correction, P15) | "Provider xAI / Grok 4.7" via Cursor | Math- main | Landed 2026-09-25/26 | SOUND at stated scope, with minor defects | P3 U1–U7 |
| Math- PR #55 (review of PR53) | Grok 4.7 via Cursor | `4e188e25` | **Merged** 17:33:20Z | AMEND. Steep-strip ACCEPT arithmetic diverges (U8-F1) | P3 U8 |
| Math- PR #53, #69, #74 (pin neighbourhood / microdisk; #74 is a same-provider review) | Grok 4.7 via Cursor | — | #53 open; #69 and #74 closed unmerged | AMEND. Load-bearing microdisk steps not established | P3 U9 |
| Math- PR #52, #72, #73 | Grok 4.7 via Cursor | — | #72 and #73 merged; #52 closed | SOUND (#52 redundant) | P3 U8, U10 |
| main PR #125 shrinking-witness m=2 bound (Cursor lane; authoring model undeclared) | Cursor | `00c9d08a` | Merged `4cc4587` 18:07:56Z | Reviewed by Math- PR #72 (Grok 4.7): five ACCEPTs plus an AMEND on the η→0 transition; that review is SOUND | P3 U10 |
| main #63 D1-A–E, #67 D2 R1–R6, #116 comments; main PR #128 | Grok 4.7 via Cursor | — | PR #128 open | Arithmetic re-derives. D1-A accepts the cap step by citation (major); PR #128 §5 gap (PLAUSIBLE major) | P3 U11 |
| main #65 D3 and #76 D4 acceptance reviews | Cursor, Grok 4.7 (D4 has no self-ID line) | — | Cited by `STATUS.md` | SOUND on their listed scope | P4 |

## 3. Blockers

| ID | Surface | Problem | State now |
|---|---|---|---|
| P2 D1 | Math- PR #87 `harper/D5_OBSTRUCTION_LEDGER.md`:115 | Says `f_ss(S) = O_p(1)` and `det H_S = O(k r)`. On the Kac–Rice event, `f_ss(S) = f_ss(M) + r f_tss + … = O_p(r)` and `det H_S = Θ(r²)`, so the product is `Θ(r⁶q²)`. This was re-derived by an exact jet, by the verifier Monte Carlo, and by [checks/d5_detHS_scaling.py](checks/d5_detHS_scaling.py), which fits slopes 1.99 and 5.99. | Open PR; fix before merge |
| P1 MD2 | Math- `reviews/d5_microdisk_20260926/QUARTIC.md`:29 | `det H_M = −12λkP²r³/Q²` comes from the leading-order solve only. Under the exact solve λ cancels at r³. See [checks/pr82_quartic_det_check.py](checks/pr82_quartic_det_check.py) (exact sympy). | **On Math- default**; needs an erratum |
| P1 PK1, HY1 | main PR #163, PR #159 | The required `public-intake` check failed: RESULT.md and IDENTITY.json were missing. | Resolved post-pin (`857904a`, `f21e222`) |
| P1 CS1 | main cycle-2 `RESULT.md`:18 | Cites an `IDENTITY.json` and a `SARD_G_A1_REPAIR_PREDICATE.md` that exist on no ref, so intake would reject it. | Latent (no PR) |

## 4. Majors still worth acting on

Status labels follow each part: CONFIRMED unless marked PLAUSIBLE.

- **D5 power claims across surfaces** (P2 D2, D3, X1).
  - "#58 over-slaves det H_S" appears on five surfaces: PR #87 body and note, cycle-2 `RESULT.md`:40/:43, Math-#58, and main #165.
  - With the corrected product, the ledger's formal cone contribution is `O(k r³)`, not `O(k r²)`.
- **PR #82 NOTE** (P1 MD1, MD3, MD4, MD5).
  - "det H_M = 0 identically" holds at leading order only; the exact r³ coefficient is nonzero.
  - The det H_X coefficient is off by a factor of 2.
  - "No O(r³) lemma" rests on a crude majorant that diverges.
  - The note does not credit PR #69, PR #74 or PR #60.
- **Axis singularity** (P1 IS1–IS3).
  - It is a power law (|z_s|⁻² or |z_s|⁻⁴), not the logarithm that cycle-2 and #165 state.
  - Dropping the block on the axis cures nothing.
  - File 3's Remark 7.2 "integrable singularity" route is unavailable.
  - `κr² ≈ 1.95` has no source.
- **False "ABSENT from public GitHub"** (P1 PK3, PK4). The July-5 Files 1–6 stack is public as OCR text in `main` blob `ed0f51f1` (`Q0_MASTER.md`). That blob is in default history and at about 90 branch tips. The claim is now in the landed PR #163 packet (`CLAUDE_FILES_1_6_MAP.md`:5, `C006_ARM1C_V2_INGEST.md`:9). Main #168 (2026-09-27) adds the general rule that a search miss is not evidence of absence, but it does not correct these lines.
- **Condition (ND) presented as OPEN** (P1 BC5). The stack's own kill entry K4 and the Euler identity refute (ND) as written. "Theorem B PROVEN-MODULO repaired ND" relabels an unpublished repaired sentence (P1 CS3; P2 R4).
- **PR #161 auto-closed #116** (P1 LB3), a sibling upper-rate audit, through the closing keyword in "Does not close #116." Reopen #116 if its audit is still wanted.
- **Math- PR #64/#85 merge sequence** (P2 P1).
  - AMEND 5848593803 was never answered.
  - `ERRATUM_CONGRUENCE.md`, `ERRATUM_POINTER.md` and `CAP_PAIRING_IDENTITIES.md` sit outside `MANIFEST.json`/`verify.py`, so edits to them pass `verify-imports` silently.
  - `PROOF_INDEX.md`:32 lost its commit-pinned link.
- **STATUS D2 chain** (P3 U11-F2, U11-F3).
  - D1-A accepted MARKED-CYLINDER-CAP §§2–4 by citation, and the D2 delta then closed that import.
  - `STATUS.md` D2 "ACCEPT — scoped" contradicts `LANDING_CLAIMS` `lifetime-remainder` FAIL_CLOSED.
- **Merged Math- PR #55** (P3 U8-F1). Its steep-strip ACCEPT's stated factors integrate to about `r⁻³`. The conclusion is recoverable by another route, and the downstream `STATUS.md` D5 row stays AMEND.
- **Same-provider reviews titled "nonauthor"** (P3 U9-F1, U9-F2, U9-F13). PR #74 reviews PR #69, and both are Grok 4.7 sessions. Two load-bearing microdisk steps were not established. Both PRs are now closed unmerged.
- **main PR #128 §5** (P3 U11-F7, PLAUSIBLE). `m ≥ c k s²` fails for small L in rotated frames. The gap is in the proof, not a counterexample.

## 5. What landed after the pins, and what now needs an erratum or successor

| Landed object | Findings that now sit on a default branch | Suggested vehicle |
|---|---|---|
| main `incoming/grok-session-20260926-replay/` (PR #163) | The false "ABSENT" premise (PK3–PK5), (ND) OPEN (BC5), and the other §1 findings of Part 1. `857904a` added a supersession note but changed no packet file. | Successor intake packet or erratum (intake is add-only) |
| main `incoming/drive-lane-map-20260926/` (PR #159) | HY2–HY5, HY7, HY11 (stale OPEN/recovery rows) | Successor packet |
| main `reviews/lb_rate_thm023_landing_20260926/REVIEW.md` (PR #161) | LB9, LB12, LB17; #116 closed by keyword | Additive note; reopen #116 if wanted |
| Math- `reviews/d5_microdisk_20260926/` (PR #82) | MD1–MD4, MD7–MD9 | Erratum next to NOTE.md and QUARTIC.md |
| Math- `reviews/d5_pin_neighborhood_20260926/` (PR #55) | U8-F1 (steep-strip derivation) | Additive correction |
| Math- `imports/lifetime_parent_20260925/` (PRs #64, #85) | MANIFEST custody gap; the three items of AMEND 5848593803 | Custody PR plus successor erratum note |

Still open and fixable before merge: Math- PRs #87 (D5 ledger), #88, #80, #81, #53; main PR #128; the cycle-2 branch.

## 6. Does a `STATUS.md` acceptance rest on an unsupported Grok verdict?

| Row | Grok basis | Answer |
|---|---|---|
| D2 unrestricted lifetime remainder | main #67 5841270276 and 5841782206, on top of #63 5841570965 | Under-supported, not shown wrong. The arithmetic of R1–R6 re-derives. The cap §§2–4 import was accepted by citation, and `LANDING_CLAIMS` still says FAIL_CLOSED (P3 §3) |
| D3 SIDE24 coefficient | main #65 5841269490 | No. C1–C6 re-derive exactly, and the published 20-digit endpoints hold. The row drops the review's own PARENT-IMPORTED-OPEN label (P4 F5, minor) |
| D4 fixed-remote RN count | main #76 5841783172 (no self-ID line in the body) | No, on its listed scope. The Grok lane's later #116 comment calls the same normalizer unproved; that is unreconciled but conservative (P4 D4 F1) |
| D6 P15 full price | Math- `reviews/p15_full_price_nonauthor_20260926/REVIEW.md` | No. Custody is exact; the defects are wording only, including the ASCII `3e-2` (P3 U7) |
| D1 and D5 (AMEND/open) | #63; PROOF_INDEX / #58 | Not acceptances. The rows correctly stay open |

## 7. Recommended actions, by priority

1. Hold Math- PR #87. Correct the D5 ledger (det H_S, product, formal contribution), then post the correction on Math-#58, main #165, and the cycle-2 `RESULT.md`.
2. Add an erratum next to Math- `reviews/d5_microdisk_20260926/{NOTE,QUARTIC}.md` for MD1–MD4, citing [checks/pr82_quartic_det_check.py](checks/pr82_quartic_det_check.py).
3. Land a Math- custody PR:
   - add MANIFEST records for `ERRATUM_CONGRUENCE.md` (1782 B, sha256 `bad7ef60…2028`), `ERRATUM_POINTER.md` and `CAP_PAIRING_IDENTITIES.md`, or move the notes out of `imports/`;
   - restore a pinned link at `PROOF_INDEX.md`:32;
   - post a disposition answering AMEND 5848593803.
4. File a successor to the landed PR #163 packet. It should retract "ABSENT from public GitHub" (citing blob `ed0f51f1`), relabel (ND) and Theorem B, and replace the log axis singularity with the power law.
5. Reconcile `STATUS.md` D2 with `LANDING_CLAIMS` `lifetime-remainder`. Either bind the cap §§2–4 import to a filed review, or narrow the row.
6. Decide whether to reopen main #116.
7. Delete the three empty Grok branches. Adopt a same-provider title convention for reviews such as PR #74.

## 8. What this review did not do

- It did not accept or reject any theorem, and it moves no status.
- Provider identity is self-declared on one shared account; nothing here attests it. The record does not show who merged Math- PRs #64/#85 or the 2026-09-27 wave.
- Drive-only sources (File-3 Table 4.1, `TRANSVERSE_CONTACT_ASYMPTOTIC.md`, the cycle-2 predicate file) and arXiv Kac–Rice statements could not be read, because network egress to arXiv was blocked.
- Post-pin merges were checked for which findings survive. The landed bytes were not fully re-reviewed.
- Grok uploads after 2026-09-27T00:45Z are out of scope. None were found on either repository when the review finished.

## 9. Method

- Each surface was read in full at a pinned head, together with its cited sources. Checks were recomputed with exact arithmetic (sympy, `fractions.Fraction`) where possible.
- Monte Carlo, quadrature and float checks were run with numpy and mpmath and are labelled **NON-CERTIFYING**.
- Every finding went through adversarial verification. Blockers and majors got three verifiers with distinct lenses (reproduce, source, skeptic); minors and nits got one reproducer. Only CONFIRMED or PLAUSIBLE findings are reported, at the verifier-adjusted severity.
- A completeness critic added gap units, which found the PR #87 packet, the Files 1–6 mirror, and the Cursor-hosted lane. Parts 3 and 4 were added when the Cursor lane turned out to underpin `STATUS.md`.
- The two scripts in [checks/](checks/) were re-run independently of the workflow agents.
