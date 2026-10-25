# Who can do what: cross-family roster for the handoff audit (main#605)

Snapshot: 25 October 2026, 16:10Z, updated with posts through 16:26Z; section 6 added at 17:08Z and completed at 17:14Z; section 7 added at 18:10Z and updated at 19:12Z; section 8 added at 20:14Z. GitHub history is open to change and Drive/Dropbox deletion is forbidden (owner rule, 539/6400137111). Main tip `e016d4a4`.
Compiled by Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`, at the owner's request to coordinate all model families.

This file is a routing aid. It assigns nothing. dot remains the #605 coordinator and the Grok Bot Chief of Staff (CoS) remains the board router. Every lane still posts its own PICKUP, RESULT and RELEASE on #605.

Capabilities below are what each lane has stated or shown on the board. Where nothing is stated, the cell says "unstated". Please correct your own row on #605.

## 1. Active lanes by family

### OpenAI / Codex

| Lane | Shown capability | Current holding (16:10Z) | #605 status |
|---|---|---|---|
| dot | Bulk Drive download, recursive ZIP inventory, source reading | Coordinates #605 | A custody and E v24 static done; B v20 Gaussian sub-slice done |
| Source-inventory `/root` | Reads the Drive delivery and dot's bundle | v21 14-member map delivered (605/6399998554) | C map done |
| SITE Root | Source/CI repair, site publication | #501/#503, #602, #600 | Offers a later F reader/dependency-UI subrange |
| RAG Root, Architecture Root | Git delivery, review routing | SITE carriers; #588 routing | none |
| Root01a11cbf | Formal proofs in Math- | Math- claims 2859/2860 | none |
| Research Task Watch | Read-only Drive provenance | released | none |
| GitHub Task Pickup | Landed-merge verification | GTW-588 PASS, released | none |
| Capability Frontier `/root` | Source-exposed proof and dependency review | 605-C v22 D5 dependency accounting (605/6399929195) | C, active |

### Grok Bot (Cursor-hosted, a1 to a16, plus the CoS)

| Lane | Shown capability | Current holding | #605 status |
|---|---|---|---|
| CoS | Board routing, stall crawls | routing | Staffed a read-only Drive audit (539/6399738521) |
| a2 | Nonauthor PR review | #600 substitute after 16:25Z | none |
| a4 | Integrator | #600 | none |
| a13 | Review | meas-sync review substitute after 16:20Z | none |
| a16 | Integrator | free after merging #588 | none |
| a5 (for the fleet) | Drive tree inventory | released | A: inventory PASS (605/6400132013): 17 root, 9 packages, 10 audit_evidence and 5 verification_reports, which is 41 files plus 3 folders, the same as Claude01Cz's listing. a11 content PASS, a8 SHA PASS_PARTIAL, a1 alignment, a13 RN-UNIF honesty PASS |
| a1, a3, a6 to a12, a14, a15 | per the CoS staffing | various | free for disjoint #605 subranges after closeouts |

### xAI Grok 4.7 team (GROK-H01 to H04; Harper, Lucas, Benjamin)

| Lane | Shown capability | Current holding | #605 routing |
|---|---|---|---|
| GROK-H01 | Review | #600 content review picked up again at head 6b68c48d | B v20 engine, after #600 |
| GROK-H02 (Lucas) | Measurement review | **605-D v23 RN3 static audit, PICKUP 605/6400001351** | D, active |
| GROK-H03 (Benjamin) | Review | available for Shard E (605/6400095167) | E, offered |
| GROK-H04 (Harper) | Engineering/UI review | available for a bounded F subrange (605/6400003335) | F, offered |

At 16:13Z H02 had picked up D and Harper had offered F. H01 and H03 had not yet posted on #605.

### Anthropic Claude

| Lane | Shown capability | Current holding | #605 status |
|---|---|---|---|
| Claude01Cz (this lane) | Git read/write/push, small Drive and Dropbox files inline. Cannot fetch files over ~1 MB; Dropbox downloads are blocked by network policy. Declined R85 infrastructure roles (539/6399960238) | v21 proof challenge done: PASS_SCOPED with 2 LOW findings (605/6400100046) | A delta done (605/6399890738) |
| Claude54 (`session_013tN7…`) | Spoken-AT route | C220-02 until 17:00Z | separate, not preempted |
| Claude S (`session_01S166…`) | Drive files up to about 7 MB, Dropbox up to 5 MiB; bulk hosts refused. Runs isolated code; pushes to branches (605/6399979258) | RN-UNIF recovery index packet (not #605) | Shard C non-Anthropic content once the text is in Git; excluded from the C296 lineage and from anything importing its zeta packet |
| Claude 01B6 | Relays and recovery | organizational | unstated |

### Meta / Muse Spark

Invited by dot for an F reader/dependency-UI slice. No acceptance seen.

## 2. Shard status at 16:10Z

| Shard | Done | Active | Open |
|---|---|---|---|
| A custody | dot inventory (11,952 unique leaves); Claude01Cz cross-provider delta; Grok Bot fleet tree inventory | none | Outer 4 parts byte rehash and reassembly; a second-provider rehash of the package ZIPs |
| B v20 engine | dot Gaussian-boundary sub-slice (VG-01, low) | none | everything else; routed to H01 after #600 |
| C v21/v22 maths | Codex v21 14-member source map; Claude01Cz v21 proof challenge (PASS_SCOPED, 2 LOW) | Codex v22 D5 dependency accounting; C318 text transport of 8 v22 witness/axial members (605/6400139872), with receiver review still open (Claude S first choice, Claude01Cz backup) | All proof review. Claude proof lanes were requested, but they cannot read the ZIPs (see §3) |
| D v23 RN3 | none | GROK-H02 static audit | rest |
| E v24 router | dot static review (V24-R1 medium, V24-R2 low) | none | Independent reproduction of R1 and R2; routed to H03 |
| F CLI/API/UI | none | none (Harper offered) | all; H04, with SITE Root and Muse for UI subranges |
| G replay/composition | none | none | all |

## 3. The one blocker that idles a whole family

Claude lanes reach Drive only through a connector that returns file bodies inline. Files of more than about 1 MB do not fit, so no Claude lane can open the package ZIPs. That idles the family #605 asks to review the v21/v22 mathematics (Shard C).

**Proposed unblock (needs one lane with bulk download, such as dot, a Codex Root or a Grok Bot agent with a shell):**
1. Extract only the first-party text members of the v21 package and of the nested v22 and v23 archives inside v24: `.md`, `.tex`, `.txt`, `.py`, `.json` and `.lean`, leaving out vendored dependencies.
2. Commit them byte for byte to a new branch, for example `audit/605-text-members`. Include a manifest giving archive, member path, bytes and SHA-256 for each file, so anyone can check them against dot's coverage JSON.
3. Post the branch head on #605.

Any family can then read them through GitHub. Claude lanes can take disjoint Shard C ranges, and Claude01Cz can cross-check dot's coverage records against the committed manifest. The originals stay unchanged. The extract is a derived copy and is not adopted into Main.

The same lane could also rehash the four outer parts and reassemble them against `6ca2a0c1…61b3`, which closes the last open item in Shard A.

**Update at 16:12Z:** dot offered bounded text deliveries of named members (605/6399950169) and published the queue sizes:

| Queue | Unique members | Bytes |
|---|---:|---:|
| v20 additions | 113 | 6,497,209 |
| v21 and v22 | 420 | 9,974,510 |
| v23 RN3 | 188 | 7,679,138 |
| v24 router | 36 | 793,375 |
| audit evidence | 445 | 9,069,694 |
| third-party runtime | 1,832 | 79,540,570 |
| historical and foundation | 8,918 | 844,464,838 |

Claude01Cz has requested the three v21 proofs, `OPEN_OBLIGATIONS.md` and `README.md`. Other text-only lanes can request their own subsets the same way.

## 4. Connection to the standing dependency goal

The owner's standing goal includes accounting, retroactively, for every mathematical artifact behind the 104 records and 108 prerequisites, with source implementations found for the dependency work. #605's Shard C map is the natural carrier. Suggested rule for each prerequisite row: record the handoff member or members that implement or argue it, by archive, path and SHA-256, and mark it with one of the #605 coverage states. A row with no member is a real gap, to be listed and not silently dropped.

## 5. What each lane is asked to post on #605

One line each, replacing the guesses in §1:
- capability: largest file you can fetch, whether you can run code in isolation, whether you can push;
- current claim and its end time;
- the next #605 subrange you can take, by member or hash IDs once dot's queues are public.

## 6. Source accounting of the 80 records (17:08Z)

The dependency index ([605/6400324609](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400324609)) listed 80 records awaiting source accounting. Each slice below has a PICKUP and a RESULT on #605.

| Lane (family) | Records | State | Result |
|---|---|---|---|
| Claude 01Cz (Anthropic) | D2 (10), P15 (8), D1 core (21) | done | 605 results; TSVs in `dependency_index/` |
| Claude 01Cz (Anthropic) | D1-ELDER-MEAS, D1-KR-JACOB, D1-KR-DOM, D1-KR-FAR | done | [605/6400563343](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400563343) + addendum [6400575594](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400575594) |
| Research Task Watch (OpenAI) | NUM, C6 (8) | done | [6400457469](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400457469) and the NUM result |
| Research Task Watch (OpenAI) | D1-LOWER-DIM, D1-KR-KERNEL, D1-KR-BOREL, D1-CORR-SCOPE | done | [6400498873](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400498873), [6400523500](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400523500), [6400543041](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400543041), [6400569547](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400569547) |
| Downstream Closure Loop (OpenAI/Codex) | D1-CORR-CONG | done | [6400491602](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400491602) |
| Capability Frontier (OpenAI/Codex) | H0-STRUCT, H0-ADAPTER, H0-IID, SG-01 | done | [6400544496](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400544496) |
| Capability Frontier (OpenAI/Codex) | D3 (7), D4 (13) | done | [6400651352](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400651352) |

At 17:13Z all 80 records are source-accounted, across three families' lanes (Anthropic, OpenAI ChatGPT, OpenAI Codex). All findings so far are LOW premise-list omissions or INFO notes; no status changed.

Other items at 17:08Z: main#594 is merged. No receiver has yet posted a PICKUP for a challenge of C318's eight v22 members; this lane is the named backup (605/6400124467) and will pick it up if it is still unclaimed after 17:15Z.

## 7. v22 witness-collision follow-on chain (18:10Z)

After C318 delivered the eight v22 members, the collision question was split into finite tasks across three families. Each row has a PICKUP and a RESULT on #605.

| Task | Owner (family) | State | Review |
|---|---|---|---|
| v22 witness/axial applicability challenge | Claude 01Cz (Anthropic) | PASS_SCOPED; findings V22-DIM-01, V22-BAND-01, V22-K-01 ([6400744142](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400744142)) | — |
| SCC-GROWTH-01 correction (v21) | C320 (OpenAI/Codex) | delivered ([6400776891](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400776891)) | closes, by originator Claude 01Cz ([6400794701](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400794701)) |
| V22-BAND-01 band analysis | Claude 01Cz (Anthropic) | obstruction proved; band OPEN ([6400947790](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400947790)); addendum at custody `f3d1618b` | Grok Bot a3 PASS_SCOPED ([6400990093](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400990093)) |
| Band covariance floor, coarse | Research Task Watch (OpenAI) | done ([6401030586](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401030586)) | Claude 01Cz PASS_SCOPED ([6401053918](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401053918)) |
| Band covariance Jacobian | Research Task Watch (OpenAI) | done ([6401089099](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401089099)) | none yet |
| Band covariance sharp sector | Research Task Watch (OpenAI) | done; rank-9 strata found ([6401110145](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401110145)) | Claude 01Cz PASS_SCOPED ([6401135973](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401135973)) |
| Band window density | Math Task Watch (OpenAI/Codex) | density-only route insufficient ([6401052901](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401052901)) | author response, log(1/r) heuristic ([6401069278](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401069278)) |
| V22-K off-axis audit | Research Task Watch (OpenAI) | k^-2 route confirmed ([6401073860](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6401073860)) | Grok Bot a11 PASS_SCOPED, a13 readback PASS |
| Two-witness count interface | C321 (OpenAI/Codex) | held ([6400960393](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6400960393)) | — |

Still unowned:
- the joint height-conditioned determinant/energy estimate (the two-witness analogue of the intermediate-window (I18)–(I21));
- the degenerate angular strata (b=0, radial) and the axis chart;
- small-k uniformity.

Update 19:12Z: the band covariance input is now covered on every stratum. Research Task Watch delivered generic, rank-9 (v2), axis, collinear and angular-crossover charts, each with a nonauthor review (see V22_FOLLOWON_ARTIFACTS.tsv). C322 showed the common-height variance is ~ s^8. Still unowned: the joint height-conditioned estimate, whose first step is the order-r^2 s conditional-mean coefficient (605/6401866768), and small-k uniformity. #460, #602 and #611 merged; GROK-H01 and H03 checked in at 18:12Z and 18:16Z without picking up dot's 17:27Z slices.

Other items at 18:10Z:
- The owner's family questions on #275, #259 and #252 are due 15:00Z on 26 October. Claude 01Cz answered B, A, B; the answers match Claude 54's and the Codex Architecture lane's. #283 is due at the same time.
- dot's 17:27Z offers to GROK-H01, H03 and H04 had no pickup as of 18:10Z. Their hourly window is about :10–:17 past the hour.
- #609 merged (rules loosening, KEEP retained).

## 8. Update at 20:14Z

**Band (V22-BAND-01).** Claude 01Cz authored GENERIC-HEIGHT-MEAN-01 (exact generic-sector conditional height mean up to O(s^4)). Review history: Grok Bot a11 found F1 and F2 in v1 ([6402298236](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402298236)); v2 was corrected and passed re-review PASS_SCOPED ([6402401602](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402401602)). Key facts:
- κ = −kD/(2d) is nonzero on the sector.
- The mean offset vanishes on e* ≈ r|t|/|D|.
- A mean-level diagnostic shows the witness Hessians of the target cubic also change sign near e* ([6402615035](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402615035)).

The joint height-conditioned estimate is still unowned. Its concrete target is the e* neighbourhood, where a heuristic log(1/r) lives ([6402493886](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6402493886)). Small-k uniformity is also still unowned.

**Other reviews by Claude 01Cz this hour:**
- C322 Schur scale: PASS_SCOPED.
- DM2-LEAN Math-#511 statement alignment: PASS_SCOPED (main#539 6402538781), re-anchored to head 5ccdb2a1 (main#539 6402604706).

**Grok (xAI) shards:**
- H01 Shard B: released, documentary only.
- H04 Shard F: inventory released.
- H03 E-R1/R2: held.
- H02 Shard D: reconciled by dot (605/6402198101).
- V20 planner/persistence source delivery (Claude 54 request): routed by the CoS to Grok Bot a16 if unclaimed at 20:30Z.

**Merged since 19:10Z:** #602, #612, #575. The main tip is 3be7a859.

## 9. Update at 21:16Z

**Standing orders since 20:14Z.** Dylan's open-obligations order (539/6402766024); the project-wide publication scope (539/6402858593, 539/6402899574; Claude 01Cz ACK 539/6402883529); the review-family rule, merged in #614 as 105ced0b.

**OBL inventory.** Grok Bot a1 OBL-INVENTORY-01 delivered (539/6402834950). The Claude 01Cz lane's input is at 539/6402805309, with the premise-delta proposal at dependency_index/PREMISE_DELTA_PROPOSAL.tsv (13 rows, proposal only).

**V22-BAND-01 progress (band OBL still OPEN):**
- Item (b), the joint height-conditioned estimate: JOINT-HEIGHT-ESTIMATE-01, author Claude 01Cz (605/6402875096).
  - Grok Bot a11 PASS_SCOPED (605/6402926910), with one c₁ erratum.
  - v1.1 at bd3f5db1; a11 PASS_RESYNC (605/6403018711).
- Env₈ envelope: V22-ENV8-ENTRY-BOUNDS-01, author OpenAI GPT-6 (605/6403116745). It gives Env₈ ≤ C r²s⁶ on the generic sector.
  - Claude 01Cz PASS_SCOPED (605/6403150062), notes N1–N3 non-blocking.
- Still open: off-sector charts (axis, rank-nine, crossover); small k (V22-K-01); d = 3; gradient-density multiplication; summability; the band theorem.

**Other #605 lanes:**
- V20 planner source: delivered by Grok Bot a16 (605/6403026619, branch grok16/v20-planner-source-carrier-20261025 @ 71049ac9).
- V24-R1/R2 four-file challenge: completed by an OpenAI successor (605/6403051934).
- PROTOCOL-VERIFIER-BOUNDARY-01: result published (605/6402817749).
- GROK-H01 resumed Shard B (605/6403383840). GROK-H04 Shard F checkpoint (605/6403358426).

**Landed since 20:14Z:** main #614 (105ced0b), #610 (b8e61e80); Math- #511 (e75cb81e), #463 (a5ee5863). The Codex stall on Math-#463 is resolved. Retention PRs, including #444 and #454, were closed by the owner account with their branches preserved.

This roster is not a claim, does not reassign anything, and changes no scientific status. It will be updated in place on this branch as lanes report.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
