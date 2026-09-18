# REV-P15-C — technical review of P15-C

**Connected unbounded-width family with one 816-colour certificate**

> **Zero organizational independence. Nothing here moves a gate.** This record is a verdict on an object and nothing else. `independence_credit: 0`; `gate_status_after: UNCHANGED`; the `RV-P15` row's `Independence status` stays `EXTERNAL_REVIEW_OPEN` whatever the verdict below says. P15 sits on the **prize reconnaissance track, HOLD / not for submission**: it must not be merged into the q0 packages, must not enter the q0 dependency graph, and `original_prize_closed` stays false. No original prize problem is solved — the count is zero.

The object constructs an even-cycle family whose critical threshold grows without bound while a fixed 816-label certificate keeps working, and then shows the obstruction is genuinely nonempty. Every number in it reproduces exactly. One sentence in section C3 asserts the negation of what that section establishes, which is why the verdict is AMEND.

## The object

| | |
|---|---|
| Route | `RV-P15` — Review Queue technical status **READY**, next action "Bind archive and exact proof paths, then execute R1–R8 packet; split into four scoped reviews" |
| Exact object | `Prize_Research_P15_Palette_20260917/proofs/03_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md` |
| Title at source | P15-C - Connected unbounded-rank, unbounded-scalar-width family with one 816-color certificate |
| Bytes | 5,796 |
| SHA-256 | `357dd57302a2ec6c9e98f454ac847a589e45f0db579df64f6cc102545b35429d` |
| Carrier | Drive `1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6`, `Prize_Research_P15_Palette_20260917.zip`, 121,404 bytes, SHA-256 `4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98` (recomputed) |
| Obtained | **yes** |
| Author family | `openai` |
| Reviewer | `anthropic` — claude-opus-5 / Claude Code session_01Cz7WZybv8znP64SpPj6sWY (nonauthor; first contact with any P15 byte was in this session) |
| Review UTC | 2026-09-18T14:33:36Z |
| **Technical verdict** | **AMEND** |
| Independence credit | **0** |
| Gate status after | **UNCHANGED** |

## 1. Was the object obtainable at all?

Yes. The queue said otherwise, and resolving that was the first job.

The Review Queue row gives no body digest: its 'Body SHA-256' cell is the literal string 'UNRESOLVED - archive manifest required'. Resolution path actually walked: (1) tools/drive_index.py id 1mLizXdV8guLCJoJGa7iyvjDc3C1XElxr resolved the row's 'Packet or source' link to P15_REVIEW_PACKET.md; (2) tools/drive_index.py find P15 listed sixteen P15 objects including the carrier Prize_Research_P15_Palette_20260917.zip (Drive 1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6, 121404 bytes); (3) drive/source_map/Archive_Members.csv gave 149 member rows for that carrier ID, including proofs/01..04; (4) the carrier was downloaded whole through mcp__Google_Drive__download_file_content and its digest recomputed (4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98), agreeing with both the source map and drive/inventory.jsonl; (5) all 149 members were extracted and hashed: 0 members present in the zip and absent from the source map, 0 in the source map and absent from the zip, 0 byte-count or digest disagreements; (6) the archive's own MANIFEST.sha256 (16483 bytes, 148 lines) was verified line by line: 148 of 148 entries match, 0 mismatches, 0 missing, and the single file on disk not listed in it is MANIFEST.sha256 itself. So the 'archive manifest required' blocker is resolved and the object is obtainable.

**Extraction rule.** Drive files.get alt=media on carrier ID 1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6 (application/zip, 121404 bytes, sha256 4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98, recomputed by me from the decoded bytes and equal to the Carrier SHA-256 in drive/source_map/Archive_Members.csv and to the inventory entry). The MCP response is base64; decoded with base64.b64decode, written to disk, then the member read with Python zipfile.ZipFile.read on the relative path 'Prize_Research_P15_Palette_20260917/proofs/03_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md'. No re-encoding, no newline or text normalisation, no export MIME conversion - the carrier is a binary zip, so R17 section 2's revision-aware native-Doc rule does not apply. The standalone Drive mirror 12EqvrTr3r2XAQFOBFOpY_HYRULHlPM61 (06_TALAGRAND_DISCRETE .../P15-C_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md, TEXT_READING_COPY) carries the identical digest and byte count.

**Digest against the register.** The register carries no digest for this route to disagree with: RV-P15's 'Body SHA-256' cell reads 'UNRESOLVED - archive manifest required' and 'Body bytes' is blank. The values recorded here, 5796 bytes and 357dd57302a2ec6c9e98f454ac847a589e45f0db579df64f6cc102545b35429d, were recomputed from the extracted member bytes and agree with the carrier's own MANIFEST.sha256 line for proofs/03, with the Payload SHA-256 column of drive/source_map/Archive_Members.csv, and with the inventory digest of the standalone Drive mirror 12EqvrTr3r2XAQFOBFOpY_HYRULHlPM61. The register was read, not written.

## 2. Author lineage

Read from the exact archive, as the Review Queue row directs ('Author lineage to be read from exact archive'). Two members carry an explicit provider line: sources/P15_PALETTE_RECON_20260917.md (2963 bytes, sha256 cacf09a50ae8ea1212f6e3adb3280ffe7529d0e6aa4c2c8865ee5323f569cbab) opens 'Policy OP-RECON-20260916-v1.0; mode FRESH; OpenAI/ChatGPT exposed author-side work; UTC 2026-09-17', and sources/P15_TRANSVERSAL_SCOPE_DELTA.md (2440 bytes, sha256 1df148bfb7490c5124badcc7321defba1147ce362340a21f44fd626432539e37) opens 'UTC2026-09-17; OpenAI/ChatGPT; mode FRESH scope delta'. Corroborating: receipts/P14_INTAKE.json cites a /mnt/data source path, README.md says 'q0/Claude sources and scientific statuses are unchanged', and NEXT_WORK.md item 8 says 'No outside referee, Kimi, Claude or another provider has been invoked in this pass'. Author family therefore recorded as openai, which is NOT this reviewer's family.

## 3. Exposure disclosure, written before the verdict

This is the part where my prior exposure was most concrete and therefore most dangerous. docs/RESEARCH_MAP.md section 8 of this repository, which I read long before opening the archive, states the answer numerically: 'P15-C an even-cycle family with global threshold exactly m(3s-2)/(2s) growing unboundedly'. I thus knew the closed form of the critical value, the growth claim and the shape of the family before I saw a line of the proof, and an anchored reviewer re-derives the number he expects. The archive's CLAIM_REGISTRY.json sharpened it further - 'fixed816 palette, exact alpha=m(3s-2)/(2s), nonempty obstruction, unbounded rank/global width/activation' - and I read that before proofs/03. REVIEW_PACKET.md item R4 then handed me the checklist: 'exact minimality/rank, local restrictions, connectedness, independent-set bound on high blocks, matching primal/dual alpha, canonical fractional-row width, 817 required parts inside one block, failure of local-bit representation, all-s probability and empty-mass estimates, transfer-matrix trace. The large implicit family must not be labeled enumerated.' Note that R4 even pre-announces the number 817, which is the hinge of C3 - so when I checked ceil(N/(r-1)) I was checking a number I had been told twice. I had also read P15-A and P15-B in full before this part, and C leans on both. Countermeasure actually taken, since disclosure without mitigation is worthless: I recomputed the maximum good cardinality by brute-force enumeration over block-count vectors for seven (s,m) shapes rather than reading the proof's counting argument, re-derived the dual coordinate loads symbolically, and verified the transfer matrix against both a count-space enumeration and a full 2^16 original-coordinate enumeration, all with code that imports nothing from the package. Reading the object's own text closely is also what turned up the C3 defect below - which R4's checklist did NOT nominate.

## 4. Hypotheses, in the object's own terms

1. s >= 2, m >= 2, r = 2s. Blocks X_0,...,X_(2m-1) are arranged on an even cycle, b = 2m blocks, and each block has N >= r vertices.
2. Forbidden sets are exactly: (1) every r-subset contained in one block, and (2) for each adjacent pair of blocks, every union of an s-subset of one with an s-subset of the other. All have size r, hence all are minimal.
3. The good sets are characterised by their block counts: 0 <= c_i <= 2s-1 for every i, and no adjacent pair both at least s. (C1).
4. Each local restriction is the uniform family |U_i| < r, for which the scalar weights 1/r are a P14 sandwich of width exactly 1, giving local demand 408 per block.
5. Palettes: even blocks receive labels 0..407 and odd blocks 408..815, so every crossing witness - which by construction meets one even and one odd block - has empty palette intersection and (PS) of P15-B holds. K = 816.
6. (C2) is consumed from P15-B at all original product probabilities and all prices c <= phi(p); it is not reproved here.
7. For C3: N = 816(r-1) + 1.
8. For C4: m = 2^s and p = 1/(100N) on every original coordinate, so that Np = 1/100. The union bound runs over b internal block tests and b cycle-edge tests; adjacent block selections are independent because the blocks are disjoint, and no independence is asserted among the overlapping cycle tests themselves.
9. For C5: (C8) is the weighted independent-set transfer matrix of the cycle of block states, with L = P(Bin(N,p) < s) and H = P(s <= Bin(N,p) < 2s).

## 5. Reconstructed argument

C1. A set is good iff it contains no forbidden set. Containing a type-(1) set means some block count reaches r = 2s; containing a type-(2) set means two adjacent block counts both reach s. Negating gives exactly (C1). I did not take this on trust: for s = 2, m = 2, N = 4 I enumerated all 2^16 subsets of the original ground set and confirmed that direct membership and the block-count condition agree on every one.

C2, upper bound on the maximum good cardinality. Blocks with count at least s form an independent set in the cycle on 2m vertices, so at most m of them are high. Against a baseline of s-1 in every block, each high block adds at most (2s-1) - (s-1) = s, giving M <= 2m(s-1) + ms = m(3s-2); equality is realised by 2s-1 in every even block and s-1 in every odd block, which needs N >= 2s-1. I recomputed M by exhaustive enumeration over count vectors for (s,m) = (2,2), (2,3), (3,2), (3,3), (2,4), (4,2) and (5,2): every one matches m(3s-2).

C2, the exact critical value. Uniform weights 1/r are feasible because every forbidden set has size exactly r, and their maximum over good sets is M/r, so alpha <= m(3s-2)/(2s). For the matching lower bound the object builds a good-set distribution (choose the high parity with probability one half, then choose the prescribed counts uniformly inside each block) whose per-coordinate inclusion probability is the average of (2s-1)/N and (s-1)/N, namely (3s-2)/(2N); and a forbidden-set distribution (uniform block, uniform r-subset) whose per-coordinate probability is r/(2mN). Scaling the second by M/r gives per-coordinate load M/(2mN) = (3s-2)/(2N), identical to the first. P15-A's (A3) then gives alpha >= M/r, so alpha = m(3s-2)/(2s) exactly. I verified both loads symbolically and checked the support conditions for five (s,m,N) shapes: the mu support consists only of type-(1) r-subsets, which really are forbidden, and the nu support consists of the prescribed count vectors, which really are good because the high blocks all share one parity and are therefore independent in the cycle.

C3. With N = 816(r-1) + 1, a single block's full vertex set needs ceil(N/(r-1)) = 817 good pieces, because a globally good subset of one block has fewer than r elements. 817 > 816, so that set is NOT partitionable into 816 good parts and the 816-obstruction is genuinely nonempty. (The object's sentence at this point says the opposite; see the findings.) C3's second half I also rebuilt: a set with exactly s vertices in each of two adjacent blocks has every local restriction good, since s <= 2s-1 < r, yet is globally bad, so local bad/good flags do not determine global goodness.

C4. With Np = 1/100, the factorial Markov step P(Z >= k) <= C(N,k) p^k <= (Np)^k / k! gives 100^(-2s)/(2s)! per internal test and, using independence of disjoint adjacent blocks, 100^(-2s)/(s!)^2 per edge test; the cycle has b edges and b blocks, so the union bound is b * 100^(-2s) * (1/(2s)! + 1/(s!)^2). With b = 2^(s+1) that is at most 4 * (1/5000)^s, below 1/4 for every s >= 2. I checked the exact rational inequality for s = 2 up to 199. The empty-activation side needs (1 - 1/t)^t < 1/2, which I checked for t = 2 up to 399 (the sequence increases to 1/e), giving pi0 <= 2^(-floor(b/100)); and floor(2^(s+1)/100) > 10(2s-1) for s >= 14, which I checked to s = 59 and which at s = 14 reads 327 > 270 as the object states. Since 816 < 2^10, 816^(r-1) pi0 < 1 follows.

C5. The transfer matrix [[L,H],[L,0]] weights each step by the weight of the NEW state and forbids a high-high transition, so its 2m-th power's trace sums the weights of closed walks on the cycle of block states, which is exactly mu_p(D). I verified this in two independent ways: against a direct enumeration over all block-count vectors for four shapes and five exact rational probabilities, and against a full 2^16 enumeration in ORIGINAL coordinates for (s,m,N) = (2,2,4).

The named instance: s = 14 gives m = 16384, b = 32768, r = 28, N = 816*27 + 1 = 22033, |X| = 32768 * 22033 = 721977344, and alpha = 16384*40/28 = 163840/7. All four numbers reproduce exactly.

## 6. Execution

Package companion: python3 code/verify_p15.py on Python 3.11.15 returned PASS with 107759 checks, of which the cycle group alone carries 65862, and my stdout matched the shipped results/baseline_normal.stdout key for key. The fourteen shipped fault variants all rejected; the intentional survivor returned UNDETECTED at exit 0. My own independent script then re-derived, without importing any package module: the (C1) characterisation against all 65536 original-coordinate outcomes at (s,m,N) = (2,2,4); the extremal cardinality M for seven (s,m) shapes by enumeration over count vectors; the exact critical value via primal feasibility plus matching (A3) coordinate loads for five (s,m,N) shapes, with the support conditions checked rather than assumed; the transfer matrix against both a count-space enumeration (four shapes, five probabilities) and a full 2^16 original-coordinate enumeration; the (C5) union bound as an exact rational inequality for s = 2..199 together with its 'bracket at most 2' step; (1-1/t)^t < 1/2 for t = 2..399; the (C7) threshold for s = 14..59; and the named instance's four numbers. I attempted a brute-force vertex-enumeration LP for alpha on the smallest cycle shape as a cross-check and it exceeded my enumeration budget; I report that as a skip, not as a pass, and (C4) therefore rests on the certificates rather than on an exhaustive LP.

## 7. Negative controls executed

| Control | What it would have caught | Fired? |
|---|---|---|
| C-NC0: weaken (C1)'s adjacency test from 'both at least s' to 'both at least s+1' and compare against direct membership in original coordinates. | An off-by-one in the characterisation of the good family, which would silently change every downstream count. | **FIRED** |
| C-NC1: perturb the extremal cardinality claim to M = m(3s-2) + 1 and enumerate. | A slack or overstated counting bound in (C3). | **FIRED** |
| C-NC2: scale the forbidden-set distribution above M/r and recheck the (A3) coordinate marginal. | A lower bound that was not tight, i.e. an alpha claim that could have been pushed higher or that was not actually certified at M/r. | **FIRED** |
| C-NC3: overlap the even and odd palettes by one label (0..407 against 407..814, still 816 labels). | A palette assignment whose (PS) status did not actually depend on the even/odd split. | **FIRED** |
| C-NC4: drop the high-high prohibition from the transfer matrix (entry [1][1] set to H instead of 0). | A transfer matrix that computed something other than the independent-set-weighted trace, i.e. a wrong (C8). | **FIRED** |
| C-NC5: perturb the probability from p = 1/(100N) to p = 1/N, i.e. Np = 1 instead of 1/100. | A union bound that did not really need the 1/100 factor, making the good-probability claim independent of the choice of p. | **FIRED** |
| C-NC6: push the (C7) threshold down from s >= 14 to s = 13. | A threshold stated more conservatively or more loosely than the arithmetic supports. | **FIRED** |
| C-NC7: perturb the block size from N = 816(r-1) + 1 down to N = 816(r-1). | A '+1' that was decorative rather than the exact thing that makes the 816-obstruction nonempty. | **FIRED** |

* **C-NC0** — The perturbed condition disagrees with direct membership on the 2^16 outcomes of the (s,m,N) = (2,2,4) instance. (C1) is exactly right and not merely sufficient.
* **C-NC1** — The enumerated maximum is strictly below the perturbed value, so m(3s-2) is attained and not exceeded - it is the exact maximum, which is what the dual needs.
* **C-NC2** — Any scale above M/r makes the forbidden-set coordinate load exceed the good-set load (3s-2)/(2N), so (A3) fails. The certificate sits exactly at the boundary, which is why the value is exact rather than merely bounded.
* **C-NC3** — Label 407 then lies in both palettes, so the intersection is nonempty on EVERY crossing witness and (PS) fails everywhere at once. The disjoint split is load-bearing.
* **C-NC4** — At (s,m,N) = (2,2,4), p = 1/3, the perturbed trace is 40960000/43046721 against the true 2686976/4782969. (C8)'s zero entry is what encodes the adjacency constraint.
* **C-NC5** — The s = 2 bound becomes 7/3, far above 1/4. The 100^(-2s) factor is the whole margin.
* **C-NC6** — At s = 13, floor(2^14/100) = 163, which is NOT greater than 10*(2*13-1) = 250; at s = 14, floor(2^15/100) = 327 > 270, exactly the numbers the object cites. s >= 14 is the right threshold and it is not slack by a whole step.
* **C-NC7** — ceil(816(r-1)/(r-1)) = 816, so the block's full set then DOES fit in 816 good pieces and the obstruction would be empty. The +1 is precisely what makes C3 true.

Full script and verbatim output: [`REV-P15-CONTROLS.md`](REV-P15-CONTROLS.md).

## 8. Findings by criterion

| Criterion | Severity | Finding |
|---|---|---|
| written correctness of a set-membership assertion (C3) | **MAJOR** | C3 states: 'The full set of one block requires ceil(N/(r-1))=817 good pieces. It therefore belongs to D^(816).' As written this is false and it asserts the negation of what the section establishes. Since a globally good subset of one block has fewer than r elements, a block of N = 816(r-1)+1 vertices needs 817 good parts, so its full set does NOT belong to D^(816) - which is precisely why the 816-obstruction is nonempty, as the section heading and the following sentence both say. The notation is used consistently elsewhere in the package (B1: 'G_i is a generator cover of D_i^(k_i)' with the proof concluding that all K merged parts are good; P11-A defines covering an obstruction the same way), so D^(816) is the partitionable family and the sentence is inverted. |
| exact-object correctness (C1, C2, C4, C5 and the named instance) | **INFO** | No other defect found. Every numerical and structural assertion I could reach was recomputed independently and agrees: the good-set characterisation, the extremal cardinality m(3s-2), the exact critical value m(3s-2)/(2s) with matching primal and dual certificates, the union bound below 1/4, the empty-activation estimates, the transfer matrix, and the four numbers of the s = 14 instance. |
| slack in a stated hypothesis | **INFO** | The (C5) union bound is enormously slack for the 1/4 conclusion: at s = 2 the displayed bound is 4*(1/5000)^2 = 1.6e-07, and the 'bracket is at most 2' step alone discards about a factor of seven at s = 2. This is not a defect - a slack bound is still a bound - but a reader should not infer that s >= 2 is near-tight for the good-probability claim. By contrast C-NC6 shows the s >= 14 threshold in (C7) IS tight to within one step. |
| dependency binding | **MINOR** | C4 concludes that 'the older P12 exact-empty-activation sufficient condition fails; its connected-component variant also fails because the witness system is connected'. P12 is not among the three inputs bound in INPUT_IDENTITIES.json, so the claim cannot be checked against P12's actual statement. The arithmetic that supports it - 816^(r-1) * pi0 < 1 for s >= 14 - I did verify, and the object is careful to add that this is a failure of a sufficient condition, not of a theorem. |
| unverified side remark | **MINOR** | C2's closing paragraph asserts that the normalized resource row cover is also large, with tau_* = 2mN/r. It is explicitly flagged 'not used in proving (C2)', and I did not verify it; it routes through P14-B's normalized-profile formalism rather than anything proved in this file. |
| consumed result | **INFO** | (C2) is not proved here; it is P15-B applied to this family. The application is legitimate - each local restriction is the uniform family |U_i| < r, whose scalar weights 1/r give a width-1 sandwich and hence local demand 408, and the even/odd palettes satisfy (PS) because every crossing witness meets exactly one even and one odd block - but a reader consuming (C2) consumes P15-B and, through it, P14-A. |

* *written correctness of a set-membership assertion (C3)* — evidence: ceil(22033/27) = 817 > 816, recomputed independently. Section title 'C3. Actual816-obstruction is nonempty'; next sentence 'The theorem's obstruction is not empty merely because a coarse macro-coloring exists.' Suggested amendment: 'It therefore does NOT belong to D^(816).'
* *exact-object correctness (C1, C2, C4, C5 and the named instance)* — evidence: 82 of 82 re-derivations OK and 21 of 21 negative controls fired in my control run.
* *slack in a stated hypothesis* — evidence: Exact rational evaluation of the displayed bound for s = 2..199.
* *dependency binding* — evidence: INPUT_IDENTITIES.json binds only P11_ATTACHED_GRAPH.md, P14_01 and P14_02.
* *unverified side remark* — evidence: proofs/03, end of C2; inputs/P14_02_FRACTIONAL_RESOURCE_COVER.md sections 1 and 4.
* *consumed result* — evidence: Re-derived; the width-1 sandwich check is immediate since a(U) = |U|/r and goodness is |U| < r.

## 9. Unresolved dependencies

Every object below is something this part rests on that this review did **not** verify.

* P15-B, proofs/02_PALETTE_SEPARATED_LOCALIZATION.md, sha256 9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3 - reviewed separately in REV-P15-B, which is itself a zero-independence record; (C2) is entirely its consequence.
* P14-A, inputs/P14_01_SCALAR_SANDWICH_THEOREM.md, sha256 b3c7ea04ee7970f6fdc148bf71268d00b1482ea098343dccb5905f238532db05 - read but not reviewed; the 408 local demand per block comes from it.
* P12 and its exact-empty-activation sufficient condition, with its connected-component variant - not in the bound inputs, so C4's closing claim about that condition is unverifiable here.
* P14-B's normalized resource row cover, for the tau_* = 2mN/r remark - not verified; explicitly not used by (C2).
* The exact critical value for the named s = 14 instance is certified symbolically, not by enumeration: the 721977344-coordinate family is not expanded, by the object's own design and by mine. My alpha verification covers five small shapes plus the symbolic marginal identity, not that instance directly.
* The prize-track HOLD status and the bounded literature reconnaissance recorded in sources/ - taken as read.

## 10. Verdicts

**Exact-object correctness verdict: `AMEND`.** From the register's own R17 status set and nothing else.

**Scope / dependency verdict.** Scope: a single explicitly constructed even-cycle family of block threshold constraints, parametrised by s >= 2, m >= 2 and N >= r = 2s. Everything proved is about that family and nothing else; it is an existence example, not a theorem about downsets. Within that scope every assertion I could reach is correct, and the one exception is editorial rather than structural: the inverted set-membership sentence in C3, which is why this verdict is AMEND rather than PASS_TECHNICAL. The amendment is a single clause - 'does NOT belong to D^(816)' - and it strengthens nothing and weakens nothing in the surrounding argument, which already argues for exactly that. Dependencies: (C2) is P15-B applied here, so consuming C imports B and, through B, P14-A; all three are author-side. The example's force - a connected witness system with unboundedly large critical width yet a fixed 816-label certificate - is therefore only as strong as P15-B and P14-A, neither of which this record certifies. The 721977344-coordinate instance is specified by formula and is not enumerated by the author or by me, and my exact-LP cross-check for alpha exceeded budget even on the smallest cycle shape, so the critical value rests on the primal/dual certificates rather than on brute force. PRIZE RECONNAISSANCE TRACK, HOLD, NOT FOR SUBMISSION. This object sits on the independent prize-reconnaissance track (Drive 01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 - PRIZE PROBLEM RECONNAISSANCE - INDEPENDENT TRACK). It must not be merged into the q0 packages and must not enter the q0 dependency graph. original_prize_closed stays false; the archive's own CLAIM_REGISTRY.json records prize_closed false, external_reviews 0, historical_novelty UNESTABLISHED for this claim, and CURRENT_STATE.md records 0 formal prover runs. Nothing in this record is progress toward any prize claim, and a technical verdict here is not novelty, not priority, and not admission to any q0 surface.

**Reviewer authorship and exposure.** Nonauthor. First contact with any P15 byte was in this session. Exposure is disclosed in full in section 3 above and it is not small.

**Organizational independence: 0.** Zero, and zero is the most this situation licenses. Two separate reasons, neither of which is the other. (1) This reviewing session is Anthropic-family and the governing instruction for this task fixes the credit at zero for every record it writes; tools/reviews_check.py enforces that for an anthropic reviewer_family. (2) Independently of provider, the OP-PROT-012 section 5 independent-eyes predicate is not met, and could not be met by a review: item (c) requires no access to the author's derivation before the reviewer's own result freeze, and item (e) requires source and result hashes frozen before comparison. I read the author's full derivation first - that is what reading an object means - and I froze nothing beforehand. R17 section 4 is also explicit that a different provider ALONE establishes nothing when the reviewer reused the same reasoning, and my reconstruction followed the object's own argument line. The reviewer family here happens to differ from the author family (anthropic reviewing openai), and that difference still buys nothing: it is a necessary condition for organizational independence, not a sufficient one. The RV-P15 row's Independence status column reads EXTERNAL_REVIEW_OPEN and this record leaves it exactly there.

## 11. What this record does not establish

This record does not establish that the family described in C1 is new, interesting to anyone outside this programme, or an obstruction to anything published; it establishes that the family's stated invariants are computed correctly. It does not establish (C2): that inequality is P15-B applied to this family, and P15-B in turn rests on P14-A's local cover, which no record in this directory certifies. It does not establish anything about the 721977344-coordinate instance by enumeration - neither the author nor I expanded it, and my brute-force LP cross-check for the critical value exceeded budget even on the smallest cycle shape, so alpha stands on the matching primal and dual certificates alone. It does not establish that P12's exact-empty-activation condition says what C4 reports, because P12 is not in the archive. It does not establish the tau_* = 2mN/r side remark, which I did not check and which the object itself marks as unused. It does not turn the AMEND verdict into a defect in the construction: the inverted sentence in C3 is one clause and the mathematics around it is sound. Above all it does not bear on the unrestricted discrete-convexity conjecture; an example with unbounded critical width and a bounded palette certificate constrains what a general theorem could look like, and constrains nothing else. No original prize problem is solved; the count is zero and stays zero. No organizational independence is created, no obligation or premise anywhere in the programme changes state, and nothing here may enter the q0 dependency graph or be composed with the 2D upper or 3D lifetime tracks.

## 12. Reproducing this review

```
python3 tools/drive_index.py sha 357dd57302a2ec6c   # resolves this body to P15-C / proofs/03
MCP mcp__Google_Drive__download_file_content fileId=1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6 -> base64 -d -> sha256 4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98
python3 -c "import zipfile,hashlib; b=zipfile.ZipFile('carrier.zip').read('Prize_Research_P15_Palette_20260917/proofs/03_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md'); print(len(b), hashlib.sha256(b).hexdigest())"   # -> 5796 357dd57302a2ec6c9e98f454ac847a589e45f0db579df64f6cc102545b35429d
python3 code/verify_p15.py            # PASS, cycle group 65862 checks
python3 -c "from math import ceil; N=816*27+1; print(N, ceil(N/27))"   # -> 22033 817, so the block full set needs 817 good parts
python3 -c "print(32768*22033)"      # -> 721977344
python3 -c "from fractions import Fraction as F; print(F(16384*40,28))"  # -> 163840/7
reviews/records/REV-P15-CONTROLS.md  # my independent control script (sha256 b0fdf24d617e4b5e440b1c1707684166d4a5dc9ca1ccc7d1e929c95b131037ba) and its verbatim output; see the [P15-C] lines, eight of which are negative controls
python3 tools/reviews_check.py
```

## Notes

Naming: reviews/SCHEMA.md requires review_id to match /^REV-.../ and to equal the filename stem, while this task's ownership line named the pattern reviews/records/RV-P15*. The two cannot both be satisfied. I followed the schema, which the checker enforces and which the records already in this directory (REV-OPS-R17-001, REV-RN3-FARZONE-20260918) also follow, and kept the route key RV-P15 in the route_key field where it belongs. Concurrency observation, not a finding of mine: a second workflow is editing this repository, and governance/protocols/OP-PROT-012.md changed on disk while this review was in progress. I did not touch it.
