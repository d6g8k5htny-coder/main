# REV-P15-B — technical review of P15-B

**Palette-separated localization of obstruction covers**

> **Zero organizational independence. Nothing here moves a gate.** This record is a verdict on an object and nothing else. `independence_credit: 0`; `gate_status_after: UNCHANGED`; the `RV-P15` row's `Independence status` stays `EXTERNAL_REVIEW_OPEN` whatever the verdict below says. P15 sits on the **prize reconnaissance track, HOLD / not for submission**: it must not be merged into the q0 packages, must not enter the q0 dependency graph, and `original_prize_closed` stays false. No original prize problem is solved — the count is zero.

The object proves a localization theorem: local obstruction covers on a block partition glue into a global one when the blocks' integer palettes have empty intersection on every crossing minimal witness. The gluing argument is setwise, short and correct. The showcase example that follows it does less work than it appears to, so the reviewer built instances that do.

## The object

| | |
|---|---|
| Route | `RV-P15` — Review Queue technical status **READY**, next action "Bind archive and exact proof paths, then execute R1–R8 packet; split into four scoped reviews" |
| Exact object | `Prize_Research_P15_Palette_20260917/proofs/02_PALETTE_SEPARATED_LOCALIZATION.md` |
| Title at source | P15-B - Palette-separated localization of obstruction covers |
| Bytes | 6,286 |
| SHA-256 | `9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3` |
| Carrier | Drive `1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6`, `Prize_Research_P15_Palette_20260917.zip`, 121,404 bytes, SHA-256 `4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98` (recomputed) |
| Obtained | **yes** |
| Author family | `openai` |
| Reviewer | `anthropic` — claude-opus-5 / Claude Code session_01Cz7WZybv8znP64SpPj6sWY (nonauthor; first contact with any P15 byte was in this session) |
| Review UTC | 2026-09-18T14:33:36Z |
| **Technical verdict** | **PASS_TECHNICAL** |
| Independence credit | **0** |
| Gate status after | **UNCHANGED** |

## 1. Was the object obtainable at all?

Yes. The queue said otherwise, and resolving that was the first job.

The Review Queue row gives no body digest: its 'Body SHA-256' cell is the literal string 'UNRESOLVED - archive manifest required'. Resolution path actually walked: (1) tools/drive_index.py id 1mLizXdV8guLCJoJGa7iyvjDc3C1XElxr resolved the row's 'Packet or source' link to P15_REVIEW_PACKET.md; (2) tools/drive_index.py find P15 listed sixteen P15 objects including the carrier Prize_Research_P15_Palette_20260917.zip (Drive 1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6, 121404 bytes); (3) drive/source_map/Archive_Members.csv gave 149 member rows for that carrier ID, including proofs/01..04; (4) the carrier was downloaded whole through mcp__Google_Drive__download_file_content and its digest recomputed (4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98), agreeing with both the source map and drive/inventory.jsonl; (5) all 149 members were extracted and hashed: 0 members present in the zip and absent from the source map, 0 in the source map and absent from the zip, 0 byte-count or digest disagreements; (6) the archive's own MANIFEST.sha256 (16483 bytes, 148 lines) was verified line by line: 148 of 148 entries match, 0 mismatches, 0 missing, and the single file on disk not listed in it is MANIFEST.sha256 itself. So the 'archive manifest required' blocker is resolved and the object is obtainable.

**Extraction rule.** Drive files.get alt=media on carrier ID 1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6 (application/zip, 121404 bytes, sha256 4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98, recomputed by me from the decoded bytes and equal to the Carrier SHA-256 in drive/source_map/Archive_Members.csv and to the inventory entry). The MCP response is base64; decoded with base64.b64decode, written to disk, then the member read with Python zipfile.ZipFile.read on the relative path 'Prize_Research_P15_Palette_20260917/proofs/02_PALETTE_SEPARATED_LOCALIZATION.md'. No re-encoding, no newline or text normalisation, no export MIME conversion - the carrier is a binary zip, so R17 section 2's revision-aware native-Doc rule does not apply. The standalone Drive mirror 19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j (06_TALAGRAND_DISCRETE .../P15-B_PALETTE_SEPARATED_LOCALIZATION.md, TEXT_READING_COPY) carries the identical digest and byte count.

**Digest against the register.** There is no digest in the register to disagree with: the RV-P15 row records 'UNRESOLVED - archive manifest required' in place of a 'Body SHA-256' and leaves 'Body bytes' blank. The digest supplied here for this part, 6286 bytes and 9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3, was recomputed by me from the extracted member bytes and agrees with three independent places: the carrier's own MANIFEST.sha256 line for proofs/02, the Payload SHA-256 column of drive/source_map/Archive_Members.csv for that member, and the inventory digest of the standalone Drive mirror 19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j. Nothing in registers/ was edited.

## 2. Author lineage

Read from the exact archive, as the Review Queue row directs ('Author lineage to be read from exact archive'). Two members carry an explicit provider line: sources/P15_PALETTE_RECON_20260917.md (2963 bytes, sha256 cacf09a50ae8ea1212f6e3adb3280ffe7529d0e6aa4c2c8865ee5323f569cbab) opens 'Policy OP-RECON-20260916-v1.0; mode FRESH; OpenAI/ChatGPT exposed author-side work; UTC 2026-09-17', and sources/P15_TRANSVERSAL_SCOPE_DELTA.md (2440 bytes, sha256 1df148bfb7490c5124badcc7321defba1147ce362340a21f44fd626432539e37) opens 'UTC2026-09-17; OpenAI/ChatGPT; mode FRESH scope delta'. Corroborating: receipts/P14_INTAKE.json cites a /mnt/data source path, README.md says 'q0/Claude sources and scientific statuses are unchanged', and NEXT_WORK.md item 8 says 'No outside referee, Kimi, Claude or another provider has been invoked in this pass'. Author family therefore recorded as openai, which is NOT this reviewer's family.

## 3. Exposure disclosure, written before the verdict

Heavy and specific to this part. docs/RESEARCH_MAP.md section 8, which I read days-equivalent earlier in this same session's corpus work, already told me that 'P15-B gives a setwise fixed-label palette-localization certificate', so I began with the conclusion in hand. The archive's CLAIM_REGISTRY.json restates it more sharply - 'Actual local compatible covers transfer under integer-palette empty-intersection conditions on all cross minimal witnesses' - and I read that before the proof. The decisive exposure, though, is REVIEW_PACKET.md item R2, which I also read first: 'every local family must be the actual restriction. Check the full intersection of the palettes touched by EVERY cross minimal witness. Pairwise intersections need not be empty. Prove union-cover containment setwise even with singleton obstructions, empty local parts and zero prices. Check global-good probability <= product local-good probabilities, not the reverse or equality.' Every one of my four controls for this part attacks a hinge that sentence names. They are therefore adversarial in form but not independent in origin. I had also run the shipped verify_p15.py fault set, which includes cross_common_color, local_demand, witness_omission and reverse_probability - again, the same hinges. Beyond the package I had read R17 section 4 and 6, OP-PROT-012 section 5 and 11, the repository README's prize-track paragraph, and docs/OPEN_PROBLEMS.md section D. I had NOT read P10, whose read-once substitution identity B2 contrasts itself against, so that contrast is opaque to me. Bias direction: strongly toward confirmation, and toward checking only nominated hinges. What I did about it: I noticed while re-deriving that the object's own showcase example B5 has an empty cover and therefore does not exercise B1's merging step at all, and I built 239 further (PS)-satisfying instances - 182 of them with a genuinely nonempty union cover - with code that does not import the package's.

## 4. Hypotheses, in the object's own terms

1. D is a decreasing family on a finite X containing the empty set. Unlike P15-A and P14-A, singletons are NOT required to be good here; B1 is stated for any downset containing the empty set.
2. H is the family of inclusion-minimal forbidden sets of D.
3. X is partitioned into nonempty disjoint blocks X_1,...,X_b, and D_i = D intersect 2^(X_i) is the ACTUAL local restriction. The object is explicit in B6 that replacing D_i by a convenient stronger condition loses (B2) unless a separate budget is proved.
4. Each block carries a nonempty integer palette P_i contained in [K] and a positive local demand k_i <= |P_i|. Palettes are integer label sets, not fractional assignments; B6 says so explicitly.
5. (PS): for EVERY minimal forbidden e meeting more than one block, the intersection of the palettes of all blocks e meets is empty. Checked on the block support of each crossing witness, not on a sample. Pairwise disjointness is sufficient but not necessary.
6. G_i is a generator cover of D_i^(k_i) with every generator contained in X_i.
7. For B2-B4: original coordinates are selected independently with arbitrary probabilities p_v; q_i = 1 - mu_p(D_i); prices satisfy c_v <= phi(p_v) with phi(t) = min(1, -log(1-t)) and phi(1) = 1; and each local G_i has cost at most phi(q_i) AT THOSE SAME original-coordinate prices.
8. The empty family costs zero and the family containing the empty generator costs one; these are distinct objects and B2's endpoint handling uses the distinction.

## 5. Reconstructed argument

B1 is the mathematical core and I rebuilt it as follows. Let U avoid every generator of G = union of the G_i. Since each G_i consists of generators inside X_i, U intersect X_i avoids G_i, so U intersect X_i lies in D_i^(k_i): it splits into at most k_i pieces each of which is a subset of X_i lying in D. Because k_i <= |P_i|, the piece labels of block i inject into P_i. Now merge, across blocks, all pieces carrying the same global label. There are at most K merged parts because the labels live in [K]. Suppose a merged part W were globally bad. Then W contains some minimal forbidden e. If e lies inside one block, e is contained in the single locally good piece of that block inside W, contradicting D_i membership. If e meets several blocks, then for each such block the vertices of e in it lie in a piece whose label is the common colour of W, so that colour belongs to P_i for every block e meets, i.e. it lies in the intersection that (PS) declares empty. Contradiction. So every merged part is good and U lies in D^(K). The argument never mentions probability, so it survives at zero-probability outcomes, and the local partitions may be chosen per U.

B3's converse I rebuilt too: if a crossing minimal e has a colour gamma common to all the palettes it touches, take U = e and give every vertex the colour gamma. Each fragment e intersect X_i is a PROPER subset of e, hence good by minimality of e, so this is a legal local colouring with one piece per block; but the gamma class is e itself, which is bad. So (PS) is exactly the uniform amalgamation property and not merely sufficient - for safety under every allowed local colouring, which is the only thing B3 claims.

B2 to B4, the hazard side. Global goodness implies every local restriction is good, since D is decreasing and D_i is the actual restriction; the local-good events live on disjoint coordinate blocks, so they are independent and their probabilities multiply. That gives mu_p(D) <= product of (1 - q_i), an inequality that crossing forbidden sets can make strict but cannot reverse. Then cost(G) <= sum of the local costs (subadditivity over the union), <= sum of phi(q_i) by hypothesis, <= sum of -log(1-q_i) since phi(t) <= -log(1-t), = -log of the product, <= -log mu_p(D) by the previous inequality and the monotonicity of -log. Capping with the empty generator when the sum exceeds one gives covercost(D^(K)) <= min(1, -log mu_p(D)). The endpoint q_i = 1 is handled by the cap, since the product is then zero and the bound is one.

B5 I checked by exhaustive computation on the 64 subsets of the six-vertex ground set: the chromatic number of the whole set is exactly 3, the palettes {0,1}, {1,2}, {0,2} have pairwise nonempty but triple-empty intersection so (PS) holds, and two colours are impossible because each block must split its forbidden internal pair, leaving an all-one-colour transversal. B4's 'K = t*k, independent of the number of blocks' special case I also checked: if the crossing support hypergraph is properly t-coloured and each macro colour gets a disjoint group of k labels, then any crossing witness is non-monochromatic, so it meets two disjoint groups and its palette intersection is empty.

## 6. Execution

Package companion: python3 code/verify_p15.py returned PASS with 107759 checks and stdout equal to the shipped results/baseline_normal.stdout; its palette group alone carries 12729 checks over 144 random exact instances and 343 uniform palette assignments. All fourteen shipped fault variants rejected, survivor UNDETECTED at exit 0. My own code, importing none of the package's: an independent downset/minimum-good-partition routine and an independent local-obstruction-generator builder. For this part it confirmed the B5 chromatic number and (PS) status by exhaustion; it then found that B5's union cover is EMPTY, so the example does not exercise B1's merging step, and it therefore built 239 further (PS)-satisfying random instances over 4 to 8 coordinates, 2 to 3 blocks and palettes of size 1 to 5, of which 182 had a nonempty union cover: zero sets needing more than K good parts escaped the cover in any of them. Cost subadditivity over the union, which (B3) uses without stating, was checked on 300 random price vectors and generator families.

## 7. Negative controls executed

| Control | What it would have caught | Fired? |
|---|---|---|
| B-NC1: delete (PS) - give blocks a common palette colour on a crossing witness and rerun the B1 construction with K equal to the number of labels actually used. | A B1 whose conclusion held for reasons other than palette separation, i.e. a decorative hypothesis. | **FIRED** |
| B-NC1b: violate the demand condition k_i <= |P_i| by asking for two local parts from a one-label palette, keeping (PS) satisfied. | An injectivity step ('map their labels injectively into P_i') that was not actually needed. | **FIRED** |
| B-NC2: flip (B2) to the reverse direction, asserting mu_p(D) >= product of (1 - q_i), across 492 random block/downset/probability instances computed exactly in rationals. | A direction error in (B2), or a hidden assumption that global and product-local good probabilities are equal. | **FIRED** |
| B-NC3: replace the ACTUAL local restriction D_i by a strictly stronger local family, as B6 warns against, and recompute the chain. | A (B2) that would survive a convenient substitution of a stronger local condition, making B6's warning unnecessary. | **FIRED** |

* **B-NC1** — With all three blocks given palette {0,1} (K = 2), seven sets that need more than two good parts avoid the union cover entirely. (PS) is the whole mechanism.
* **B-NC1b** — On a seven-vertex instance, 18 hard sets escape the union cover. The injective label map genuinely requires k_i <= |P_i|.
* **B-NC2** — Zero violations of the direction the object states, and 357 of 492 instances strictly less. The reverse assertion fails often, so (B2) is an inequality and never an identity - which is exactly what the object says and what the shipped reverse_probability fault also tests.
* **B-NC3** — On a four-coordinate instance with p = 1/2 everywhere, mu_p(D) = 1/2 while the strengthened product is 3/16 < 1/2: the inequality reverses and the chain (B3) to -log mu_p(D) is lost. B6's warning is not boilerplate.

Full script and verbatim output: [`REV-P15-CONTROLS.md`](REV-P15-CONTROLS.md).

## 8. Findings by criterion

| Criterion | Severity | Finding |
|---|---|---|
| exact-object correctness (B1, B2, B3, B5) | **INFO** | No defect found. B1's setwise merging argument, B3's converse, the (B2) direction, the (B3)/(B4) cost chain and the B5 example all reconstruct exactly as written, and the independent computations agree. |
| example strength / what B5 actually demonstrates | **MINOR** | B5 is presented as the small exact example showing overlapping palettes are genuinely useful, and it does show that. But every block in it is properly 2-colourable, so every local cover is EMPTY and the union cover is empty. B5 therefore exercises the palette arithmetic and the chromatic-number claim, and does NOT exercise B1's merging step, which is the part of the theorem a reader most wants to see demonstrated. A reader who treats B5 as evidence for the cover transfer is over-reading it. |
| unstated step | **MINOR** | The first inequality of (B3), cost_c(union of G_i) <= sum of cost_c(G_i), is used without being stated. It is true for the cost functional in use - a sum of generator price products taken over the SET union, so duplicated generators are absorbed rather than double-counted - but a reader could mistake it for a further assumption about the local covers, next to the assumption that is stated ('Assume the local G_i have costs at most phi(q_i)'). |
| dependency binding | **MINOR** | B2 distinguishes itself from 'the read-once substitution identity from P10'. P10 is not among the three inputs bound in INPUT_IDENTITIES.json, so the contrast cannot be checked against P10's actual statement. The contrast is not load-bearing for B1 to B4, so this is a readability and provenance point rather than a gap in the argument. |
| hypothesis economy | **INFO** | B1 needs strictly less than the surrounding text: it does not need singletons to be good, it does not need the palettes disjoint, and it does not need any probability. The probabilistic content is confined to B2 to B4, and it enters only through the assumption that each local cover already costs at most phi(q_i) at the original prices - which is where P14-A is consumed. |
| interface to B4's special case | **INFO** | B4's claim that a properly t-coloured crossing support hypergraph with all scalar widths at most kappa yields K = t*ceil(408 kappa) independent of the number of blocks is correct: a proper colouring makes every crossing witness non-monochromatic, so it touches two disjoint label groups and its palette intersection is empty, which is exactly (PS). |

* *exact-object correctness (B1, B2, B3, B5)* — evidence: 239 (PS)-satisfying instances with zero uncovered hard sets; B5 chromatic number 3 by exhaustion over all 64 subsets.
* *example strength / what B5 actually demonstrates* — evidence: My b1_failures run on B5 reports 0 generators and 0 gaps. I supplied 182 nonempty-cover instances separately.
* *unstated step* — evidence: Verified on 300 random instances; equality holds whenever no generator repeats across blocks, which is automatic here because generators live inside disjoint blocks.
* *dependency binding* — evidence: INPUT_IDENTITIES.json binds P11_ATTACHED_GRAPH.md, P14_01_SCALAR_SANDWICH_THEOREM.md and P14_02_FRACTIONAL_RESOURCE_COVER.md and nothing else.
* *hypothesis economy* — evidence: Reconstruction above; B4's list of admissible local certificates.
* *interface to B4's special case* — evidence: Re-derived; consistent with the 816 = 2 * 408 instantiation that P15-C uses.

## 9. Unresolved dependencies

Every object below is something this part rests on that this review did **not** verify.

* P14-A, inputs/P14_01_SCALAR_SANDWICH_THEOREM.md, 6004 bytes, sha256 b3c7ea04ee7970f6fdc148bf71268d00b1482ea098343dccb5905f238532db05 - read but NOT reviewed here. (B3)'s standing assumption that each local G_i costs at most phi(q_i) is supplied by it, so B4's headline numbers are only as good as P14-A.
* P14-B, inputs/P14_02_FRACTIONAL_RESOURCE_COVER.md, 5802 bytes, sha256 a4a23b1e3c3455dfa1da798612495bffb6c419a190bae73f03fc4bc760559ba6 - read but not reviewed; B4's 'exact fractional row-cover certificate' option routes through it.
* P10 and its read-once substitution identity - not in the bound inputs, so B2's contrast is unverifiable here.
* B4's third option, 'an already established graph/threshold compatible bound, with its exact source and hypotheses' - no such external bound was examined.
* The claim in B6 that 'the generic certificate checker may inspect every explicit witness; succinct families require their own proved structural formulas' - the checker's 16-vertex local scalar-enumeration cap is an implementation bound I confirmed by reading code/certify_palette_plan.py, but no succinct-family structural formula was reviewed here.
* The prize-track HOLD status and the reconnaissance memos' bounded literature search - taken as read.

## 10. Verdicts

**Exact-object correctness verdict: `PASS_TECHNICAL`.** From the register's own R17 status set and nothing else.

**Scope / dependency verdict.** What passed: proofs/02 as written, for the class it describes - a finite downset containing the empty set, a true coordinate partition, integer palettes satisfying (PS) on every crossing minimal witness, and local covers of the actual restrictions. The transfer (B1) is setwise and unconditional; the hazard bound (B4) is conditional on each local cover already costing at most phi(q_i) at the original prices, which is an imported P14-A fact and not proved here. What is explicitly outside: there is no claim, and this record makes none, that an arbitrary downset admits a suitable block partition, suitable palettes, or bounded local demands. B6 says a stronger-than-actual local family loses the bound, and my control B-NC3 exhibits that loss numerically. No polynomial-time discovery algorithm is asserted by the object or checked by me. Dependency posture: consuming B4's numbers imports P14-A and, on one branch, P14-B, both author-side and both carrying their own 'no independent review' banners. PRIZE RECONNAISSANCE TRACK, HOLD, NOT FOR SUBMISSION. This object sits on the independent prize-reconnaissance track (Drive 01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 - PRIZE PROBLEM RECONNAISSANCE - INDEPENDENT TRACK). It must not be merged into the q0 packages and must not enter the q0 dependency graph. original_prize_closed stays false; the archive's own CLAIM_REGISTRY.json records prize_closed false, external_reviews 0, historical_novelty UNESTABLISHED for this claim, and CURRENT_STATE.md records 0 formal prover runs. Nothing in this record is progress toward any prize claim, and a technical verdict here is not novelty, not priority, and not admission to any q0 surface.

**Reviewer authorship and exposure.** Nonauthor. First contact with any P15 byte was in this session. Exposure is disclosed in full in section 3 above and it is not small.

**Organizational independence: 0.** Zero, and zero is the most this situation licenses. Two separate reasons, neither of which is the other. (1) This reviewing session is Anthropic-family and the governing instruction for this task fixes the credit at zero for every record it writes; tools/reviews_check.py enforces that for an anthropic reviewer_family. (2) Independently of provider, the OP-PROT-012 section 5 independent-eyes predicate is not met, and could not be met by a review: item (c) requires no access to the author's derivation before the reviewer's own result freeze, and item (e) requires source and result hashes frozen before comparison. I read the author's full derivation first - that is what reading an object means - and I froze nothing beforehand. R17 section 4 is also explicit that a different provider ALONE establishes nothing when the reviewer reused the same reasoning, and my reconstruction followed the object's own argument line. The reviewer family here happens to differ from the author family (anthropic reviewing openai), and that difference still buys nothing: it is a necessary condition for organizational independence, not a sufficient one. The RV-P15 row's Independence status column reads EXTERNAL_REVIEW_OPEN and this record leaves it exactly there.

## 11. What this record does not establish

This record does not establish that any interesting family satisfies (PS). Palette separation is a hypothesis the theorem consumes, not a property it produces, and nothing here shows that a given downset can be blocked and palettised so that every crossing minimal witness has empty palette intersection - B5 exhibits one six-vertex family that can, and that is all. It does not establish P14-A, whose low-failure cover supplies the phi(q_i) local costs on which (B3) and (B4) entirely depend; without P14-A the hazard chain has no starting point, and P14-A is unreviewed author-side work. It does not establish anything about the read-once substitution identity of P10, which B2 contrasts itself against and which I could not obtain. It does not establish that B5 demonstrates the cover transfer: B5's union cover is empty, and I had to build separate instances to exercise the merging step at all. It does not establish an algorithm: no way of finding blocks, palettes or local covers is claimed or verified, and the shipped checker caps local scalar enumeration at sixteen vertices. It does not establish historical novelty for the palette-separated transfer; the archive's own registry records that as UNESTABLISHED and this record leaves the word alone. It solves no original prize problem - the count remains zero - and it is not a statement about the discrete-convexity conjecture. It creates no organizational independence and moves no gate, premise or obligation anywhere in the programme, and it must not be carried into the q0 dependency graph or composed with either the 2D upper or the 3D lifetime track.

## 12. Reproducing this review

```
python3 tools/drive_index.py sha 9b18b6e9abc90d18   # resolves this body to P15-B / proofs/02
python3 tools/drive_index.py archive P15_Palette
MCP mcp__Google_Drive__download_file_content fileId=1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6 -> base64 -d -> sha256 4bcaf6717187ec9e70f05b2ae92410f4e8ed2eab41e676e8acf2614e30b65c98
python3 -c "import zipfile,hashlib; b=zipfile.ZipFile('carrier.zip').read('Prize_Research_P15_Palette_20260917/proofs/02_PALETTE_SEPARATED_LOCALIZATION.md'); print(len(b), hashlib.sha256(b).hexdigest())"   # -> 6286 9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3
python3 code/verify_p15.py            # PASS, palette group 12729 checks
python3 code/verify_p15.py --fault cross_common_color   # REJECT cross-palette-common-color, exit 1
python3 code/verify_p15.py --fault local_demand         # REJECT local-palette-demand, exit 1
python3 code/certify_palette_plan.py inputs/PALETTE_THREE_BLOCKS.json
reviews/REV-P15-CONTROLS.md  # my independent control script (sha256 b0fdf24d617e4b5e440b1c1707684166d4a5dc9ca1ccc7d1e929c95b131037ba) and its verbatim output; see the [P15-B] lines
python3 tools/reviews_check.py
```

## Notes

Naming: reviews/SCHEMA.md requires review_id to match /^REV-.../ and to equal the filename stem, while this task's ownership line named the pattern reviews/records/RV-P15*. The two cannot both be satisfied. I followed the schema, which the checker enforces and which the records already in this directory (REV-OPS-R17-001, REV-RN3-FARZONE-20260918) also follow, and kept the route key RV-P15 in the route_key field where it belongs. Concurrency observation, not a finding of mine: a second workflow is editing this repository, and governance/protocols/OP-PROT-012.md changed on disk while this review was in progress. I did not touch it.
