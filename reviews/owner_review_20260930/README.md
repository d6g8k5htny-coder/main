# Dylan Roy's review desk: the pilot and refinement

**Reviewer attribution: Dylan Roy — delegated AI review.**  
**Actual performer:** OpenAI / Codex, agent `/root/c36_delegated_review_design`.  
**Basis:** source-bound digest of existing C34/C35 reviews and deliveries, prepared 30 September 2026.  
**Dylan's personal reading:** PENDING; no scoped response is recorded here.  
**Independent human review:** not performed by this work. **Organizational-independence credit:** zero.

Dylan authorized AI review on his behalf and asked to find that work under his
name so he can inspect and correct it later. This desk implements that request
under the [delegation rule](../../governance/OP-WORKFLOW-20260930.md#delegated-owner-review).
It does not relabel the people or agents who performed earlier reviews. No new
proof acceptance, experiment replay or human sign-off is claimed by this digest.
Authorized work can continue while personal reading is pending.

**Current decision:** keep the pilot and refinement available as qualified
exploratory evidence; advance certified approximation inputs and manuscript
assembly. The computed observations do not yet confirm the continuum lifetime
coefficient. The following five points are the shortest useful reading path.

| Item | Evidence and existing disposition | What to inspect or correct |
|---|---|---|
| **R1 — What the experiment measures** | [C34 protocol and implementation](https://github.com/d6g8k5htny-coder/main/blob/c78156a9a7c6c1015a88a4cf723051c0de79e008/experiments/periodic_h0/README.md): periodic ordinary superlevel H₀, finite positive bars, essential class separate, per-area lifetime-bin masses. | Does this observable match the scientific question? It is neither an independently sampled population of bars nor a typed-contact estimator. |
| **R2 — What the observations show** | [C34 pilot](https://github.com/d6g8k5htny-coder/main/blob/c78156a9a7c6c1015a88a4cf723051c0de79e008/experiments/periodic_h0/results/pilot32/RESULTS.md) uses 32 independent fields and 192 coupled evaluations. [C35 refinement](https://github.com/d6g8k5htny-coder/main/blob/490a09913518c7edf30fb9753f9b8dfc8b0752bb/experiments/periodic_h0/results/refinement8/RESULTS.md) reuses eight fields in 96 evaluations through 1024² and three filtrations. Short bins remain resolution-sensitive. | Check that numerical agreement in the larger bins is never described as a certified continuum window. Neither study is held-out confirmation. |
| **R3 — What the new bound establishes** | [C35 approximation note](https://github.com/d6g8k5htny-coder/main/blob/490a09913518c7edf30fb9753f9b8dfc8b0752bb/experiments/periodic_h0/APPROXIMATION.md) gives deterministic diagram error `ε ≤ η + h²H/4`, under its C², nodal-error and Hessian hypotheses, and endpoint-safe bin bounds. The clean upper count bound requires lower endpoint `a > 2ε`. | Separate the proved implication from its numerical inputs. The stored floating Hessian diagnostic is not an outward-enclosed certificate; convergence between computed diagrams does not bound an unknown continuum diagram. |
| **R4 — What the manuscript now supplies** | [Appendices](https://github.com/d6g8k5htny-coder/main/blob/490a09913518c7edf30fb9753f9b8dfc8b0752bb/docs/research-translation/20260930/APPENDICES.md) assemble displayed derivations against eight pinned project sources. [Literature](https://github.com/d6g8k5htny-coder/main/blob/490a09913518c7edf30fb9753f9b8dfc8b0752bb/docs/research-translation/20260930/LITERATURE.md) compares seven primary sources. Existing review covers source alignment and displayed derivations. | The full cap/ridge proof and interval implementation remain imported. This is not a renewed full-depth acceptance of the parent chain, exhaustive novelty clearance or independent human referee report. |
| **R5 — What was repaired and checked** | [PR #215](https://github.com/d6g8k5htny-coder/main/pull/215) repairs the C34 custom-configuration receipt collision and derived-float verifier sensitivity. The native [C35 technical review](https://github.com/d6g8k5htny-coder/main/pull/215#pullrequestreview-5367453973) records source exposure and independent reconstruction tests. | Tests support their stated behavior. They do not promote the program's theorem status. Preserve the original failure records and the actual model-review lineage. |

## Decisions already supported, and the next unresolved inputs

The C35 review accepted qualified publication of the deterministic approximation
derivation, exploratory refinement, execution-custody repair and source-bound
exposition. Its reviewer was a nonauthor OpenAI Codex agent with inherited
context and a shared account, giving zero organizational-independence credit.
The [C34 review](https://github.com/d6g8k5htny-coder/main/pull/214#pullrequestreview-5367091068)
is a separate earlier scoped review; the later receipt finding and repair remain
part of the history.

The [readiness assessment at C35](https://github.com/d6g8k5htny-coder/main/blob/490a09913518c7edf30fb9753f9b8dfc8b0752bb/experiments/periodic_h0/CONFIRMATION_READINESS.md)
sets the next scientific dependencies: certified finite-realization nodal and
derivative bounds, an applicable infinite-field tail budget, a numerical
remainder constant and validity range, then a predeclared held-out design with
field-level sampling. Later work may supply individual inputs; this as-of digest
does not pre-credit that work or silently enlarge the C35 verdict.

Independent human review remains an external evidence goal. Dylan can direct,
inspect and correct the delegated work without being described as an independent
referee of his own project. Owner authorization, personal reading, technical
acceptance and outside independence are separate facts.

## Exact evidence behind this digest

| Delivery | Landed source | Existing source-bound review | Immutable delivery |
|---|---|---|---|
| C34 | [`c78156a9a7c6c1015a88a4cf723051c0de79e008`](https://github.com/d6g8k5htny-coder/main/commit/c78156a9a7c6c1015a88a4cf723051c0de79e008) | [Review 5367091068](https://github.com/d6g8k5htny-coder/main/pull/214#pullrequestreview-5367091068) | [80-payload bundle](https://drive.google.com/file/d/12EJeEI00fB2iHk2r4YljiXiXuDVn4OWf/view), 3,271,566 bytes |
| C35 | [`490a09913518c7edf30fb9753f9b8dfc8b0752bb`](https://github.com/d6g8k5htny-coder/main/commit/490a09913518c7edf30fb9753f9b8dfc8b0752bb) | [Review 5367453973](https://github.com/d6g8k5htny-coder/main/pull/215#pullrequestreview-5367453973) | [93-payload bundle](https://drive.google.com/file/d/10KYjWszuv1UhpbUJNjj1WB2FCV0ngwcq/view), 1,585,474 bytes |

The source-bound reviews identify their original candidate; each delivery's
landed-byte comparison identifies the unchanged consumed files. The bundles
require existing Drive access. This page changes neither sharing nor the bundles.

* C34 ZIP SHA256: `f4a2c4eb9e53f572316ac45878a1a273e04b65403be820df76120d6ada3ba7e7`.
* C35 ZIP SHA256: `d434acd757ffe23c7e00decc2d8bb527d24f330bb677f54771392b6a3c6a3c82`.
* C34 pilot observations SHA256: `fa310ea8a259fbf58413bd49713260c7b7c73f9c3c9387426c10947e487000b4`.
* C35 refinement observations SHA256: `5ff75ce6517b236e4f6b3f03d764f9a136521001d8af2fb31e08d44781dc43a3`.
* C35 approximation note SHA256: `dae4e01b35602077442796a1150294cce3eb544f2d0c2297f8fa7aa6ab16caef`.
* C35 detailed nonauthor review SHA256, retained in its bundle: `d9426169e5bbdee90f0e7eaa6829efe5e309916c06ec66ee4081f1b090534046`.

This digest's bounded audit read the retained completion receipts, the C35
detailed review, the linked source summaries and the current review tooling. It
rehashes the local approximation and observation files to the identities above;
it does not claim a fresh remote-CI audit, full mathematical re-review, raw Drive
download or experiment replay. The prior deliveries contain those prior checks.

## Review-attribution audit limitation

The existing formal-alignment checker deliberately reports a **structure-only**
pass and requires separate retrieval and authentication of review evidence. A
read-only synthetic probe confirmed that a principal labelled `Human / Dylan Roy`
can satisfy its string-difference checks even when ignored extra metadata names
an author-side AI executor. Replacing the reviewer fields with that actual
executor correctly triggers the same-provider rejection. No synthetic record was
written to a review lane, no scientific promotion was observed, and this delivery
does not claim to have added identity authentication to that checker.
The continuation's audit evidence retains `delegation-structure-probe.py` and
`delegation-structure-probe.json`, binding the actual checker and manifest hashes
and the observed rejection diagnostic; the synthetic review exists only in memory.

Therefore this desk supplies no alignment `ACCEPTED` record. Machine reviewer
fields must name the actual performer; the owner's delegated attribution is
separate. A display name or GitHub account alone cannot establish who read a proof.

## When Dylan reads or corrects a point

A short response such as “R3: the derivative bound must cover this additional
term” is enough to identify a follow-up. Record the actual response and affected
source in the relevant PR or review, apply the correction, and link its verified
successor here when that occurs. Do not infer a response from elapsed time or
advance authorization. No response is required merely to continue authorized
work; the existing [review form](../../.github/ISSUE_TEMPLATE/review-record.yml)
can capture the eventual scoped response without another registry.
