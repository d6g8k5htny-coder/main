# Retrospective implementation-reference presence audit — 23 October 2026

**Classification:** engineering/source-identity evidence only; **NOT A RESULT**, **scientific effect NONE**, independence 0. This is a bounded derived audit of the pre-existing 313-unit retro inventory, **not** a second inventory, new scientific register, authoritative dependency graph, graph-v2 binding, or mathematical review.

## Exact immutable input cuts

- Canonical 313-unit inventory: `main:docs/retro/RETRO-MATH-BIND-01.md@5a1b152a30ebc2fcb770fa264756ef36594419ac`, Git blob `1af2fb1f8aad2e7036e3d142758f1caa5c8c61b5`. The 291 orphan-unit rows below are the source population; the **other 22 already graph-associated units are excluded** from these classifications.
- Math- referenced files: [`9f3fb6918b28e6378c034ebc19e76d432f5391cb`](https://github.com/d6g8k5htny-coder/Math-/tree/9f3fb6918b28e6378c034ebc19e76d432f5391cb); complete recursive Git tree **3216 entries**, `truncated=false`.
- Main referenced files: [`a487cfd41bfe095842d9d9f5cec26fc23877b028`](https://github.com/d6g8k5htny-coder/main/tree/a487cfd41bfe095842d9d9f5cec26fc23877b028); complete recursive Git tree **1654 entries**, `truncated=false`. This is the retro inventory's declared **source** cut, not the later inventory publication commit.
- Exact path resolution only; **no historical Git byte-body rehash**, dependency-semantic audit, test execution, proof checking, private-source access or external network fetch. A valid Git blob path does not by itself establish that the content implements the associated mathematical claim.

## Reproducible extraction and classification

1. Select only inventory Markdown rows matching ``^- `(Math-|main):([^@`]+)@([^`]+)`: (\d+) files; impl: (.*)$`` (the row counts and frozen short source SHA are carried as recorded). All **291** expected rows parse; no malformed selected row.
2. An implementation field equal to `NO IMPLEMENTATION FOUND` has zero declared reference tokens. Otherwise split on `, `, preserving all tokens. A `main:` or `Math-:` prefix explicitly changes the referenced repository; unqualified paths are resolved in the unit's own repository. Resolve each path literally in the matching **complete pinned Git tree**, and accept a reference as present only when its entry is a `blob`. No title/prose inference or branch-tip substitution.
3. **Local code-language reference:** at least one declared path equals its unit path or begins with `<unit path>/` in that same repo, and ends with `.py`, `.lean`, `.rs`, `.jl`, `.ipynb`, `.mjs`, `.js`, `.c`, `.cc`, `.cpp`, `.f90`, `.r`, or `.sh`. This is a syntactic, intentionally broad indicator; a test or self-reference qualifies, and neither execution nor proof implementation is implied.
4. **Reference-only:** declared references exist, but no code-language reference within the unit is named. These are further grouped below. **No recorded reference:** inventory says `NO IMPLEMENTATION FOUND`, not that the project lacks the software.

## Verified counts at those source cuts

| Unit/reference classification | Count | What is established |
| --- | ---: | --- |
| Orphan units with ≥1 code-language reference in their own unit | **223** | A declared local code path is listed, not verified to implement its associated theorem. |
| Orphan units with references only outside their unit | **20** | Listed workflow, test or other-repository files exist; no in-unit executable-language path is named. |
| Orphan units with `NO IMPLEMENTATION FOUND` | **48** | No implementation reference recorded in this heuristic snapshot; absence in all history is NOT proven. |
| **Total orphan-unit rows** | **291** | Exhaustive for the specified exact-row population, not all mathematical work. |
| **Referenced path mentions present as Git blobs** | **663 / 663** | All declared entries resolve to a blob at the pinned repository cuts. |
| **Distinct repository-qualified referenced paths** | **530** | Includes repeated references shared across different units; no deduced dependency identity. |

The original inventory's published **313 total units / 265 heuristic implementation links / 48 `NO IMPLEMENTATION FOUND`** remains unchanged. The 243 orphan rows with at least one reference split as 223 local-code-reference rows + 20 other-reference rows. The remaining 22 already graph-associated units were not reclassified or rechecked here. **Do not rewrite the 265 figure as '265 executable mathematical implementations verified.'**

## The 20 reference-only unit exceptions and exact Git objects

All 20 contain real references, but NONE names a code-language file in its own unit under the conservative unit-prefix rule. They are not automatically defects or missing mathematics. These lists are source-bound examples for the existing GB-8 ownership lanes, **not new accepted graph nodes**.

### workflow only (13 units)

- `Math-:claims/LANDING_CLAIMS.json` — frozen row source key `6b21169`
  - `Math-:.github/workflows/landing-claims.yml`; Git blob `0516a6df00e1c33b0b725fd5a6ec6e929f44cfc7`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/landing-claims.yml)
  - `Math-:.github/workflows/lifetime-parent-imports.yml`; Git blob `b81cd0f6c5838718b4be5d49352cfbc8c00ac04c`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/lifetime-parent-imports.yml)

- `Math-:formal/lake-manifest.json` — frozen row source key `867d9e3`
  - `Math-:.github/workflows/c297-regular-superlevel.yml`; Git blob `1201e225e25956b385cb8e68584c70c4f0e65528`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c297-regular-superlevel.yml)
  - `Math-:.github/workflows/lean-specifications.yml`; Git blob `9e20e91aaa41de808ac8e83055b1c8ef9ae6fe05`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/lean-specifications.yml)

- `Math-:formal/lakefile.toml` — frozen row source key `867d9e3`
  - `Math-:.github/workflows/c297-regular-superlevel.yml`; Git blob `1201e225e25956b385cb8e68584c70c4f0e65528`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c297-regular-superlevel.yml)
  - `Math-:.github/workflows/torus-filtration.yml`; Git blob `920bbcf0b0e58cc13a0e9e0d6debfd6e90f1cd11`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/torus-filtration.yml)

- `Math-:formal/lean-toolchain` — frozen row source key `867d9e3`
  - `Math-:.github/workflows/c297-regular-superlevel.yml`; Git blob `1201e225e25956b385cb8e68584c70c4f0e65528`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c297-regular-superlevel.yml)
  - `Math-:.github/workflows/torus-filtration.yml`; Git blob `920bbcf0b0e58cc13a0e9e0d6debfd6e90f1cd11`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/torus-filtration.yml)

- `Math-:frontiers/concave_fibre_elder_20260930` — frozen row source key `9c6a734`
  - `Math-:.github/workflows/soft-fold-limit.yml`; Git blob `839b99a47a4b9d0a79525216118717b22bad4614`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/soft-fold-limit.yml)
  - `Math-:.github/workflows/soft-rejected-pairs.yml`; Git blob `a930f5334aeb40cb7b5365ed64b21fe773afaa1e`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/soft-rejected-pairs.yml)

- `Math-:frontiers/local_elder_geometry_20260930` — frozen row source key `79f18f0`
  - `Math-:.github/workflows/c8-cap-failure-coefficient-planar.yml`; Git blob `a9c9daba3de200025f1d45da63777b89b51bf957`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c8-cap-failure-coefficient-planar.yml)
  - `Math-:.github/workflows/c8-elder-failure-coefficient.yml`; Git blob `81e3a0b02e10606e9de0c301ccb1d271fa13832c`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c8-elder-failure-coefficient.yml)

- `Math-:frontiers/README.md` — frozen row source key `2f67704`
  - `Math-:.github/workflows/price-budget.yml`; Git blob `da03eca7c72db4c605a294efa5ca13266551a5b7`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/price-budget.yml)

- `Math-:frontiers/short_bar_endpoint_law_20261001` — frozen row source key `ccc7057`
  - `Math-:.github/workflows/ordinary-short-bar-occurrence-all-d.yml`; Git blob `1687c868ebcae1c568f54138148448d3a237e40d`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/ordinary-short-bar-occurrence-all-d.yml)
  - `Math-:.github/workflows/ordinary-short-bar-occurrence.yml`; Git blob `77c172cd6ce86a219f7243343d110166b839f96c`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/ordinary-short-bar-occurrence.yml)

- `Math-:frontiers/window_multiplicity_laws_20260928` — frozen row source key `0507e3a`
  - `Math-:.github/workflows/local-cluster-tightness.yml`; Git blob `072a5a4dec7439aefbb25e81bf8dc5ac5d2874c0`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/local-cluster-tightness.yml)
  - `Math-:.github/workflows/remote-mixed-inverse.yml`; Git blob `ddcb90dceaed6fa7cf5ec43692862e15e0b90633`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/remote-mixed-inverse.yml)

- `Math-:reviews/c6_fourier_completion_20260929` — frozen row source key `cf49c28`
  - `Math-:.github/workflows/c6-witness-collision-reconciliation.yml`; Git blob `a176b69a2a0440eecf5eb4292ffe4ebe293e98e8`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c6-witness-collision-reconciliation.yml)

- `Math-:reviews/candidates_pending_20260928` — frozen row source key `64a170f`
  - `Math-:.github/workflows/c6-witness-collision-reconciliation.yml`; Git blob `a176b69a2a0440eecf5eb4292ffe4ebe293e98e8`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c6-witness-collision-reconciliation.yml)

- `Math-:reviews/d5_remote_collision_grok_20260928` — frozen row source key `9c7278f`
  - `Math-:.github/workflows/c6-witness-collision-reconciliation.yml`; Git blob `a176b69a2a0440eecf5eb4292ffe4ebe293e98e8`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/c6-witness-collision-reconciliation.yml)

- `Math-:reviews/literature_lifetime_law_recon_20260929` — frozen row source key `1b29496`
  - `Math-:.github/workflows/d1-third-order-law.yml`; Git blob `e188a4082c06c1ed0e16bcdf7e48a4ed6510c83d`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/d1-third-order-law.yml)

### main-only external (4 units)

- `Math-:formal/GLOSSARY.md` — frozen row source key `867d9e3`
  - Cross-repo: `main:formal/manifest.json`; Git blob `0d694d7051670e954c35d6e2f55e753ceada64a2`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/formal/manifest.json)
  - Cross-repo: `main:tests/test_architecture_evidence_adapter.py`; Git blob `927274620307e6e4e983527106b3eda1e723f377`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/tests/test_architecture_evidence_adapter.py)

- `Math-:formal/manifest.json` — frozen row source key `04f98d4`
  - Cross-repo: `main:.github/workflows/formal-verification.yml`; Git blob `17ff98033c429821403e0e22f22ecfc6f051c875`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/.github/workflows/formal-verification.yml)
  - Cross-repo: `main:architecture/evidence_adapter.py`; Git blob `780b13b8b37e1fe3bcbbb35640e9080736a3379d`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/architecture/evidence_adapter.py)

- `Math-:formal/README.md` — frozen row source key `7168696`
  - Cross-repo: `main:formal/manifest.json`; Git blob `0d694d7051670e954c35d6e2f55e753ceada64a2`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/formal/manifest.json)
  - Cross-repo: `main:tests/test_architecture_evidence_adapter.py`; Git blob `927274620307e6e4e983527106b3eda1e723f377`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/tests/test_architecture_evidence_adapter.py)

- `Math-:formal/SCOPE.md` — frozen row source key `7168696`
  - Cross-repo: `main:formal/UniversalLaw/Side24/Endpoints.lean`; Git blob `adccc6c2d5dd987c785198b292c0ab2da6602c26`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/formal/UniversalLaw/Side24/Endpoints.lean)
  - Cross-repo: `main:formal/manifest.json`; Git blob `0d694d7051670e954c35d6e2f55e753ceada64a2`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/formal/manifest.json)

### workflow / external-test combination (3 units)

- `Math-:reviews/downstream_boundary_20260925` — frozen row source key `6e4085a`
  - `Math-:.github/workflows/contact-small-gap.yml`; Git blob `9cde8e049501fe5ea98efdcf6d8bf9a24d2d7e4b`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/contact-small-gap.yml)
  - Cross-repo: `main:tests/test_museum_data.py`; Git blob `95a89e3f5675c8f432be3bc1f0ce725471d4705c`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/tests/test_museum_data.py)

- `Math-:reviews/proof_dependencies_20261008` — frozen row source key `bf93325`
  - `Math-:.github/workflows/downstream-gate.yml`; Git blob `f05bbec845a260e2448d76d0bf33c9461b562db9`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/.github/workflows/downstream-gate.yml)
  - `Math-:tests/test_proof_slice_check.py`; Git blob `fc774e8477f5f960276d45737b0e42677675a3b1`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/test_proof_slice_check.py)

- `Math-:reviews/retrofit_20261006` — frozen row source key `7676d9c`
  - Cross-repo: `main:tests/test_reader_links.py`; Git blob `ef90a3b941447dc9f939577b8bc7c7fd3b5145dc`; [pinned file](https://github.com/d6g8k5htny-coder/main/blob/a487cfd41bfe095842d9d9f5cec26fc23877b028/tests/test_reader_links.py)
  - `Math-:tests/test_retrofit_record_check.py`; Git blob `b02c230ec2ad312436d2697e7288171469e9d8d0`; [pinned file](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/tests/test_retrofit_record_check.py)

## Interpretation and handoff

- **223 local-code-reference units:** follow existing source manifests, frozen output/test receipts and same-object SHA-256 checks to establish actual derivation→implementation→result evidence. A co-located file or self-reference is only a discovery lead until that happens. Scientific claims, formal acceptance, and review remain separate.
- **13 workflow-only units:** inspect the frozen workflow commands/consumed files to find whether they execute another source unit or merely gate formatting, navigation or status; do NOT register the workflow itself as the theorem implementation.
- **4 main-only and 3 mixed-reference units:** distinguish cross-repository tests/Lean/formal manifests from an actual algorithm implementation; pin the second repo and exact code/result separately if later source evidence supports it.
- **48 reference-empty units:** continue the already assigned CoS/GB-8 lanes' historical-source search. `NO IMPLEMENTATION FOUND` denotes a limited prior search, **not proof of nonexistence**; preserve withheld/private-source boundaries.
- Do not create graph nodes, infer historical acceptance, promote `lemma_closed`, change math proofs/status/REGISTRIES or use a quota-affected Codex review as PASS. This report complements the separate [`#565` semantics addendum](https://github.com/d6g8k5htny-coder/main/pull/565) without editing that author-owned PR or its underlying historical counts.

## Observed data / execution limits

- This audit parsed one frozen Markdown inventory and two complete native Git trees. It performed **no project tests**, CI reruns, code execution, artifact bytes SHA-256 rehash, all-branch-history search, private Drive/Dropbox retrieval, or semantic proof-to-program review.
- The observed code-language suffixes and unit directory rules are not a universal definition of implementability. Paths may point at integration tests, verification scripts or stubs rather than a research computation. Existing author/reviewer verdicts and source records remain authoritative for those matters.
- All 663 reference path hits mean 'existed at these exact cuts,' not 'the cited historical unit commit held the same bytes' or 'the executable reproduces the stored result.' The full Git SHA-1 object names of the 20 exceptions are provided above as addressable evidence only.

**Custody:** generated in an isolated docs-only engineering audit, not a competing master register. Actual OpenAI/GPT-6 contributor, source/coordination exposed, organizational/blind/human independence0. Author stage released after verified publication. [Owner mandate](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OWNER_MANDATE_20261015.md) · [Standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md).
