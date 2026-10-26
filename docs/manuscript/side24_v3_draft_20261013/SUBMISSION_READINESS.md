# SIDE24 V3: submission readiness, novelty search and venue comparison

Status as of 26 October 2026, about 17:05Z (r19 made: supplement draft 17 cited; the r18 and draft-17 build reads passed). This is a living record. Update it when any item changes, and supersede lines rather than delete them.
Author of this record: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), for Dylan Roy. Dylan Roy — delegated AI work. Credit 0. Scientific effect NONE.
**Nothing here submits anything. Dylan has said not to submit yet.**

Marks: **[HUMAN]** needs Dylan or another person: judgement, identity, an account, or a decision. **[SUBSCRIPTION]** needs paid or institutional access. **[BROWSER]** the source is free, but this session's tools could not open it. **[AI]** an agent can do it, but only on Dylan's request. **[DONE]** done, with its record.

## 1. Where the documents stand

| Item | State | Record |
|---|---|---|
| Main text | r13, `04a3ee03`, 23 pages (documentary over r12 `d550d238`) | r12: Cursor PASS 6324514156. r13: Cursor AMEND 6325193355 (wording), applied; readback AMEND 6325247357 (change-list identity), applied; readback PASS 6325266500 |
| Script commits | AMEND applied, readback PASS (recorded separately from the r12 PASS) | main#389 6324563568, 6324618319 |
| Author decisions 1–4 | **[DONE]** defaults kept for 1–3; Supplement-only form of 4 | main#389 6324313740 |
| Supplement | draft 17, cited as [R1] at `71ee360c` (from r19), 162 pages; statement numbers unchanged since draft 12 | drafts 13–15: readbacks PASS 6290560837, 6325795760, 6326245246; draft 16 (agent 9's AMEND; G5 closed): Cursor commit 1 PASS, commit 2 AMEND applied, readback PASS 6326705347; draft 17 (build procedure only): OpenAI PASS_SCOPED_BUILD_PROCEDURE 6421820710 |
| Provider-distinct AI reads | every section of both documents has at least one (OpenAI or xAI), PASS_SCOPED | main#389 PR body; main#419 6294239616 |
| Organizational independence | 0: every AI lane posts from one account | §7 of the draft |

## 2. Remaining gaps before submission (exact)

| # | Gap | Mark | What closes it |
|---|---|---|---|
| G1 | **No human specialist has reviewed the proofs.** §7 says so. Every journal below holds the human author fully accountable for correctness; Springer's policy says that accountability "cannot be delegated to AI systems". | **[HUMAN]** | Either Dylan reads the main text line by line and can stand behind every step, or a probabilist with Kac–Rice and persistence experience reads it, ideally both. Then update §7 to say what was done. |
| G2 | **A whole read of the Supplement (162 pages).** Two whole reads have been done. The first was a same-provider fresh-context read of draft 14 (`reads/WHOLE_READ_D14_ANTHROPIC_20261021.md`; no error found, five matching gaps). The second was a **provider-distinct** whole-document consistency read of draft 15 by Grok Bot agent 9 (main#389 6326419744; G1–G4 confirmed closed; four small amendments, applied in draft 16). G5 was closed in draft 16 with a certified check. **Still open (style pass):** the notation clashes (c_2[K] in §S4 vs §S11, and others) and the abstract's wording for the computer-assisted theorems. Neither read certifies mathematical correctness. | **[DONE]** for AI whole reads; **[HUMAN]** for an expert read (G1) | |
| G3 | **The MathSciNet search has not been run.** | **[SUBSCRIPTION]** | Search MR for "persistence diagram" or "barcode" with "Gaussian field" or "random function", and for close critical-point pairs. Follow the forward citations of Adler–Taylor, Azaïs–Wschebor, Chazal–Divol, Perez and Baryshnikov (2025). |
| G4 | **The zbMATH Open search.** Run by Cursor through the zbMATH Open API (main#389 6325580607): four queries; no hit gives a near-diagonal short-lifetime density for H₀ persistence of a smooth Gaussian field in d ≥ 2. The HTML site was blocked by a bot check, and one query returned no results. | **[DONE]** at API level; MathSciNet (G3) remains **[SUBSCRIPTION]** | |
| G5 | **Bibliography additions found in this pass** (§3). | **[DONE]** in r13 (Cursor AMEND 6325193355 applied; readbacks PASS 6325266500) | Add Baryshnikov, *Brownian motions, persistent homology and chirality*, J. Appl. Comput. Topol. 9 (2025) 28. Mention it in §6 as the one-dimensional near-diagonal law (Brownian motion with drift). Its outlook names higher-dimensional Gaussian processes as future work. |
| G6 | **Publication status of arXiv-only references.** The r15 corrections from Cursor record B (Crossref, zbMATH Open; main#389 6325580607) are retained: Perez → J. Appl. Comput. Topol. 7 (2023); Hirsch–Lachièze-Rey → Ann. Appl. Probab. 36 (2026); Chenavier–Hirsch and HLO pages added. Pranav (arXiv:2109.08721) remains cited as a preprint: **no journal version was verified in the sources checked on 21 October 2026**. The dated evidence and limits are in §2.1. | **[DONE]** for the bounded bibliographic check; Pranav journal publication **[UNVERIFIED]** | Revisit if a publisher record, journal DOI or author confirmation supplies a journal version. |
| G7 | **The §7 "Remaining before submission" paragraph is partly stale.** "The treatment of its imports from the parent record" is settled by decision 4 in r12. | **[DONE]** in r13 (same read) | Rewrite that sentence to list only what remains: G2, G3/G4, provenance labels, and the journal style pass. |
| G8 | **The front matter is incomplete.** (r14: MSC 2020 codes and keywords added; the identity details remain.) `main.tex` has `\author{Dylan Roy}` only, with no address, email, affiliation (or "independent researcher") or ORCID. | **[HUMAN]** for identity details; **[AI]** can propose MSC codes (60G60, 60G15, 55N31, 60D05) | Dylan supplies what he wants printed. |
| G9 | **The repository records cited in Appendix A are paths and commits** in a GitHub repository. Referees need stable, citable access. | **[HUMAN]** (needs an account and a decision) | Archive a tagged snapshot with a DOI (for example Zenodo from GitHub) and cite the DOI in Appendix A, or make the Supplement self-contained and post it on arXiv as a companion. Decision 4 already makes the leading term rest on the Supplement alone. |
| G10 | **How the Supplement is hosted.** At 162 pages it will not be a normal journal supplement everywhere. | **[HUMAN]**: choose the venue first | Options: an arXiv companion paper cited by the main text, or a journal-hosted supplementary file. See §4. |
| G11 | **Draft-only material must be removed.** The provenance wording in the Supplement's mathematical sections was removed in draft 14 (readback PASS 6325795760). The main text's §0 and the Supplement's §S0 status sections remain. | **[AI]**, after the venue choice | A final "submission build" revision. |
| G15 | **Build procedure.** A Copilot review of main#389 (discussion_r4365148684), verified by an OpenAI/Codex lane (389/6379998598), found that `build.sh` never copied `main.pdf` to the distributed `SIDE24_V3_DRAFT_20261013.pdf`; every revision so far did the copy by hand and r17's 14 manifest entries were verified correct, but a future revision could have shipped a stale PDF with a passing manifest. | **[DONE]** in r18: `build.sh` copies the PDF, pins `SOURCE_DATE_EPOCH` so the build is byte-reproducible, and verifies (or, with `--update-sums`, rewrites) `SHA256SUMS`. The supplement's `build.sh` gets the same change in draft 17. | Read passed: OpenAI slice A (389/6421761943; build PASS, record AMEND A-F1 applied as `e9583f09`) and slice B for the supplement's draft 17 (389/6421820710). The Copilot thread is resolved. |
| G12 | **Journal style pass** (class file, bibliography style, length). | **[AI]**, after the venue choice | BibTeX "missing pages" warnings: none since r15 (BobrowskiBorman and LadghamLR fixed in r14; ChenavierHirsch and HLO in r15). |
| G13 | **arXiv posting.** A first-time math.PR submitter may need an endorsement. | **[HUMAN]** (account) **[BROWSER]** (check arXiv's current endorsement page) | Dylan's arXiv account and, if asked, an endorser. |
| G14 | **AI-use disclosure wording** must match the chosen journal's policy. §7 already discloses AI checks; the venue may want a statement about AI-generated text and proofs. | **[HUMAN]** decision, **[AI]** draft | After the venue choice, compare §7 with the journal's policy text. |

**Not a gap for submission:** main#389 does not need to merge for submission. The draft lives on its branch with full records.

## 2.1. G6 bibliographic verification addendum — 21 October 2026

Dylan Roy — delegated AI work. Actual performer: OpenAI / ChatGPT GPT-6 Astra Pro,
with three OpenAI bibliographic research agents; no human, blind or organizational
independence credit. Canonical pickup:
[main#389 6328112653](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6328112653).
This is a citation-metadata check. The manuscript's mathematical claims, existing
review evidence and instruction not to submit are unchanged.

**Compared source.** Main-text r17 at
`10e9120301ff5c5d9b1d6f3ff70a79f2f6959465`; `refs.bib` Git blob
`f7b84a63ba6376c3dcd89cc3d7fe86a2da711ee0`.
The journal corrections were already authored in r15
(`18543a204e40810276b56a3619b4377cdbba499a`) from
[Cursor record B](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6325580607)
and received the
[r15 nonauthor PASS](https://github.com/d6g8k5htny-coder/main/pull/389#issuecomment-6326222268).
This addendum verifies the three G6 references against additional sources; it
does not reattribute those corrections or extend the earlier review's scope.

| Reference | Verified bibliographic identity | Dates and persistent identifiers | Comparison with r17 |
|---|---|---|---|
| Perez | Daniel Perez, sole author. *On the persistent homology of almost surely $C^0$ stochastic processes*. **Journal of Applied and Computational Topology 7(4), 879–906 (2023)**. | Online/version of record **5 July 2023**; issue **December 2023**. Journal DOI [10.1007/s41468-023-00132-x](https://doi.org/10.1007/s41468-023-00132-x). [arXiv:2012.09459](https://arxiv.org/abs/2012.09459): v1 17 December 2020; latest listed v3 7 November 2022. | `Perez` already has the correct journal, volume, issue, pages, year and DOI, retaining the arXiv identifier. |
| Hirsch–Lachièze-Rey | Christian Hirsch and Raphaël Lachièze-Rey. *Functional central limit theorem for topological functionals of Gaussian critical points*. **The Annals of Applied Probability 36(5), 4430–4464 (2026)**. | Issue **October 2026**; Crossref print/issued date **1 October 2026**. Publisher's indexed record says first available in Project Euclid **25 September 2026**. Journal DOI [10.1214/26-AAP2333](https://doi.org/10.1214/26-AAP2333). [arXiv:2411.11429](https://arxiv.org/abs/2411.11429): v1 18 November 2024; latest listed v3 15 December 2025. | `HirschLR` already has the correct journal, volume, issue, pages, year and DOI, retaining the arXiv identifier. Crossref supplies no separate `published-online` field. |
| Pranav | Pratyush Pranav, sole author. *Topology and geometry of Gaussian random fields II: on critical points, excursion sets, and persistent homology*. **arXiv preprint (2021)**; no journal version verified in this check. | [arXiv:2109.08721v1](https://arxiv.org/abs/2109.08721v1), submitted **17 September 2021**; only v1 is listed. Preprint DOI [10.48550/arXiv.2109.08721](https://doi.org/10.48550/arXiv.2109.08721), issued through DataCite. No journal date, journal DOI, volume, issue, pages or article number is verified. | Retain `Pranav` as `@misc`, year 2021, arXiv:2109.08721. The arXiv DOI is a preprint identifier, not a journal DOI. |

**Positive journal evidence.** Perez's
[Springer article record](https://link.springer.com/article/10.1007/s41468-023-00132-x),
[volume 7 issue 4 contents](https://link.springer.com/journal/41468/volumes-and-issues/7-4)
and directly retrieved
[Crossref DOI metadata](https://api.crossref.org/works/10.1007/s41468-023-00132-x)
agree. For Hirsch–Lachièze-Rey, the
[publisher record](https://projecteuclid.org/journals/annals-of-applied-probability/volume-36/issue-5/Functional-central-limit-theorem-for-topological-functionals-of-Gaussian-critical/10.1214/26-AAP2333.full),
[Crossref DOI metadata](https://api.crossref.org/works/10.1214/26-AAP2333) and
[Aarhus University record](https://pure.au.dk/portal/en/publications/functional-central-limit-theorem-for-topological-functionals-of-g/)
agree on the final citation. Both are paginated journal articles; neither needs
an article number. The live arXiv records omit journal-reference fields even for
these two published papers, so that omission alone cannot establish nonpublication.

**Pranav evidence and bounded search.** The
[ERC-ARThUs project bibliography](https://www.arthus-erc.net/ercpub_en.html),
explicitly last updated **8 May 2026**, places the exact Part II title under
“Not resubmitted Preprints” and gives an MNRAS submission with a placeholder
volume. The
[European Commission CORDIS results](https://cordis.europa.eu/project/id/740021/results)
classify the work as a 2021 preprint, 29 pages, with the same arXiv DOI.
The current arXiv abstract record supplies the title, sole author, date and
version above. These sources support retaining the preprint citation; their
update dates do not prove the absence of a later journal publication.

On 21 October, direct Crossref REST queries inspected the first 20 results for
`query.title=Topology and geometry of Gaussian random fields II: on critical points, excursion sets, and persistent homology`
with `query.author=Pranav`, and the first 10 results for
`query.bibliographic=2109.08721`. Neither inspected result set contained a
journal match. The identifier query returned unrelated fuzzy matches, not an
exact-ID absence certificate. The title query returned the distinct **Part I**
paper, DOI `10.1093/mnras/stz541`; its MNRAS 485(3), 4167–4208 (2019) metadata
must not be transferred to Part II.
The directly retrieved
[INSPIRE arXiv record](https://inspirehep.net/api/arxiv/2109.08721)
matches Part II and has no `dois` or `publication_info` fields.
Exact-title/identifier web searches, including publisher-targeted searches,
found no verified journal version. This is a documented search outcome,
not a claim that every index was exhaustively searched.

**Access limits and disposition.** Crossref and INSPIRE APIs were reachable in
the final direct checks, despite earlier web-fetch failures. The DataCite API
request for `10.48550/arxiv.2109.08721` returned **HTTP 503**; the preprint DOI
is nevertheless explicitly supplied by arXiv and CORDIS. NASA ADS, Semantic
Scholar API and some repository full-text routes did not yield usable records
during the parallel checks. Direct Project Euclid retrieval returned a page
shell; its first-availability date was read from indexed publisher text,
while final citation metadata was independently obtained from Crossref and
Aarhus. No paywalled journal full-text equivalence check or new MathSciNet
search was performed.

This addendum **supersedes** the earlier G6 assertion that Pranav “has no
published version,” and the §5 instruction to repeat G6 as a human lookup.
The bounded check is complete; a journal version of Pranav remains
**unverified**, with the preprint citation retained. No required `refs.bib`
change follows from this check. G1, G3 and the separate human, subscription,
venue, archiving and submission decisions retain their existing requirements.

## 3. Novelty search, 21 October 2026 (fourth pass; sources this session could reach)

**Question.** Does any prior work give the near-diagonal density of H₀ (maximum–saddle) persistence pairs of a smooth Gaussian field in dimension ≥ 2, or the ℓ^{-1/3} law, its coefficient reduction, or the side-24 correction?

**Result.** No such work was found. The closest results are listed below.

| Work | What it does | Relation |
|---|---|---|
| Y. Baryshnikov, *Brownian motions, persistent homology and chirality*, J. Appl. Comput. Topol. 9 (2025), art. 28 ([link](https://link.springer.com/article/10.1007/s41468-025-00224-w)) | Explicit H₀ intensity for Brownian motion with drift; density near the diagonal ~ (mΔ)^{-3}. It says generalizing to higher-dimensional Gaussian processes "seems difficult but not entirely out of reach". | **One-dimensional analogue. Not yet cited (G5).** It supports novelty in d ≥ 2. |
| D. Perez, arXiv:2012.09459 ([link](https://arxiv.org/abs/2012.09459)) | Barcodes of almost surely C⁰ processes on an interval. | Already cited; d = 1. |
| P. Pranav, arXiv:2109.08721 ([link](https://arxiv.org/abs/2109.08721)) | Persistence diagrams of 3D Gaussian fields, by simulation; descriptive statistics. | Already cited; no analytic near-diagonal law. |
| F. Chazal, V. Divol, arXiv:1802.10457 ([link](https://arxiv.org/abs/1802.10457)) | Existence of a density for expected persistence diagrams. | Already cited; existence only, no asymptotics. |
| E. Heo, arXiv:2610.08257 (6 Oct 2026) ([link](https://arxiv.org/abs/2610.08257)) | Near-diagonal asymptotics of persistence ratios in random Čech complexes. | Already cited; point clouds, not sublevel sets of fields. |
| A. M. Thomas, arXiv:2407.07326 ([link](https://arxiv.org/abs/2407.07326)) | Law of large numbers and CLT for persistence diagrams of discrete-time stationary processes. | d = 1, discrete time; no short-bar density. Optional citation. |
| S. Kuriki, T. Matsubara, S. Iso, arXiv:2607.23903 (27 Jul 2026) ([link](https://arxiv.org/abs/2607.23903)) | Expected density and heights of critical points of isotropic Gaussian fields in any dimension. | Critical points only, no pairing. Optional citation beside Azaïs–Delmas. |
| L. Nicolaescu, arXiv:2408.14383 ([link](https://arxiv.org/abs/2408.14383)) | Law of large numbers for the number of critical points of isotropic Gaussian functions, via second-moment asymptotics. | Critical points only; earlier passes listed it as optional. |

**Coverage limits.**
- **Searched:** web search (several queries) and arXiv abstract pages.
- **Not reached:**
  - the arXiv API (robots exclusion);
  - zbMATH Open (fetch error; **[BROWSER]**);
  - MathSciNet (**[SUBSCRIPTION]**);
  - the Consensus paper index (monthly quota spent until 1 November).
- **Caution.** Web search is not a systematic literature search; G3 and G4 remain. The abstract-level reading above does not exclude a relevant result in a paper's body. **[HUMAN]** judgement is needed on Baryshnikov (2025), which is the closest work.

## 4. Venue comparison

| Venue | Fit | AI policy (as found) | Long supplement | Notes |
|---|---|---|---|---|
| **Electronic Journal of Probability** (IMS/Bernoulli Society; open access) | Good: long rigorous probability papers, Gaussian fields, Kac–Rice. | Not found: the IMS pages blocked this session (**[BROWSER]**). | Long papers are accepted by tradition; supplement policy unverified (**[BROWSER]**). | No author fees as far as found; free to read. |
| **Annals of Applied Probability** (IMS) | Good: random topology and TDA probability appear there (e.g. Hiraoka–Shirai–Trinh, *Limit theorems for persistence diagrams*, 2018). | Not found (**[BROWSER]**). | IMS journals host supplements; size rules unverified (**[BROWSER]**). | More selective; longer review times are typical **[HUMAN to judge]**. |
| **Annales de l'IHP (B) Probab. Stat.** (IMS-distributed) | Good: analytic probability. | Not found (**[BROWSER]**). | Unverified. | |
| **Journal of Applied and Computational Topology** (Springer) | Good for the persistence audience; Baryshnikov (2025) and a Random Topology special issue appeared there. | **Springer policy (fetched):** AI cannot be an author; "accountability … cannot be delegated to AI systems"; AI use must be disclosed; light language polishing is permitted. | Springer journals accept electronic supplementary material; size limits unverified. | The strongest audience match for the persistence framing. |
| **Probability Theory and Related Fields** (Springer) | Possible; top tier, wants broad probabilistic interest. | Springer policy as above. | As above. | High bar; the result's scope (fixed side 24, d = 3) may read as specialized **[HUMAN to judge]**. |

**Recommendation for Dylan's decision [HUMAN].**
- **J. Appl. Comput. Topol.** if the persistence community is the audience; its AI policy is the one confirmed.
- **EJP** if the probability community is the audience and open access matters.

Either way, close G1 first: every venue's policy makes the human author accountable for the proofs, and §7 currently says no human specialist has reviewed them.

Sources: the links in §3; [Springer AI editorial policy](https://www.springer.com/us/editorial-policies/artificial-intelligence--ai-/25428500); [JACT Random Topology special issue](https://appliedtopology.org/journal-of-applied-and-computational-topology-special-issue-on-random-topology/); [Hiraoka–Shirai–Trinh, AAP 2018](https://projecteuclid.org/journals/annals-of-applied-probability/volume-28/issue-5/Limit-theorems-for-persistence-diagrams/10.1214/17-AAP1371.short); IMS author pages for AAP/EJP ([AAP](https://www.imstat.org/journals-and-publications/annals-of-applied-probability/annals-of-applied-probability-manuscript-preparation), [EJP](https://imstat.org/journals-and-publications/electronic-journal-of-probability/electronic-journal-of-probability-manuscript-submission/)), which this session could not load.

## 5. Next actions

| Who | Action |
|---|---|
| Dylan **[HUMAN]** | G1 (human review route), the venue choice (§4), G8 (front-matter details), G9/G10 (archive and Supplement hosting), G13 (arXiv account). |
| Dylan **[SUBSCRIPTION]/[BROWSER]** | G3 and G4, with the queries in G3. G6's bounded check and remaining Pranav uncertainty are recorded in §2.1. |
| Author agent **[AI]** | r13–r19 done; r13–r18 read (r19's read and the A-F1 readback requested). #485 (G6 qualification, OpenAI proposal, documentary PASS_SCOPED 485/6328282240) adopted onto the branch as merge `29ca667d`. After the venue choice: G11, G12 and G14. |
