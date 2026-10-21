# SIDE24 V3: submission readiness, novelty search and venue comparison

Status as of 21 October 2026, about 20:20Z (r14 made). This is a living record. Update it when any item changes, and supersede lines rather than delete them.
Author of this record: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), for Dylan Roy. Dylan Roy — delegated AI work. Credit 0. Scientific effect NONE.
**Nothing here submits anything. Dylan has said not to submit yet.**

Marks: **[HUMAN]** needs Dylan or another person: judgement, identity, an account, or a decision. **[SUBSCRIPTION]** needs paid or institutional access. **[BROWSER]** the source is free, but this session's tools could not open it. **[AI]** an agent can do it, but only on Dylan's request. **[DONE]** done, with its record.

## 1. Where the documents stand

| Item | State | Record |
|---|---|---|
| Main text | r13, `04a3ee03`, 23 pages (documentary over r12 `d550d238`) | r12: Cursor PASS 6324514156. r13: Cursor AMEND 6325193355 (wording), applied; readback AMEND 6325247357 (change-list identity), applied; readback PASS 6325266500 |
| Script commits | AMEND applied, readback PASS (recorded separately from the r12 PASS) | main#389 6324563568, 6324618319 |
| Author decisions 1–4 | **[DONE]** defaults kept for 1–3; Supplement-only form of 4 | main#389 6324313740 |
| Supplement | draft 13, cited as [R1] at `d28e6ab4`, 162 pages | Cursor read with 2 AMENDs, readback PASS, main#389 6290560837 |
| Provider-distinct AI reads | every section of both documents has at least one (OpenAI or xAI), PASS_SCOPED | main#389 PR body; main#419 6294239616 |
| Organizational independence | 0: every AI lane posts from one account | §7 of the draft |

## 2. Remaining gaps before submission (exact)

| # | Gap | Mark | What closes it |
|---|---|---|---|
| G1 | **No human specialist has reviewed the proofs.** §7 says so. Every journal below holds the human author fully accountable for correctness; Springer's policy says that accountability "cannot be delegated to AI systems". | **[HUMAN]** | Either Dylan reads the main text line by line and can stand behind every step, or a probabilist with Kac–Rice and persistence experience reads it, ideally both. Then update §7 to say what was done. |
| G2 | **The Supplement (162 pages) has not been read as a whole.** It has slice reads only. | **[HUMAN]** (an AI whole read would add evidence but not close G1) | A whole read for consistency of notation, dependencies and numbering. The `checks/dep_graph_check.py` script already reports no unexplained proof cycle. |
| G3 | **The MathSciNet search has not been run.** | **[SUBSCRIPTION]** | Search MR for "persistence diagram" or "barcode" with "Gaussian field" or "random function", and for close critical-point pairs. Follow the forward citations of Adler–Taylor, Azaïs–Wschebor, Chazal–Divol, Perez and Baryshnikov (2025). |
| G4 | **The zbMATH Open search has not been run** from this session: the site returned an error to the fetch tool. It is free. | **[BROWSER]** | Run the same queries at zbmath.org. |
| G5 | **Bibliography additions found in this pass** (§3). | **[DONE]** in r13 (Cursor AMEND 6325193355 applied; readbacks PASS 6325266500) | Add Baryshnikov, *Brownian motions, persistent homology and chirality*, J. Appl. Comput. Topol. 9 (2025) 28. Mention it in §6 as the one-dimensional near-diagonal law (Brownian motion with drift). Its outlook names higher-dimensional Gaussian processes as future work. |
| G6 | **Publication status of three arXiv-only references is unverified** (r14, 21 Oct: the arXiv pages of all three still list no journal reference; Chenavier–Hirsch's title corrected; HLO's Extremes DOI and Chenavier–Hirsch's pages still unconfirmed and asked of Cursor): Perez (arXiv:2012.09459), Pranav (arXiv:2109.08721), Hirsch–Lachièze-Rey (arXiv:2411.11429). | **[SUBSCRIPTION]** or **[BROWSER]** | Check each in MathSciNet or zbMATH, and add the journal reference where one exists. |
| G7 | **The §7 "Remaining before submission" paragraph is partly stale.** "The treatment of its imports from the parent record" is settled by decision 4 in r12. | **[DONE]** in r13 (same read) | Rewrite that sentence to list only what remains: G2, G3/G4, provenance labels, and the journal style pass. |
| G8 | **The front matter is incomplete.** (r14: MSC 2020 codes and keywords added; the identity details remain.) `main.tex` has `\author{Dylan Roy}` only, with no address, email, affiliation (or "independent researcher"), ORCID, MSC 2020 codes or keywords. | **[HUMAN]** for identity details; **[AI]** can propose MSC codes (60G60, 60G15, 55N31, 60D05) | Dylan supplies what he wants printed. |
| G9 | **The repository records cited in Appendix A are paths and commits** in a GitHub repository. Referees need stable, citable access. | **[HUMAN]** (needs an account and a decision) | Archive a tagged snapshot with a DOI (for example Zenodo from GitHub) and cite the DOI in Appendix A, or make the Supplement self-contained and post it on arXiv as a companion. Decision 4 already makes the leading term rest on the Supplement alone. |
| G10 | **How the Supplement is hosted.** At 162 pages it will not be a normal journal supplement everywhere. | **[HUMAN]**: choose the venue first | Options: an arXiv companion paper cited by the main text, or a journal-hosted supplementary file. See §4. |
| G11 | **Draft-only material must be removed:** §0 (author status) and the provenance labels in the Supplement. | **[AI]**, on request after the venue choice | A final "submission build" revision. |
| G12 | **Journal style pass** (class file, bibliography style, length). | **[AI]**, after the venue choice | BibTeX "missing pages" warnings: BobrowskiBorman and LadghamLR fixed in r14; ChenavierHirsch and HLO remain (asked of Cursor, which can reach Crossref). |
| G13 | **arXiv posting.** A first-time math.PR submitter may need an endorsement. | **[HUMAN]** (account) **[BROWSER]** (check arXiv's current endorsement page) | Dylan's arXiv account and, if asked, an endorser. |
| G14 | **AI-use disclosure wording** must match the chosen journal's policy. §7 already discloses AI checks; the venue may want a statement about AI-generated text and proofs. | **[HUMAN]** decision, **[AI]** draft | After the venue choice, compare §7 with the journal's policy text. |

**Not a gap for submission:** main#389 does not need to merge for submission. The draft lives on its branch with full records.

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
| Dylan **[SUBSCRIPTION]/[BROWSER]** | G3, G4 and G6, with the queries in G3. |
| Author agent **[AI]** | r13 (G5, G7) is done and read (main#389 6325193355, 6325266500). After the venue choice: G11, G12 and G14. |
