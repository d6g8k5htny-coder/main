---
name: AI edit 0000.1 — WORKING LEDGER
supersedes: AI edit 0000 (2026-07-20, retained, not deleted)
revised: 2026-07-20
governs: the `AI edit NNNN` and `AI edit WNN` series
---

# AI edit 0000.1 — Working ledger

**Revision note.** Supersedes AI edit 0000. Changes: added the **W-series** convention (§Conventions); added ISS-11 through ISS-13; ISS-03 extended to general \(\kappa\); ISS-04 folded into ISS-13; **ISS-08's direction corrected** — it pointed at the wrong file.

**What this is.** A live working document for the AI-edit pass. Issues get raised, worked, and resolved here. Not an audit report and not outside the project — expected to be edited, contradicted, and superseded like anything else in the set.

**What this is not.** A verdict on the mathematics. Nothing in the `AI edit` series changes a claim's status. Status is set by the author and by the review package's own §1 table (0002).

---

## Conventions

| Rule | Detail |
|---|---|
| **NNNN series** | `AI edit NNNN - <original filename>` — a duplicate of an existing file with a prepended navigation header. Body below the header rule is the original, unchanged. |
| **W series** | `AI edit WNN - <descriptor>` — **new material**, not a duplicate of anything. Proposed lemmas, constructions, scans. Marked PROPOSED and machine-generated. Currently: W01. |
| Numbering | NNNN assigned in **reading order for a cold reader**, which may differ from original file numbering. |
| Revisions | `0001` → `0001.1` → `0001.2`. Prior versions retained, never deleted. This file is the first such revision. |
| Concerns | Raised inside the relevant file's **AI REVIEW NOTES** block *and* logged here. Not raised only in conversation. |
| 0000 | Deviation from the scheme — replaces no original, sorts to the front. |

**Limitation A.** Prose headers break `.json` and `.csv` parsing and cannot attach to `.png`. See ISS-09.
**Limitation B.** A trail header is only worth writing for a file actually read. Inferring dependencies from a filename defeats the purpose. Throughput is batch-wise.

---

## Manifest

| AI edit | Original | Status |
|---|---|---|
| 0000 | — | superseded by this file |
| 0000.1 | — (this file) | live |
| 0001 | `00_READ_ME_FIRST.md` | created |
| 0002 | `05_RELIABILITY_AND_EVIDENCE.md` | created |
| 0003 | `01_CORE_PAIRING_THEOREM.md` | created |
| 0004 | `02_GAUSSIAN_TRANSVERSALITY.md` | created |
| 0005 | `03_NEAR_DIAGONAL_LIFETIME_LAW.md` | created |
| 0006 | `04_THERMODYNAMIC_AND_PERCOLATION_EXTENSION.md` | pending |
| 0007 | `06_REVIEW_RESPONSE_FORM.md` | pending |
| W01 | — (new) | proposed lemma: adjacency positivity |

Reading order deviates from original numbering once: `05_RELIABILITY_AND_EVIDENCE.md` moved from position 6 to **2**, ahead of all proof files, because its §1 claim-status table is the master key.

Beyond 0007 the corpus continues into the `C09x` / `C10x` / `GATE` / `q0_` families — several hundred files, **not covered by this pass**.

---

## Issue states

`OPEN` — raised, not worked · `WORKED` — analysis done, awaiting author decision · `PROPOSED` — specific repair on the table · `RESOLVED` — closed, resolution recorded · `DEFERRED` — real, not blocking

---

## Issues

### ISS-01 · Supersession conflict, Master v3.2 vs review package
**RESOLVED-BY-RULE · recorded in 0002**

v3.2 (mod. 2026-07-14) carries live statuses the review package (2026-07-20) withdraws: Theorem A two-sided vs one-sided; C_UB = 4.3 live vs withdrawn; C*(0.025) = 0.946 live vs retired; R0 resolved-at-program-grade vs uncertain. Applied rule **R2**, latest governs. Full notice at the head of 0002.

**Residual for the author:** v3.2 carries no forward marker and is the most recently *modified* file in the Drive, so recency-ordered crawlers hit the over-claimed statuses first. A one-line stamp at its head closes this. That is an edit to an original and has not been made.

---

### ISS-02 · Duplicate copies of package files
**OPEN — escalated by ISS-11**

`02_GAUSSIAN_TRANSVERSALITY.md`: three copies, all 19,242 bytes (root 18:33, root 19:27, folder `12noAmwDlUEyHhiuW4ov2YRpDAZ9k6qTf` 20:23).
`03_NEAR_DIAGONAL_LIFETIME_LAW.md`: **five** copies — see ISS-11.

Both found incidentally while resolving file IDs, not by systematic scan. **Next step:** a title-collision scan across the whole Drive producing a canonical-copy table with sizes. Not yet run.

---

### ISS-03 · §11 sum-of-squares is a convex combination of two squares
**PROPOSED · 0003 Note 1; extended in 0005 Note 2**

\((K+8\alpha c\eta)^2+64\alpha(1-\alpha)c^2\eta^2=(1-\alpha)K^2+\alpha(K+8c\eta)^2\); transverse chart \((1-\alpha)(a+\eta)^2+\alpha(a-\eta)^2\). \(\alpha^2\) terms cancel identically. Verified by independent expansion.

**Extension (0005 Note 2):** holds at every \(\kappa\), with \(K\to K_\kappa\) and \(c\eta\to c\eta\kappa\). This converts 0005 §7's *Status* caveat — "argued by compactness rather than displayed constants" — into something checkable by inspection, since the \(\kappa\)-dependence is confined to two endpoint squares, each polynomial in \(\kappa\).

Strengthens; nothing breaks. Localizes 0003 §11's "most delicate step" to a codimension-one set where one endpoint square vanishes.

**Author decision:** adopt the convex form as the primary statement in 0003 §11 and 0005 §7.

---

### ISS-04 · Typed-to-adjacent transfer needs one lemma, not two
**PROPOSED · 0003 Note 2 · now folded into ISS-13**

\(W_{MS}\ge0\) and \(\mathbf 1_{\mathcal A_r}\le1\) give \(P^{\mathrm{adj}}_r(\text{defect})\le P^{MS}_r(\text{defect})/a_r\) directly. No reweighted regional counts needed. Burden reduces to \(\inf_ra_r>0\). If adopted, 0003 §13's second hypothesis can be struck.

---

### ISS-05 · Transversality §6 has an unexcluded configuration
**PROPOSED · 0004 Note 2**

Prop 6.1's "continuous and nonzero" is insufficient alone: a tangential density with \(a_fn_f^{\tan}\) constant integrates by parts to zero against every compactly supported interior test function. Excluded by conservation: \(w\cdot\nabla f\) is constant along the orbit, and the section normalization sets it to zero, so \(w\perp\gamma'\) everywhere. **Prop 6.1 is correct, one line short.**

---

### ISS-06 · §8.3 chart conditions omit a lower bound on ν
**PROPOSED · 0004 Note 3**

§8.1 bounds eigenvalues away from zero but not the ratio \(\nu=|\lambda_{\mathrm{transverse}}|/\lambda_{\mathrm{departure}}\). That ratio has a continuous distribution at a Gaussian saddle, so \(\|G_{\mathrm{curve}}\|_{\mathcal H}\) has no uniform chart bound. **Repair:** add "eigenvalue ratio bounded below by rational \(\nu_0>0\)" to the §8.3 list. Family stays countable; coverage unaffected.

---

### ISS-07 · 0.8411 is not named in the §4 withdrawal list
**OPEN — question for the author**

§4 names 0.8501 and 0.84 on the lower side. 0.8411 (v3.2 registry #26, rigorized tier) appears nowhere in §4, and is barred only by the blanket no-certified-coefficient statement in 0001. Either deliberate or an omission. Recorded rather than assumed.

---

### ISS-08 · Characterization of the exponent experiment
**WORKED — direction corrected in this revision**

0000 recorded this as a disagreement with 0002 §3.5's "supportive under the primary criterion." That was right about the language but wrong about the target. **0005 §12.2 already gets it right** — it says the evidence is "mixed," names the cutoffs (primary 0.2, secondary 0.3), and reports the failed proxy's exact figures (0.1487, CI \([0.1370,0.1616]\), vs predicted \(2/3\)).

So the inconsistency is *internal*: the reliability dossier, which reviewers are told to read first as the master key, is the softer of the two files on the same data. **Recommend bringing 0002 §3.5 into line with 0005 §12.2, not the reverse.**

Substance unchanged: point estimates \(-0.4814\) and \(-0.5005\) both sit near \(-1/2\); the better-powered fit excludes \(-1/3\); the primary interval contains it largely because it is 0.47 units wide. Finite-size or truncation bias toward steeper slopes is a live explanation and should be stated, not left to interval width.

**Not a claim that \(-1/3\) is wrong.** 0005 §4's Jacobian is exact arithmetic, \(d/p-1\) with \(d=2,p=3\), independent of every experiment.

---

### ISS-09 · Header scheme does not extend to JSON, CSV, PNG
**OPEN — decision needed before the pass reaches those families**

Options: (1) an `_ai_edit` key inside the JSON — parse-safe but mutates hash-frozen artifacts; (2) leave untouched, cover from an index; (3) a sidecar `AI edit NNNN - <name>.json.md` carrying only the header. Option 1 conflicts with the freeze/attestation discipline in the `C09x`/`C10x` families. Option 3 is probably right for hashed artifacts.

---

### ISS-10 · Escape transform on the duplicates is unverified
**OPEN**

`read_file_content` returns markdown escaped (`\#`, `\\(`). Duplicates were written with that reversed — a judgement about the originals, not a measurement. If any AI edit renders with visible backslashes or broken math, this is the cause and the whole series needs regenerating. **Cheapest check:** open 0001 and confirm the display equations render.

---

### ISS-11 · One copy of `03_NEAR_DIAGONAL_LIFETIME_LAW.md` differs by a byte
**OPEN — escalates ISS-02 from redundancy to possible silent divergence**

| Time (2026-07-20) | Bytes | Location |
|---|---|---|
| 18:32 | 15,874 | root — treated as canonical |
| 19:30 | 15,874 | root |
| **19:36** | **15,873** | root |
| 19:39 | 15,874 | root |
| 20:23 | 15,874 | folder `1xaOQAcMGZOLPtc82XyZL2nIX2HeLeMDT` |

At least one copy is not a pure duplicate. A one-byte delta could be a trailing newline or a dropped digit in a constant — very different consequences, not resolvable by reading. Needs a byte-level diff or hash comparison. **Until then, treat 19:36 as suspect.** 0005 was built from the 18:32 copy.

---

### ISS-12 · Reviewer-load consolidation in 0007
**PROPOSED — depends on ISS-13**

If W01 survives review, Track A question 1 and the Track C §6 question should be **merged into one numbered item** in `06_REVIEW_RESPONSE_FORM.md`, stated as the uniform positivity of \(a_r\), with the limiting case marked attempted and the \(C^2\)-convergence extension marked open. Two principal review questions become one, and the survivor is sharper.

---

### ISS-13 · The adjacency lemma is shared by Track A and Track C
**WORKED — attempted in W01**

0003 §2.4's \(a_r=P_r^{MS}(\mathcal A_r)\) and 0005 §6's \(a_r(\theta,b,\kappa)\) are the same object, mark-integrated and mark-resolved respectively. Combined with ISS-04, **one lemma closes both tracks**:

> \(a_r(\theta,b,\kappa)\ge c>0\), uniformly on \(0<r\le r_0\) and compact \(\theta,b,\kappa\).

**W01 attempts the limiting case** and obtains an explicit \(a_0\gtrsim10^{-5}\) from a trapping-box construction on the canonical cubic fold, using the independence of the free jets \((Q,a,w,z)\sim N((-b,0,0,0),\operatorname{diag}(2,2,2,6))\).

Not established by W01: uniformity in \(r\) (needs 0005 §6.1's \(C^2\) convergence); step 4, nullity of the nonadjacency boundary; uniformity in \(\theta,b,\kappa\); the entry-corner detail.

**Check first:** whether "gradient adjacency" as defined in 0005 §2 is strictly stronger than "the leftward unstable branch of \(S\) has \(\omega\)-limit \(M\)." Definitional, cheap, and determines whether W01 is on target at all.

**A reviewer working this should be pointed at 0005 §6.2, not 0003 §2.4** — the four-step program there is further along.

---

## Working protocol

1. Issues found while processing a file are logged here **and** in that file's AI REVIEW NOTES block.
2. Nothing here modifies an original file. Changes to originals are the author's.
3. A resolved issue keeps its entry; the resolution is appended, not substituted.
4. Where this ledger and a file's header disagree, the header governs for that file and the discrepancy becomes a new issue.
5. Disagreement with the author's own characterization is recorded as `WORKED`, not `RESOLVED` — see ISS-08.
6. W-series material is `PROPOSED` until the author accepts, repairs, or kills it. Each W file carries its own "what would kill this" section; that section is not optional.
