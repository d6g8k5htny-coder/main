---
name: AI edit 0000 — WORKING LEDGER
opened: 2026-07-20
governs: the `AI edit NNNN` series
---

# AI edit 0000 — Working ledger

**What this is.** A live working document for the AI-edit pass over the corpus. Issues get raised here, worked here, and resolved here. It is not an audit report and does not sit outside the project — it is part of it, and it is expected to be edited, contradicted, and superseded like anything else in the set.

**What this is not.** A verdict on the mathematics. Nothing in the `AI edit` series changes a claim's status. Status is set by the author and by the review package's own §1 table (see 0002).

---

## Conventions for the `AI edit NNNN` series

| Rule | Detail |
|---|---|
| Naming | `AI edit NNNN - <original filename>` |
| Header | Prepended block, marked **AI-GENERATED NAVIGATION HEADER — NOT AUTHOR CONTENT**, above a horizontal rule. Body below the rule is the original, unchanged. |
| Numbering | Assigned in **reading order for a cold reader**, which may differ from the original file numbering. Current mapping in the manifest below. |
| Revisions | A re-edit of an existing AI edit increments a decimal: `0001` → `0001.1` → `0001.2`. The prior version is not deleted. |
| Concerns | Raised inside the file, in an **AI REVIEW NOTES** block within the header, and logged here. Not raised only in conversation. |
| This file | `0000` is a deviation from the scheme — it replaces no original. It sits at 0000 so it sorts to the front. |

**Known limitation.** A prose header cannot be prepended to `.json` or `.csv` without breaking machine parsing, and cannot be added to `.png` at all. Those file types need either an in-structure key (e.g. an `_ai_edit` object inside the JSON) or coverage from an index rather than a header. **Undecided — see ISS-09.**

**Second limitation.** A trail header is only worth writing for a file that has actually been read. Inferring a dependency map from a filename would defeat the purpose. Throughput is therefore batch-wise, not corpus-wide in one pass.

---

## Manifest

| AI edit | Original | Status |
|---|---|---|
| 0000 | — (this file) | live |
| 0001 | `00_READ_ME_FIRST.md` | created 2026-07-20 |
| 0002 | `05_RELIABILITY_AND_EVIDENCE.md` | created 2026-07-20 |
| 0003 | `01_CORE_PAIRING_THEOREM.md` | created 2026-07-20 |
| 0004 | `02_GAUSSIAN_TRANSVERSALITY.md` | created 2026-07-20 |
| 0005 | `03_NEAR_DIAGONAL_LIFETIME_LAW.md` | pending |
| 0006 | `04_THERMODYNAMIC_AND_PERCOLATION_EXTENSION.md` | pending |
| 0007 | `06_REVIEW_RESPONSE_FORM.md` | pending |

Reading order deviates from original numbering in one place: `05_RELIABILITY_AND_EVIDENCE.md` was moved from position 6 to position **2**, ahead of all proof files. Rationale: its §1 claim-status table is the master key, and reading a proof before knowing whether it is proven or sketched is the single largest source of misreading in the package.

Beyond 0007 the corpus continues into the `C09x` / `C10x` / `GATE` / `q0_` families, several hundred files. Those are a different and larger body of work and are **not** covered by this pass yet.

---

## Issue states

`OPEN` — raised, not worked · `WORKED` — analysis done, awaiting author decision · `PROPOSED` — a specific repair is on the table · `RESOLVED` — closed, with the resolution recorded · `DEFERRED` — real but not blocking

---

## Issues

### ISS-01 · Supersession conflict between Master v3.2 and the review package
**State: RESOLVED-BY-RULE · recorded in 0002**

`00 INVENTORY AND TRACE.md` (Master v3.2, modified 2026-07-14) carries live statuses that the review package (2026-07-20) withdraws:

| Object | v3.2 | Review package |
|---|---|---|
| Theorem A | two-sided, 0.8411·r³ ≤ 1−q ≤ 4.3·r³ | one-sided, C unspecified |
| C_UB = 4.3 | live (registry #64) | withdrawn |
| C*(0.025) = 0.946 | live (#17) | retired |
| R0 / transversality | resolved at program grade | uncertain |

Resolution applied: rule **R2** of the extraction directive — latest governs live text. The review package post-dates v3.2 by six days and its withdrawals are deliberate and individually reasoned. A full supersession notice is at the head of AI edit 0002.

**Residual, for the author:** v3.2 itself carries no marker pointing forward. It is the most recently *modified* file in the Drive, so recency-ordered crawlers reach it first and read the over-claimed statuses. A one-line stamp at its head would close this permanently. That is an edit to an original file and has not been made.

---

### ISS-02 · Duplicate copies of package files
**State: OPEN**

`02_GAUSSIAN_TRANSVERSALITY.md` exists in three byte-identical copies (19,242 bytes each): Drive root at 18:33 and 19:27, and one inside folder `12noAmwDlUEyHhiuW4ov2YRpDAZ9k6qTf` at 20:23, all 2026-07-20.

Found incidentally while resolving a file ID, not by a systematic scan. Other package files probably have the same pattern. Any process enumerating the corpus will multiply-count.

**Next step:** a title-collision scan across the whole Drive, producing a canonical-copy table. Not yet run.

---

### ISS-03 · §11 sum-of-squares is a convex combination of two squares
**State: PROPOSED · detail in 0003, Note 1**

The generic-chart bracket \((K+8\alpha c\eta)^2+64\alpha(1-\alpha)c^2\eta^2\) has cancelling \(\alpha^2\) terms and equals \((1-\alpha)K^2+\alpha(K+8c\eta)^2\). Transverse chart likewise: \((1-\alpha)(a+\eta)^2+\alpha(a-\eta)^2\). Verified by independent expansion.

This **strengthens** §11; nothing breaks. Three consequences: the no-go is structurally coextensive with the persistence window rather than coincidentally aligned with it; the endpoints \(\alpha=0,1\) are perfect squares with direct meaning; and the quartic boundary layer that §11 calls "the most delicate step" is localized to where one endpoint square vanishes — \(K\to0\) or \(K+8c\eta\to0\) — which is codimension one, not diffuse.

**Author decision needed:** whether to adopt the convex form as the primary statement in §11.

---

### ISS-04 · §2.4's typed-to-adjacent transfer needs one lemma, not two
**State: PROPOSED · detail in 0003, Note 2**

Since \(W_{MS}\ge0\) and \(\mathbf 1_{\mathcal A_r}\le1\), dropping the indicator in the numerator gives \(P^{\mathrm{adj}}_r(\text{defect})\le P^{MS}_r(\text{defect})/a_r\) directly. No reweighted regional counts are required. The whole burden reduces to \(\inf_r a_r>0\).

Consequence if adopted: §13's second hypothesis ("regional count estimates remain uniformly bounded after adjacency reweighting") can be struck, and review question 1 in 0007 narrows to a single positivity statement.

---

### ISS-05 · §6 of the transversality file has an unexcluded configuration
**State: PROPOSED · detail in 0004, Note 2**

Proposition 6.1 argues the curve term is nonzero because \(a_fn_f\) is "continuous and nonzero." That is insufficient on its own: a *tangential* density with \(a_fn_f^{\tan}\) constant integrates by parts to zero against every compactly supported interior test function.

The configuration is excluded by a conservation law. With \(\dot w=-H_fw\) and \(H_f^\top=H_f\), the quantity \(w\cdot\nabla f\) is constant along the orbit; the section normalization sets it to zero at \(t_+\); hence \(w\perp\gamma'\) everywhere. The adjoint covector is normal, never tangential, and the test-function argument goes through.

**Proposition 6.1 is correct** — it is one line short. Adding that line removes the strongest local objection to §6.

---

### ISS-06 · §8.3 chart conditions omit a lower bound on ν
**State: PROPOSED · detail in 0004, Note 3**

§8.1 bounds Hessian eigenvalues away from zero, which does not bound the ratio \(\nu=|\lambda_{\mathrm{transverse}}|/\lambda_{\mathrm{departure}}\) away from zero. Since that ratio has a continuous distribution at a Gaussian saddle, \(\nu\) is arbitrarily small with positive probability and \(\|G_{\mathrm{curve}}\|_{\mathcal H}\) has no uniform bound across a chart.

**Repair, cost-free:** add "eigenvalue ratio bounded below by a rational \(\nu_0>0\)" to the §8.3 list. Family stays countable, coverage is unaffected, and every chart gains a uniform bound — which §9's selection and §10's disintegration will want.

---

### ISS-07 · 0.8411 is not named in the §4 withdrawal list
**State: OPEN — question for the author**

The review package §4 names 0.8501 and 0.84 on the lower side. 0.8411 (v3.2 registry #26, the rigorized tier) appears nowhere in §4. It is barred only by the blanket "no certified finite upper or lower coefficient" statement in 0001.

Either that is deliberate — 0.8411 survives as a tier value under a different status — or §4 has an omission. Recorded rather than assumed.

---

### ISS-08 · §3.5's exponent fits center near −1/2, not −1/3
**State: WORKED — disagreement recorded**

Primary fit \(\hat\alpha=-0.4814\), CI \([-0.7115,-0.2437]\), contains \(-1/3\). Larger-cutoff fit \(\hat\alpha=-0.5005\), CI \([-0.6385,-0.3791]\), **excludes** \(-1/3\). Both point estimates sit near \(-0.5\); the primary interval contains \(-1/3\) largely because it is 0.47 units wide.

The file calls this "supportive under the primary criterion." That reading is defensible under pre-registration but generous. Two windows centering on \(-1/2\), with the better-powered one rejecting \(-1/3\), is closer to soft disconfirmation than to support. Finite-size or truncation bias toward steeper slopes is a live explanation and should be stated as such rather than resting on interval width — a referee will read the point estimates before the intervals.

**Not a claim that \(-1/3\) is wrong.** The exact fold Jacobian giving \(\ell^{-1/3}\) is independent of this experiment. The issue is how the evidence is characterized.

---

### ISS-09 · Header scheme does not extend to JSON, CSV, or PNG
**State: OPEN — decision needed before the pass reaches those families**

A prose header breaks JSON and CSV parsing and cannot attach to PNG. Options:

1. an `_ai_edit` key inside the JSON object (parse-safe, but mutates the artifact — and several of these files are hash-frozen);
2. leave those files untouched and cover them from an index;
3. a sidecar `AI edit NNNN - <name>.json.md` carrying only the header.

Option 1 conflicts with the freeze/attestation discipline in the `C09x`/`C10x` families. Option 3 is probably right for hashed artifacts. Undecided.

---

### ISS-10 · Escape transform on the duplicates is unverified
**State: OPEN**

`read_file_content` returns markdown escaped (`\#` for `#`, `\\(` for `\(`). The duplicates were written with that escaping reversed, which is a judgement about what the originals contain, not a measurement. If any AI edit renders with visible backslashes or broken math, this is the cause and every file in the series needs regenerating.

Cheapest check: open 0001 and confirm the display equations render.

---

## Working protocol

1. Issues found while processing a file are logged here **and** in that file's AI REVIEW NOTES block.
2. Nothing here modifies an original file. Changes to originals are the author's.
3. A resolved issue keeps its entry; the resolution is appended, not substituted.
4. Where this ledger and a file's header disagree, the header governs for that file and the discrepancy becomes a new issue.
5. Disagreement with the author's own characterization is recorded as `WORKED`, not `RESOLVED` — see ISS-08.
