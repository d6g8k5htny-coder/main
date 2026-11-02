# QP-C54 tranche 4: the main objects that no catalog record names (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6567093162; checkpoint 6569878801. No repository code was run.

## Scope

`main_dirs.py` (output `main_dirs.json`) lists every `.md`, `.tex`, `.py`, `.json`, `.lean` and `.txt` file of main at
`d2b0216e` outside the engineering directories (`cbz/`, `tests/`, `tools/`, `.github/`, `docs/site/`, `docs/retro/`,
`governance/`, `ctx/`, `history/`, `interface/`, `records/`, `architecture/`, `data/`). It groups the files by directory and marks
each directory covered when a catalog location, or a disposed `PROOF.md` packet, is the directory or an ancestor. Each file is
marked `copy_of` when its Git blob equals a file at a disposed main or Math- location.

`objects.py` (output `objects.json`) turns the uncovered directories into **76 objects** (1,367 files):
- 72 are TO_CLASSIFY;
- 4 are resolved mechanically: `formal/UniversalLaw/Side24` (every Lean file is CLOSED), `formal/sources/p15_v1` and
  `reviews/lb_rate_thm023_landing_20260926` (every file disposed or a copy), and `formal/UniversalLaw/P15` (main#740's lane).

## Pipeline

1. **Batches** (`manifest.json`, `tbatch1.json`–`tbatch8.json`; workflow `workflow_t4.js`, as run, with its scratch paths).
   - Batches 1–6 are math-likely. Each gets a draft, a **fidelity** verifier (partition, classification, verbatim statements,
     hypotheses and limits) and a **Layer A** verifier (floating point from the code, overclaiming, open steps, blocking
     reasons, hidden mathematics in non-obligations).
   - Batches 7–8 are engineering-likely. Each gets a draft and one combined adversarial **screen** that searches every object
     for mathematical claims.
   - 22 agents ran. The verifiers fixed **268 problems**: 146 by fidelity, 115 by Layer A and 7 by the screens.
2. `verdicts_verified_t4.json` is the workflow's return value. Parsed as JSON, it equals the final-stage results recorded
   in the workflow journal.
3. `assemble_t4.py` checks mechanically:
   - every object appears exactly once, and its items partition its file list;
   - item ids are unique within an object;
   - every OBLIGATION has statements, each verbatim at its lines at `d2b0216e`, and a floating-point field that begins
     USED, NONE or NO_SCRIPTS;
   - every REVIEW_OF or COPY_OR_EXPOSITION_OF item names its targets;
   - the bindings come from Git;
   - each `refers_to` path is resolved against main, Math- (`3b11bff6`) and the catalog.

   An OBLIGATION item becomes a **PINNING_AMENDMENT** when an adopted record's text names one of the item's files without
   binding it. Otherwise it is a NEW_RECORD.
4. **One assembler rule was added after the run.** The first assembly failed on one item: `zf-numerics` in
   `experiments/zeta_fold_exploration` quotes two statements from the packet's `README.md`, which belongs to the item
   `zf-heat-fold-exact`.
   - The statements are the README's numeric fold table (`README.md:610-613`) and its sentence on the collisions found
     (`README.md:639`). They belong to the numeric item.
   - The rule: a statement may quote a file that another item of the same object holds. The file is then bound under
     `source_binding.context_files`, the statement is still checked verbatim, and the partition is unchanged. The quote is
     reported (`QP-C54-T4-055`).
   - On the other seven batches, the old and new assemblers give byte-identical output.
5. Rerunning the assembler on the retained verdicts reproduces `records_verified_t4.json` byte for byte.
6. Manual checks (`manual_checks.py`, output `manual_checks_output.txt`):
   - **NO_SCRIPTS** is consistent: no NO_SCRIPTS record binds a script, and scripts stored as `.py.txt` count as scripts.
     The check flags one record that is not NO_SCRIPTS but binds no recognised script: T4-003 (USED). Its code is bound
     only as a numbered listing, `lpw_constant.py.numbered.txt`, and its class is judged from that listing.
   - **Overclaim lint**: after quotations, file names and fixed phrases such as "closed form" are removed, 5 our-voice
     assertions remain. All 5 are negations or open items ("None of this is CLOSED", "No persistence software … is
     certified", "any claim that … are discharged").
   - **Privacy**: no e-mail address, token, signed link, SSN pattern, `/Users/` path or device code. The 4 device-code-shaped
     strings are claim-ID ranges (`C009-C104`, `C041-C046`, `C099-C101`) and a token name (`BAND-CERT`).
   - Three Google Drive file ids occur in one REVIEW_OF item's `refers_to`. They are quoted from main's own committed
     `docs/public-math/recovery-20260927/README.md`, so they add no new disclosure.

## Result (`records_verified_t4.json`)

251 items:

| Class | Count | What it means |
|---|---|---|
| OBLIGATION | **105** | `QP-C54-T4-001` to `-105`, in 23 objects: 85 NEW_RECORD and 20 PINNING_AMENDMENT. 287 quoted statements. Floating point: NO_SCRIPTS 55, USED 26, NONE 24. |
| REVIEW_OF | 21 | Each names what it reviews |
| COPY_OR_EXPOSITION_OF | 30 | Each names what it copies or expounds |
| NOT_OBLIGATION | 95 | Indexes, custody manifests, harnesses, plans and similar, each with a cited reason |

| Records | Object | What they state |
|---|---|---|
| T4-001 to -004 | `audits/dropbox_import/2026-09-27` | Lifetime-tail convergence erratum, the persistence-thermodynamics programme, the LPW_CONSTANT explicit pair, and the **LPW Return-05 findings** (exact c_rat, E[(\|X\|+\|Y\|)⁴] = 12 + 32/π, the Rayleigh-majorant replacement lemma and the nonzero Cov(J) entry) |
| T4-005 to -007 | `audits/exact_sample_shrinking_bin` | The exact-sample shrinking-bin lemma, its nonauthor review (threshold cA² > 10/3) and the reviewer's controls |
| T4-008, -009 | `docs` | Shrinking witness pairs away from the pins, and a conditional universality transfer |
| T4-010 | `docs/manuscript/side24_v3_draft_20261013` | Theorem 2.1 of the SIDE24 v3 draft |
| T4-011 to -022 | `docs/manuscript/side24_v3_supplement_draft_20261013` | Supplement sections S1–S13 and its concordance with errata |
| T4-023 to -025 | `docs/research-translation/20260930` | The planar side-24 cap-to-elder bridge, the axial older-path lemma, and the T24 completion audit (**amendment to Q-2012**) |
| T4-026 | `experiments/h3_band_v2` | Outward-rounded H3 band ceiling and floor of E[G_r] on eight cells |
| T4-027 to -037 | `experiments/periodic_h0` | Two new lemmas (T4-027 interpolation and bins, T4-037 mean-adaptive Chernoff–KL confidence), and **nine amendments to Q-1160–Q-1168** (T4-028 to -036) |
| T4-038 to -046 | `experiments/periodic_h0/results` | The nine stored certificates, as **amendments to Q-1160–Q-1168** |
| T4-047, -048 | `experiments/reactor_v0`, `experiments/st_contract_pilot` | A finite barcode control, and the ST R2 pushforward exponent algebra |
| T4-049 to -053 | `experiments/universality` | Exact counterexamples and controls for covariance-only exponents; T4-052 is an **amendment to Q-1169** |
| T4-054, -055 | `experiments/zeta_fold_exploration` | The backward-heat fold Proposition 1 with lobe identities, and the numerical de Bruijn–Newman collisions |
| T4-056 to -100 | `incoming/` custody packets (7 objects) | C022/Q0 ledgers, LS-DER and GP-DER derivations, CL-ERR and CL-RNU audits, LPW_CONSTANT v3/v4, B4LOC, D1/D3 and H3/H5 lanes, Grok replays, RN-UNIF |
| T4-101 to -105 | `reviews/grok_uploads_20260926`, `reviews/sard_g_*` | Review results with counterexamples of their own |

**Pinning amendments and the byte-pin sweep agree.** The 19 periodic-H0 and universality amendments target Q-1160–Q-1169,
the same records the sweep pins for these packets (`../pinning_sweep_20261102/pins.json`, which also pins Q-1025, Q-1032,
Q-1041 and Q-1043). Each amendment binds its whole item, where the sweep pinned only the files a record names.

**Referents.** 165 `refers_to` targets exist but had no disposition in catalog candidate v1. 152 of them are now covered by this
tranche's own rows. Of the other 13:
- 5 are files in four Math- review directories that tranche 3 covers;
- 4 are P15 Lean files or `formal/sources/p15_v1`, in main#740's lane;
- `experiments/periodic_h0/spectral_shrinking_bins/README.md` is inside Q-2002's packet;
- Math- `coefficients/side24_v1/coefficient.py` and `test_coefficient.py` sit beside Q-1001's `PROOF.md`;
- the notebook below.

145 further targets are PR numbers, comment ids or claim ids. They are reported by design, not failed.

## Reconciliation with tranche 6 (`reconcile_t6.py`, output `reconciliation_t6.json`)

The script matches tranche 6's bound files against this tranche's by Git blob. Main is inside the catalog's scope and the
federation repositories are not, so **main's copy is primary** for every shared blob.

- **QP-C54-T6-007** (trial's copy of LPW Return-05) shares 7 of its 8 files with main
  `audits/dropbox_import/2026-09-27/files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/`:
  - `00_READ_FIRST.md`, `02_CONSTANT_FINDINGS_AND_REPAIR.md`, `FINDINGS_NORMAL.json` and `tools/check_findings.py` are bound by
    **QP-C54-T4-004**, the primary record;
  - `CURRENT_STATE.json`, `FINAL_EXECUTION_RECEIPT.json` and `LOCAL_MUTATION_RECEIPTS.json` are bound by the main
    NOT_OBLIGATION item `lpw05-custody-receipts`.
- `04_RETURN_TO_KIMI.md` is in trial only. Its line 49, an editorial addendum on the nonzero A–D3 entry of Cov(J), is the one
  statement main does not hold. So T6-007 stays PROPOSED for that file only.
- Catalog candidate v2 attaches this as `reconciliation` on T6-007 and `federation_replicas` on T4-004. No row is removed.
- No other tranche-6 record shares a blob with this tranche's rows.

## Scope addendum (`scope_addendum.py`, output `scope_addendum.json`)

The only main file outside the census's file types that could carry mathematics is
`docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb`. I read all six cells by hand.
- It fetches `Math-@9d7b6802:coefficients/side24_v1/ENCLOSURE.json`, which adopted record Q-1170 covers, and checks its size
  and SHA-256.
- It prints the published endpoints and draws a chart labelled NON-CERTIFYING.
- It is recorded as COPY_OR_EXPOSITION_OF, with a Git binding.

The other excluded types are outputs and data (`.png`, `.log`, `.stdout`, `.stderr`, `.out`, `.csv`, `.tsv`, `.diff`), the two
compiled manuscript PDFs (their `.tex` sources are covered), build scripts and archives.

## Limits

- All agents are Anthropic Claude. This is engineering accounting, not a review of the mathematics, and no record is a verdict
  on any object's mathematics.
- The OBLIGATION / REVIEW_OF / COPY / NOT_OBLIGATION lines are judgements, and each one is open to challenge.
- Output files beside the scripts (`.out`, `.csv`, `.log`) are not bound. A record's floating-point field is judged from the
  code, not from those outputs.
- **Reproduce:**
  1. `STAGING` is `git show 32ab7d9013264d2874aa4b778a90eede0cda0cdf:cbz/quarantine/catalog.json`, and `CLOSED` is the same
     commit's `cbz/evidence/kernel/all-closed-declarations.json` (SHA-256 `c25b7e46…`).
  2. Build candidate v1: `python3 -I -B ../qp_c54_catalog_candidate_20261102/build_candidate.py STAGING
     ../qp_c54_quarantine_proposals_20261101/records_verified.json ../qp_c54_t2_20261102/records_verified_t2.json CAND`
     (SHA-256 `0f956ace…`).
  3. `python3 -I -B assemble_t4.py MAIN_CLONE MATH_CLONE . objects.json CAND CLOSED verdicts_verified_t4.json OUT`
  4. `python3 -I -B reconcile_t6.py records_verified_t4.json ../qp_c54_t6_federation_20261102/records_verified_t6.json OUT2`
  5. `python3 -I -B scope_addendum.py MAIN_CLONE OUT3`
