**Target.**
- **Main repo:** `experiments/zeta_fold_exploration/`. This is a new directory, disjoint from every open draft.
  - Do NOT place it under `experiments/universality/`. That suite is stdlib-only and its CI runs `-S`, under which mpmath cannot be imported. Five open drafts (#330, #331, #334, #336, #340) also edit its README.
  - Do not edit `docs/RESEARCH_INDEX.md` in this PR; another job is editing it now.
- **Math-:** only later, and only if the T-CUE theorem (worth_exploring #1) is actually proved. Math- AGENTS.md:13 makes calculations stdlib-only, so the mpmath/numpy numerics cannot live there.

**File list** (sources are the scratch workdirs claims/, fold/, stats/, project-map/, synth/):
- `README.md`, from the draft at `.../scratchpad/zeta/synth/README_draft.md`, sha256 d0763758fb8e19d56e41edff876e5e1d75ff9ef73dbae24100696e63e735671d.
- `CLAIMS.md`: the claim-by-claim verification table from the claims job (verdicts C/CC/I/W), with the provenance of every citation (snippet-checked versus from memory). Paraphrase the note; do not quote it verbatim.
- `requirements.txt`: `numpy==2.5.3` (same pin as `experiments/periodic_h0`) and `mpmath==1.4.1`.
- `dbn/` (de Bruijn–Newman scripts):
  - from claims/: `dbn_common.py`, `check_dbn_basic.py`, `check_collision.py`, `check_collision_persistence.py`, `pers1d.py`;
  - from fold/: `hflow.py`, `zeros.py`, `track.py`, `fold_check.py`, `refine_check.py`, `count_check.py`, `validate_H0.py`, `analyze.py` renamed `fold_analyze.py`, `test_basic.py` renamed `smoke_hflow.py`;
  - `kappa_check.py` from synth/.
  - Keeping two independent H_t implementations is deliberate: they cross-check each other.
- `theta/`: `check_theta.py` (claims) and `checks.py` from project-map, renamed `project_theta_checks.py`.
- `stats/`: `rs_zeta.py`, `gauss_proc.py`, `cue.py`, `analyze.py`, `matched_tail.py`, `make_plots.py`, `check_gauss.py`, `check_zeta_quad.py`, `check_gue_transfer.py`, `sample_Z.py`, `check_Z_persistence.py`.
- `results/`:
  - JSON and CSV summaries: `fold_*.json`/`.csv`, `collision_events.csv`, `summary.json`, `matched_tail.json`, `rs_validation.json`, `zeta_quad_check.json`, `*_info.json`, `local_slopes.csv`, plus the ECDF CSVs if each is under about 1 MB;
  - the `*.out` logs;
  - `effective_exponent_*.svg`;
  - `MANIFEST.json` with the sha256 of every committed file and of each regenerable `.npz`. About 190 MB of raw `.npz` is not committed.
- `test_exact.py` (stdlib only, exact `fractions`):
  - the fold normal form gap r³/6;
  - backward heat polynomials v2–v5 satisfy ∂_t = −∂_x² (exact polynomial arithmetic);
  - κ_r, κ_ℓ, κ_t and the midpoint drift from exact power series in σ with rational A, B, C;
  - the §D2 exact pushforward exponents for α = 1, 2, including the log case.
- `test_numeric.py`: `unittest.skipUnless` mpmath and numpy are importable. Quick smoke checks under 2 minutes:
  - H_0 against Ξ/8 at 3 points;
  - the heat-equation residual;
  - re-converging the (9/10) collision t* from a seed to 15 digits;
  - Riemann–Siegel against `siegelz` at 5 points.
- **Before committing:** strip absolute scratch paths from the scripts, fix imports for the new subdirectories, and run everything once on Python 3.12 (the CI precedent) as well as 3.13.16.

**Labels and status.**
- Main has no "exploratory" label, and recent math PRs are unlabeled drafts. Do not create a label.
- Open an unlabeled **draft** PR titled `exploratory: zeta-zero folds and CUE short-bar control (scientific effect NONE)`. Avoid "Riemann Hypothesis" in the title.
- The README and PR body both state: "Exploratory. Scientific effect: NONE. No status change. Not RH progress."
- No change to STATUS.md, `lemma_closed`, prizes, premises or any register.
- Use the PR template's disposition block: head, owner (Claude lane), review scopes A–D, unresolved findings, validation run, next action.
- Dylan's personal reading is marked PENDING.

**Board and research guide.**
1. Before writing, post a scope claim on main#307 naming the exact path `experiments/zeta_fold_exploration/**`. Confirm that no other files are touched and that the paths are disjoint from #330/#331/#334/#336/#340 and from the RESEARCH_INDEX job.
2. Post the PR link and a review request on #307. Release the claim when the PR is ready for review.
3. After the in-flight RESEARCH_INDEX edit lands, open a separate small follow-up, coordinated with that file's owner. Add one line under item 4 (numerical evidence / exploratory runs) or under Open work:
   > "Exploratory comparison with zeta zeros and CUE (draft PR #N; scientific effect NONE): with the same cubic fold but quadratic pair repulsion, the short-bar density exponent is ≈0 rather than −1/3, an altered pair-count-tilt control. This is not a result about the Riemann Hypothesis."
   Run `python -B tools/navigation_check.py --verify-public`; navigation.yml triggers on `docs/**`.
4. Nothing goes on `docs/site` (the public home) before a nonauthor review. If it ever does, the non-claims must go with it.

**Review route.**
- Request a nonauthor read from a different provider (an OpenAI/Codex lane, or a Grok lane on request per the coordination rules), in four slices:
  - A: Proposition 1 and the κ coefficients;
  - B: CLAIMS.md corrections and the citations, ideally by a lane with working arXiv/journal access, to confirm every item marked "from memory" or "snippet";
  - C: the numerics, independently re-run, with zero counts checked against nzeros and a different seed or an independent Riemann–Siegel check;
  - D: a wording audit of the non-claims.
- Record the reviewer's lineage. A Claude-on-Claude read earns zero organizational-independence credit.

**Dependencies.**
- Precedent: `experiments/periodic_h0` pins numpy and gudhi in `requirements.txt`, and its own workflow `periodic-h0-pilot.yml` pip-installs them on Python 3.12. `experiments/universality` is stdlib-only and runs `python3 -B -E -S`.
- This packet: pin `numpy==2.5.3` and `mpmath==1.4.1`.
- Required tests are stdlib-only (`test_exact.py`), runnable with `python3 -B -E -S -m unittest discover -s experiments/zeta_fold_exploration -p 'test_exact*.py'`. The numerics are scripts plus skip-gated `test_numeric.py`.
- A new `experiments/` directory gets no CI automatically. Add none in the first PR. If the packet becomes maintained, add `.github/workflows/zeta-fold-exploration.yml` with a path filter on the packet directory and Python 3.12. It would have two jobs: the stdlib `-S` exact tests, and a pinned-dependency numeric smoke run under 2 minutes.

**Math- route (conditional on T-CUE).**
- Create `frontiers/cue_short_bar_<YYYYMMDD>/` containing:
  - `README.md` headed "Author-side proof candidate (Anthropic Claude). Scientific effect: NONE. Nonauthor review required.", with object id `CL-CUE-SHORTBAR-<date>-v1`;
  - `PROOF.md`;
  - `SOURCES.json` (schema 1, consumed/cited_only, sha256/git_blob, mutants) and `RESULTS.json`;
  - a stdlib checker (exact N = 2 closed form plus Fraction checks of the transfer constants) whose output is byte-identical under `-B -S` and `-B -O -S`, where mutants exit 1 and an unknown mutant exits 2;
  - its own path-filtered workflow.
- Compare the coefficients numerically in the main packet, not in Math-.

**Privacy.** The forwarded note is described as external. Do not commit it verbatim or name its author without the owner's agreement; refer to it as "an external note forwarded by the owner, October 2026".