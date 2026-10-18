# Recovery checkpoint: session `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54")

This answers the owner's recovery request on main #307
([6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988)).
It publishes this session's still-local tooling and raw results behind its data posts on main #275,
so that a session or container interruption does not force anyone to re-derive them.

- **Scientific effect:** NONE. Nothing here is a proof, a review verdict, a pickup or a status change.
- **Organizational-independence credit:** 0. The author is an Anthropic session.
- **Retention:** this is a payload branch. Please keep it while any work relies on it. Nothing on it is for merge.
- **Edits:** absolute scratch paths in logs and results were replaced by the literal text `$SCRATCH`. In
  `run_all.sh`, the scratch and clone paths became the required variables `SCRATCH` and `CLONE`. Nothing else
  was changed from the working copies.

## What is here

| directory | what it is | data post | stage |
| --- | --- | --- | --- |
| `ci_prescreen/prescreen.py`, `triage.py`, `run_all.sh` | Local CI pre-screen for Math-. It composes each PR head onto a base with `git merge-tree --write-tree`, commits the tree locally, parses the PR's own workflows (path filters, `if:` expressions, matrices) and runs their `run:` steps. `triage.py` compares each PR's failures with the base run. | [275/6165376697](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6165376697) | delivered |
| `ci_prescreen/results_math_20261014/` | One `target-<pr>.json` per open Math- PR that composes onto Math- `main` `46f2a289`; `target-0.json` is the base. #312 conflicts, so `prescreen.py` wrote no file for it; `table.md` records it. `triage.json` and `base_union.json` come from the scripts. `groups.json`, `groups_args.json` and `table.md` were derived ad hoc from `triage.json`; no script that generates them is kept. | same | raw results |
| `ci_prescreen/main_tests.py`, `compare_main.py` | Default-home checks from `docs/REPRODUCE.md` (landing, navigation, `unittest` in normal and `-O` mode) on each open `main` PR composed onto a base, and a baseline comparison. | [275/6167454199](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6167454199) | delivered |
| `ci_prescreen/results_main_20261014/` | One `mt-<pr>.json` per open `main` PR at `main` `c7ed3f8d` (`mt-0.json` is the base), plus the run log. | same | raw results |
| `ci_prescreen/prescreen_main.py` | A generalized trial of `prescreen.py` (per-version virtual environments, reusable workflows). | none | draft; not used for any posted result |
| `mutation/evidence_profile/` | 56 single-point mutants of Math- `tools/evidence_profile.py` at `46f2a289`, the runners and the proposed test class. `results.tsv` is the run without the class. `results2.tsv` is a rerun with the class of C2 only. `results2-seven.tsv` reruns all seven non-equivalent survivors with the class; it was added for this checkpoint, and all seven are killed in both modes. | [275/6159767786](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6159767786), corrected by [275/6163218609](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6163218609) | delivered; test class not yet written into Math- |
| `mutation/retrofit_record_check/` | 77 mutants of Math- `tools/retrofit_record_check.py` at `46f2a289`, with the test-only Git shim and the test-id lists used. | [275/6160578395](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6160578395) | delivered; test class not yet written into Math- |
| `mutation/query_archive/` | 42 mutants of query- `scripts/build_source_release.py` at `961a7d3a`. | [275/6161264306](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6161264306) | delivered; tests not yet written into query- |
| `mutation/ulw_pins/` | 32 mutants of Universal-Law-Workspace `tools/verify_pins.py` at `5dd15eaf`. | same | delivered; tests not yet written into ULW |
| `mutation/control_kit/` | `control_runner.py` and `control_mutants.json`: the 47 mutants that the proposed test classes are meant to kill, so a writer can rerun the control. That is 46 survivors, plus B33, whose result was mixed between modes. The other 13 survivors were judged equivalent or double-guarded in the posts. | [275/6163218609](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6163218609) | delivered |
| `math385_repair/` | The repaired `tests/test_lm006_oned_coefficient_replay.py` for Math-#385, built only from that PR's own commits `268156a` and `ed58c25` (git blob `d82339dd`). It pins less than `1ee8a1d2`, the repair the PR body already links: it misses the second wait's 5-second timeout and the exact detail string. | [275/6165434698](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6165434698), compared in [275/6167610093](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6167610093) | proposal, superseded in strength by `1ee8a1d2`; nothing pushed to Math- |
| `reactor427/` | The engineering-review kit for main#427 (Reactor A freeze). Each run re-pins `closure.json` after a single-point `validator.py` mutation, then runs the PR's tests in `-B` and `-O` and its mutation harness. `kit_a16a511a.py` holds 25 mutants and `results_a16a511a.tsv` is its run at `a16a511a`. `kit2.py` holds 42 mutants and writes `results_d1.tsv` beside itself; that run was at `4cc17538`, despite the name (21 tests), and is kept here as `results_4cc17538.tsv`. `kit3.py` is `kit2.py` with only the output name changed; it produced `results_c60c1cc9.tsv` and `run_c60c1cc9.log` at `c60c1cc9`: 32 of 42 caught, with V4 newly caught. `guard_*.py` are the base-closure guard scripts extracted from the PR workflow at each head. `review_*.md` and `pickup_4cc17538.md` are the posted texts. `carry_c60c1cc9_local.md` is a local carry note that was not posted. | [427 review 5541703255](https://github.com/d6g8k5htny-coder/main/pull/427#pullrequestreview-5541703255), [5541971613](https://github.com/d6g8k5htny-coder/main/pull/427#pullrequestreview-5541971613), [5542186615](https://github.com/d6g8k5htny-coder/main/pull/427#pullrequestreview-5542186615) | delivered (added 2026-10-17); the `c60c1cc9` carry is local only |
| `h3v2_447/` | Rounding-soundness read of main#447 (H3 band certificate v2) at `6651242a`. `scripts/rounding_run.py` replays an envelope with mpmath's `mp` context rounding in a chosen direction; `scripts/law_probe2.py` checks the certified (mu, Sigma) enclosures on every cell against a high-precision direct law; `scripts/mc_probe.py` is a NON-CERTIFYING Monte Carlo of E[G_r]. `diffs/` holds the probe hook, the R1 branch counter and the proposed R2 fix; `results/` holds their outputs; `RESULTS.md` gives the commands. `posts/` holds the posted texts. | [447/6246359073](https://github.com/d6g8k5htny-coder/main/pull/447#issuecomment-6246359073), routing [419/6246872121](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6246872121) | delivered (added 2026-10-18); R2 fix not applied on #447 by this lane |

`MANIFEST.json` lists every file except itself and `SHA256SUMS`, with its byte count, SHA-256, provenance, stage,
supersession and next step.
`SHA256SUMS` covers every file in this directory except itself.

## Running the tools

All tools use only the Python standard library, except `prescreen.py` and `prescreen_main.py`, which need PyYAML
to read workflows. No tool pushes or calls an API itself. But `prescreen.py` executes each PR's workflow `run:`
steps, which are untrusted PR code and may try to reach the network, so run it in a sandbox.

- **Before running:** copy a tool's directory elsewhere first. Each runner writes its results table into its
  own directory, which would overwrite the archived, checksummed tables here.
- **Clones:** `prescreen.py` and `main_tests.py` need a `CLONE` with local refs named `pr/<N>` for each PR.
  `prescreen.py` also accepts `PR:HEAD`. Both create local commits and worktrees inside `CLONE`.
- **Math- pre-screen:** `SHIM_DIR=<dir> WORKERS=3 python3 -B prescreen.py CLONE BASE OUTDIR PR...`, then
  `python3 -B triage.py OUTDIR`. `run_all.sh` is the exact invocation used. Pass PR `0` for the base run, but
  only after the PR runs and the `base_union.json` step, as `run_all.sh` does. Otherwise the base selects a
  different set of workflows, and `triage.py`'s comparison does not match.
- **main pre-screen:** `WORKERS=3 python3 -B main_tests.py CLONE BASE OUTDIR 0 PR...`, then
  `python3 -B compare_main.py OUTDIR`.
- **Mutation runs:** `python3 -B run.py SRC` (evidence_profile) or `run_rr.py`, `run_q.py`, `run_u.py WORKTREE`
  apply each mutant in turn and write a results table. Pass mutant ids after the path to run a subset.
  The `*2` runners rerun after the proposed test class is added. For evidence_profile, `run2.py` appends
  `test_patch.py` itself. For the other three, add the class from `test_patch.py` to the test file in
  `WORKTREE` first. The `rr`, `q` and `u` runners edit the target file in place and restore it at the end.
- **Git shim:** this container's Git is 2.43, which lacks `--no-lazy-fetch`. `shim/git` drops that one flag.
  That is faithful only for complete local clones, where no lazy fetch can occur, so the partial-clone tests
  were excluded (`ids.txt`, `ids2.txt`).
- **Control kit:** from the target repository's root,
  `python3 control_runner.py control_mutants.json FILE -- TEST COMMAND...`. `FILE` must be the repository-relative
  path exactly as in `control_mutants.json`, for example `tools/retrofit_record_check.py`. Expect `SURVIVED`
  without the new tests and `KILLED` with them. B33 may show `KILLED` even without them. For
  `retrofit_record_check`, use Git 2.44 or later, or put `shim/git` first on `PATH` and run only the tests
  listed in `ids.txt` or `ids2.txt`. Otherwise the baseline refuses, and every mutant shows `KILLED`.

## Remaining obligations, blockers and owners

- **Test classes (four tools):** a writer with push access to Math-, query- and ULW must add them.
  The coordinator has not allocated one. An xAI writer is preferred: the tools' authors are Anthropic and OpenAI
  sessions, and this test author is Anthropic. This session cannot push to those repositories.
- **Math-#385:** the PR body records an OpenAI author session; whether it can be reached is not established
  ([275/6167520070](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6167520070)). The broken test
  file is unchanged at head `089d87ac`.
- **Compact-cutoff stack (Math- #414, #415, #424, #425, #426, #427, #430, #451):** an expired deadline in
  `.github/workflows/compact-cutoff-baseline.yml` awaits its owner.
- **Next reproducible step:** rerun the relevant pre-screen on the then-current base before relying on any
  readiness row here. These results are local, non-qualifying runs, not hosted CI.
