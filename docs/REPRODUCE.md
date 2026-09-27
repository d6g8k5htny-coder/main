# Reproduce a result

[Home](../README.md) · [Research guide](RESEARCH_INDEX.md) · [Workspace](WORKSPACE.md)

## Choose the checkout

The default main repository contains the home and navigation. Current mathematical packages live in Math-. The larger numerical research tree remains on its named hardening branch. A check of one is not a check of the others.

Both source repositories are public; these checkouts require no GitHub account. Choose new destination directories:

```sh
git clone --single-branch --branch main https://github.com/d6g8k5htny-coder/Math-.git Math-
git clone --single-branch --branch chatgpt/drive-github-hardening-20260919 https://github.com/d6g8k5htny-coder/main.git research
```

To inspect one work item using the research checkout's existing dispatcher:

```sh
cd research
git rev-parse HEAD
python -B -S engine/next_action.py --lane A5
```

That command was run successfully against source commit `2f7a5a9f10c9ed5f5b7792a8f2521318d9208532`. It reads the declared work item and prints its scope and unresolved obligations; it does not prove or close the RN uniform lemma. The [dispatcher source](https://github.com/d6g8k5htny-coder/main/blob/2f7a5a9f10c9ed5f5b7792a8f2521318d9208532/engine/next_action.py) and [engine guide](https://github.com/d6g8k5htny-coder/main/blob/2f7a5a9f10c9ed5f5b7792a8f2521318d9208532/engine/README.md) can also be opened directly without cloning.

The historical complexity-framework README advertises a different package layout. The current research engine uses the named branch and paths above; its execution entry point is `engine/run.py`.

## Mathematics

In a Math- checkout first record `git rev-parse HEAD` and `python --version`. These packages use the Python standard library.

```sh
python -B -S -m unittest discover -s coefficients/side24_v1 -p 'test_*.py' -v
python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v
python -B -S -m unittest discover -s frontiers/price_budget_20260924 -p 'test_*.py' -v
python -B -S -m unittest discover -s frontiers/full_price_20260924 -p 'test_*.py' -v
python -B -S coefficients/side24_v1/coefficient.py
python -B -S frontiers/price_budget_20260924/price_budget.py
python -B -S frontiers/full_price_20260924/full_price.py
```

At the named September 24 sources these groups contain 30, 57, 26 and 36 distinct tests respectively, 149 total. Record actual results after later source changes. Repeat with `-O` to test optimized Python. Repeated modes do not increase distinct test counts.

## Deliberate error controls

Use a new absolute output directory outside the source tree each time:

```sh
python -B -S frontiers/price_budget_20260924/run_validation.py --output /tmp/price-budget-run-001
python -B -S frontiers/three_fronts_20260924/run_validation.py --output /tmp/three-fronts-run-001
python -B -S frontiers/full_price_20260924/run_validation.py --output /tmp/full-price-run-001
```

The price-budget runner covers 26 tests and five semantic mutations in both modes. The three-front runner keeps its 54-test core and ten mutations; the discovery command above includes its three supplementary price-boundary tests. The full-price runner covers 36 tests and seven mutations; `--mode normal` or `--mode optimized` permits separately bounded runs. A deliberately faulty variant must not hide a failed unmodified baseline.

## Source lookup and cross-repository replay

With sibling Math-, meta-framework, query- and google-drive checkouts, run from their common parent:

```sh
python -B -S query-/research_query.py --registry meta-framework/registry.json
python -B -S query-/research_query.py --registry meta-framework/registry.json --key side24-coefficient
python -B -S query-/research_query.py --registry meta-framework/registry.json --key p15-full-price
python -B -S query-/research_query.py --registry meta-framework/registry.json --verify --workspace .
```

The first command lists keys actually available in that catalog. Proposed entries are not available until integrated. The catalog records source identity, not currentness or theorem acceptance; it is not a complete Drive inventory. The query command has no network access or retrieved-code execution.

[trial's original federation replay](https://github.com/d6g8k5htny-coder/trial/blob/main/federation/replay.py) intentionally fetches its original immutable coefficient-era sources. A successful historical replay does not test later mathematics. A skipped federation fixture in a trial-only checkout is not a successful cross-repository run.

## Default-home navigation checks

From this main repository's default checkout:

```sh
python -B -S tools/workspace_landing_check.py
python -B -S tools/navigation_check.py
python -B -S -m unittest discover -s tests -p 'test_*.py' -v
python -B -O -S -m unittest discover -s tests -p 'test_*.py' -v
```

The landing checker verifies declared local links and two historical byte identities. The navigation checker covers declared inline links and ATX/custom heading fragments, not arbitrary Markdown or every repository path. With network access, explicitly request the seven named public proof-byte checks:

```sh
python -B -S tools/navigation_check.py --verify-public
```

This bounded readback does not crawl every external URL, inspect private sandbox material or prove mathematical correctness. A changed legitimate source also needs a new declared identity. The navigation Actions workflow retains actual logs and runs on pull requests/manual invocation; it is not another autonomous research scheduler.

## Legacy numerical work

Use the [legacy execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md) in the hardening checkout. Its RN/24-jet sources, commands and predicates remain separate. Do not apply a main-cut patch onto that different tree or present navigation CI as full research CI.

## Interpret the evidence

Retain exact source commits, interpreter versions, baseline and mutation logs. Finite tests do not verify Gaussian continuum estimates or provide independent analytic review. Read hypotheses and source-bound mathematical arguments before composing results.
