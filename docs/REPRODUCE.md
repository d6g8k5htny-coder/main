# Reproduce a result

[Home](../README.md) · [Research guide](RESEARCH_INDEX.md) · [Workspace](WORKSPACE.md)

## Start with one exact calculation

The [reader example](site/reproduce.html) runs the SIDE24 coefficient calculation
from one pinned Math- checkout with Python’s standard library. It provides the
commands, expected enclosures and their precise scope. No second repository is
needed for that example. The choices below are the deeper contributor routes.

## Choose the checkout

The default main repository contains the home and navigation. Current mathematical packages live in Math-. The larger numerical research tree remains on its named hardening branch. A check of one is not a check of the others.

Both source repositories are public; these checkouts require no GitHub account. Choose new destination directories:

```sh
git clone --single-branch --branch main https://github.com/d6g8k5htny-coder/Math-.git Math-
git clone --single-branch --branch chatgpt/drive-github-hardening-20260919 https://github.com/d6g8k5htny-coder/main.git research
```

The named branch tracks its current tip. To inspect one work item using that
checkout's existing dispatcher:

```sh
cd research
git rev-parse HEAD
python -B -S engine/next_action.py --lane A5
```

That command was run successfully against source commit `2f7a5a9f10c9ed5f5b7792a8f2521318d9208532`. It reads the declared work item and prints its scope and unresolved obligations; it does not prove or close the RN uniform lemma. The [dispatcher source](https://github.com/d6g8k5htny-coder/main/blob/2f7a5a9f10c9ed5f5b7792a8f2521318d9208532/engine/next_action.py) and [engine guide](https://github.com/d6g8k5htny-coder/main/blob/2f7a5a9f10c9ed5f5b7792a8f2521318d9208532/engine/README.md) can also be opened directly without cloning.

To repeat that historical run, use the fresh `research` clone above, select the
exact commit, and run the same dispatcher command. This leaves the clone detached
at the historical source; it does not test the current branch tip:

```sh
git checkout --detach 2f7a5a9f10c9ed5f5b7792a8f2521318d9208532
python -B -S engine/next_action.py --lane A5
```

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

Use Python 3.11+ for the query package. The catalog refers to six public
repositories: `Math-`, `meta-framework`, `query-`, `google-drive`, `trial`, and
`governance-`. For local working-tree verification, create the five siblings below
alongside the `Math-` clone above, using new destination directories. Run these
commands from their common parent, outside the `research` directory:

```sh
git clone https://github.com/d6g8k5htny-coder/meta-framework.git meta-framework
git clone https://github.com/d6g8k5htny-coder/query-.git query-
git clone https://github.com/d6g8k5htny-coder/google-drive.git google-drive
git clone https://github.com/d6g8k5htny-coder/trial.git trial
git clone https://github.com/d6g8k5htny-coder/governance-.git governance-
```

Lookup and offline verification use the existing query command:

```sh
python -B -S query-/research_query.py --registry meta-framework/registry.json
python -B -S query-/research_query.py --registry meta-framework/registry.json --key side24-coefficient
python -B -S query-/research_query.py --registry meta-framework/registry.json --key p15-full-price
python -B -S query-/research_query.py --registry meta-framework/registry.json --verify --workspace .
```

The first command lists keys actually available in that catalog. Proposed entries are not available until integrated. The catalog records source identity, not currentness or theorem acceptance; it is not a complete Drive inventory. The query command has no network access or retrieved-code execution.

`--verify --workspace .` checks the **working-tree files** at each catalog path
for safe location, regular-file status, byte count and SHA-256. It does not check
Git HEAD or fetch the recorded commits. It stops at the first mismatch, writes
`REFUSED: ...` to stderr and exits 2; a refusal is not an all-entry drift report.
Default-branch clones can therefore legitimately refuse verification when a
cataloged file has changed. Keep historical catalog identities intact; checking
out one repository commit cannot necessarily supply files pinned at several
different commits.

### Replay the pinned public bytes

This route needs only the fresh `meta-framework` and `query-` clones, Python
3.11+, Git and HTTPS access to public GitHub. It requires no GitHub account or
repository write permission. From the common parent, select this reproducible
catalog/tool cut in those fresh clones:

```sh
git -C meta-framework checkout --detach 75685db7eafa6c459e085dad223aab172cfa4da2
git -C query- checkout --detach 91722844f745ea96552a66a50ddf63ea59a0db9b
git -C meta-framework rev-parse HEAD
git -C query- rev-parse HEAD
python --version
```

The following command validates the catalog with the existing query library,
downloads each artifact at its recorded commit into a **new temporary directory**,
and runs the existing offline verifier there. Downloads are data only: none of
the downloaded payloads are imported or executed, and source checkouts are not
overwritten. Its public-repository, count and total-byte bounds follow the
[existing catalog workflow](https://github.com/d6g8k5htny-coder/meta-framework/blob/75685db7eafa6c459e085dad223aab172cfa4da2/.github/workflows/catalog.yml).

```sh
python -B -S - <<'PY'
import hashlib, json, pathlib, sys, tempfile, urllib.parse, urllib.request

sys.path.insert(0, str(pathlib.Path('query-/src').resolve(strict=True)))
from universal_law_query.catalog import load_catalog, verify

catalog = pathlib.Path('meta-framework/registry.json')
data = load_catalog(catalog)
rows = data['artifacts']
allowed = {'Math-', 'meta-framework', 'query-', 'google-drive', 'trial', 'governance-'}
if not 1 <= len(rows) <= 600 or sum(r['bytes'] for r in rows) > 2000000:
    raise ValueError('catalog outside this bounded replay')
identities = {}
for r in rows:
    if r['repository'] not in allowed or r['bytes'] > 1000000:
        raise ValueError('artifact outside this bounded replay: ' + r['key'])
    key = (r['repository'], r['path'])
    identity = (r['bytes'], r['sha256'])
    if key in identities and identities[key] != identity:
        raise ValueError('conflicting identities at one workspace path: ' + r['key'])
    identities[key] = identity

root = pathlib.Path(tempfile.mkdtemp(prefix='public-catalog-'))
print('Scratch workspace:', root, flush=True)
for r in rows:
    path = urllib.parse.quote(r['path'], safe='/')
    url = ('https://raw.githubusercontent.com/d6g8k5htny-coder/'
           + r['repository'] + '/' + r['commit'] + '/' + path)
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(r['bytes'] + 1)
    if len(raw) != r['bytes'] or hashlib.sha256(raw).hexdigest() != r['sha256']:
        raise ValueError('source mismatch: ' + r['key'])
    dest = root / r['repository'] / r['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)

checked = verify(data, root)
receipt = dict(checked, artifact_count=len(rows), python=sys.version,
               catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
               private_sources_fetched=False)
(root / 'VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print('ALL_DECLARED_PUBLIC_ARTIFACTS_VERIFIED', len(checked['verified']))
print('Receipt:', root / 'VERIFIED.json')
PY
```

At the pinned cut, success reports **592** verified entries. Preserve the printed
workspace and `VERIFIED.json` with the catalog and tool commit IDs. A download or
identity failure exits nonzero and leaves partial scratch data for inspection;
it is not a successful replay. This command verifies catalog payload identity
only, not currentness, federation tests or mathematical acceptance. The separate
[hosted catalog workflow](https://github.com/d6g8k5htny-coder/meta-framework/actions/workflows/catalog.yml)
also runs its own pinned federation controls; its execution evidence is a separate
receipt. Do not equate this local data-only replay with that hosted run.

[trial's original federation replay](https://github.com/d6g8k5htny-coder/trial/blob/main/federation/replay.py) intentionally fetches its original immutable coefficient-era sources. A successful historical replay does not test later mathematics. A skipped federation fixture in a trial-only checkout is not a successful cross-repository run.

## Default-home navigation checks

From this main repository's default checkout:

```sh
python -B -S tools/workspace_landing_check.py
python -B -S tools/navigation_check.py
python -B -S -m unittest discover -s tests -p 'test_*.py' -v
python -B -O -S -m unittest discover -s tests -p 'test_*.py' -v
```

The landing checker verifies declared local links and two historical byte identities. The navigation checker covers declared inline links and ATX/custom heading fragments, not arbitrary Markdown or every repository path. With network access, explicitly request the public proof-byte checks declared in [NAVIGATION.json](NAVIGATION.json) (`public_targets`; nine at this guide's revision):

```sh
python -B -S tools/navigation_check.py --verify-public
```

This bounded readback does not crawl every external URL, inspect private sandbox material or prove mathematical correctness. A changed legitimate source also needs a new declared identity. The navigation Actions workflow retains actual logs and runs on pull requests/manual invocation; it is not another autonomous research scheduler.

## Legacy numerical work

Use the [legacy execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md) in the hardening checkout. Its RN/24-jet sources, commands and predicates remain separate. Do not apply a main-cut patch onto that different tree or present navigation CI as full research CI.

## Interpret the evidence

Retain exact source commits, interpreter versions, baseline and mutation logs. Finite tests do not verify Gaussian continuum estimates or provide independent analytic review. Read hypotheses and source-bound mathematical arguments before composing results.
