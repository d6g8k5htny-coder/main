# Public intake and complete engineering-path discovery

`public_intake_check.py` runs as trusted default-branch code for `pull_request_target`. It reads submitted paths and API bytes without checking out or executing the submitted branch. Its `maintainer-engineering` result is a routing outcome with scientific effect `NONE`; it does not verify an engineering change's contents or certify mathematics.

The old 500-file `snapshot()` limit ran before lane discovery. That rejected a same-repository, 1,458-file engineering/evidence contribution before the checker could inspect its paths. The repair separates complete path discovery from the unchanged public-intake limits:

| Boundary | Limit |
| --- | ---: |
| Complete changed-path discovery | GitHub API ceiling of 3,000 files |
| Public intake submission | 500 changed files |
| One public package | 50 files |
| One public file | 262,144 bytes |
| One public package's bytes | 2,097,152 bytes |
| One API response | 8,388,608 bytes |

A fork or a `results-for-review` label identifies intake before pagination, so a submission over 500 paths still rejects before file reads. Other same-repository PRs require every expected page, with an exact page length, exact total count and unique changed paths. An `incoming` path or a rename from `incoming` on any page selects intake, with the original 500-file limit checked before package validation. The 50-file package and all byte, source, artifact and add-only controls are unchanged. A final API snapshot must match the original head, base, repository, labels and count. Counts over 3,000, malformed counts or incomplete discovery fail closed. The public `snapshot(pr, repo)` API keeps its original 500-file default.

Run the no-network focused suite from the repository root:

```sh
python -B -S -m unittest discover -s tests -p 'test_public_intake*.py'
python -B -O -S -m unittest discover -s tests -p 'test_public_intake*.py'
```

The separate `Public intake checker controls` workflow now uses the same wildcard suite and watches all `test_public_intake*.py` files. This includes the new 19 discovery controls and the existing four artifact-identity controls; previously its exact test-file selection ran only the original 85 controls. The trusted `pull_request_target` workflow remains unchanged.

Observed at this repair: the original 89 tests and trusted `public-intake.yml` workflow remain byte-identical; 19 new tests reproduced 13 assertion failures before the fix, then all 108 tests passed in each mode. Fourteen actual source guard mutants were rejected in each mode by the separate verified synchronous lifecycle classifier, with source hashes and logs retained in the contribution evidence. An independent engineering reviewer repeated both focused runs and found no unresolved routing defect. This does not claim hosted CI completion or a full federation gate run.

`public_intake_discovery.registry.json` records the modified tool, exact self-test commands and narrowly scoped observed verification. Canonical governance/status admission remains the existing federation integrator's single-writer operation. Because the workflow reads trusted default-branch code, the repair must land before a later PR event can evaluate the large contribution with this checker; rerunning an old event may retain its old trusted source.
