# Engineering hard-gate source refresh

At `2026-11-10T04:01:01Z`, separate GitHub API reads observed Math default `9bdfc54b76173b57912503bfcf901a5f9d43b09c`. The prior stale node record is preserved byte-for-byte as `prior-record.json`. Its former source `dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` has directory tree `715449bbe16b471f68c4913d08e72baac903582b`; the observed Math default has tree `132725157175a85dbbd5a1cfc2c668fc7b518283`.

Across the 13 top-level directory entries, only `SOURCE_FILES.json` and `test_transition_contract.py` differ in Git identity. The manifest's only changed entry identifies the test file. The exact old and new test texts differ only by two additions of `git config maintenance.auto false` in temporary repositories. Both manifests and test texts are retained; their filenames ending in `.txt` are evidence, not additional executable tests. `observation.json` records exact Git object identities and the non-atomic scope.

Math run `42714668178` on `9bdfc54b76173b57912503bfcf901a5f9d43b09c` completed its downstream job `129492972799`, source checks and refusal controls successfully; the decoded full job log is retained. Its artifact `13717143202` had reported SHA-256 `ba56de446461de8a425c9153bf2ef938e76917ef6a140862e290344f61646cd5`; this artifact was not independently unpacked here. This is not whole-federation or scientific evidence.

The new Main record is a candidate based on the separately timed observation. Existing hosted node-status live checks, validation, required checks and fresh ref guards must be observed before integration. No prior stale outcome is erased.

## Preserved initial Main contract failure

At tested merge `d1e2d717522b65359a6c94569c35e110585b611a`, job
`129621544420` ran 23 live-contract tests in each Python mode and failed
two in each because the original expectations named three fresh nodes while
the candidate has four. Its dependent live-freshness job `129621764656`
was skipped. The 20 CLI contract tests passed per mode. The exact prior test
source and full decoded job log are retained here; neither the failing outcome
nor the skipped live job is credited as success.

The correction retains the explicit cohort check, updates it to four named
fresh records and twelve source observations, and extends existing
changed-object and unavailable-source refusal tests to both hard-gate source
URLs. Fresh hosted results for the corrected source are required.
