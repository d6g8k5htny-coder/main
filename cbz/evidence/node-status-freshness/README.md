# Q-9006 bounded engineering repair — candidate, not closure

Base Main c93cec70223281492709ff74ca84fed5dc6e40e8. No canonical B-003
surfaces/catalog, production records or published projection changed.

## Contract and limits

CLI `cited-object-content-v1` compares each pinned blob/tree's Git type, mode
and object identity with the same path at one HTTPS-observed default HEAD per
repository. Each nonrecursive API tree must identify the requested object,
explicitly be untruncated, and have valid unique entries. Missing paths,
malformed/partial responses, symlinks and gitlinks are unknown/non-green.
Directory identity includes the complete cited subtree. Immutable pins remain
unchanged; unchanged content at different snapshot commits remains fresh.
Mutable discussions require matching object identity and valid timestamps;
newer is stale, earlier than recorded is unknown. Transport failures remain unknown.

This trusts GitHub HTTPS/API provenance and Git object identity, not a formally
verified network stack or cryptographic replay attestation. Test injection is
not authenticated observation. Report start/end times bound separate Git and
mutable-comment observations; there is no atomic federation snapshot or claim
that heads cannot advance afterwards. The report records heads and individual
source observations; failures are included as refresh_error rows.

`--check` means **all currently observed cited sources are fresh**, not “every
record's declared status is truthful”. It validates untouched production schema
first, rejects empty/invalid/stale/unknown domains and cannot write. Report-only
exit0 remains explicitly non-certifying. Legacy github_fetch/revision-token
injection remains available for historical callers; the production CLI instead
uses ContentObserver. The previous test expecting an older observed comment
timestamp to count fresh was tightened to unknown, not removed or weakened.

No original RealRecordTests method is changed. Its five skips remain. In
particular the unconditional live-head test is not called closed. Records
remain unchanged pending genuine current bound observations and source review.
Actual prior read-only object comparisons showed eng.hard-gate's cited directory
changed, so an all-fresh check must fail until a deliberate truthful disposition;
repinning to hide this result is forbidden. This tool certifies no mathematics,
source ownership, kernel result, or whole-federation G5.

## Tool registry fragment for canonical integrator

- Tool: tools/node_status_refresh.py, existing report/refresh CLI plus
  ContentObserver and strict --check; purpose: eliminate snapshot-vs-last-touch
  false freshness and expose non-green source observation results (Q-9006/G8).
- Usage: `python -B -S tools/node_status_refresh.py records/node_status`
  (read-only report); append `--check` for strict all-fresh exit status.
  Existing `--write` is explicit and incompatible with --check; this delivery
  did not run it. Network reads use normal HTTPS trust, optionally GITHUB_TOKEN.
- Verification: commands below; owner none — self-contained, repository owned.
- Mutation experiment: run_mutations.py copies only needed source into a temporary
  directory; requires each unmodified designated baseline to pass; accepts only
  exit1 + intended named AssertionError, no error/import/syntax failure as kill.
  Nine distinct source mutants executed in both modes;18 intended assertion kills.
  This bounded engineering experiment is not a Layer-A auditor or proof of soundness.

## Reproduce

```
python -B -S -m unittest tests.test_node_status_freshness tests.test_node_status_refresh tests.test_living_project_validation
python -B -O -S -m unittest tests.test_node_status_freshness tests.test_node_status_refresh tests.test_living_project_validation
python -B -S cbz/evidence/node-status-freshness/run_mutations.py
```

Observed108 methods/mode:103 executed,5 existing real-record skips. Within that,
69 refresh/content methods execute,39 living-project methods include the5skips.
Initial RED includes8 explicit missing-feature assertions and unsupported--check
parser error; review RED separately reproduced future-time and schema-normalizing
false greens, both fixed. No claim that an argparse error is a mutation kill.
The broad project suite was not rerun in this bounded lane: previously denied
local browser execution must not be retried and parent owns hosted qualification.
No full G5, remote publish/merge, actual live CLI gate, record reconciliation,
or Q-9006 terminal closure is claimed.


Strict input custody: capture a bounded regular-file byte snapshot, validate and
observe exactly that snapshot in a temporary directory, report per-file SHA256,
and reject any final filename/byte drift in the original directory. An actual
adversarial-write RED during head fetch was reproduced then fixed; the9th mutant
removes that refusal. This binds the observed captured records, not future edits,
and assumes the running tool/validator/schema/graph/OS are trusted and not being
maliciously replaced during execution. Snapshot capture is not an atomic filesystem
transaction; cryptographic filesystem adversary resistance is not claimed.
