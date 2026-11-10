# Empty Pages deployment status adverse control

Read-only hosted job `129632294202` observed HTTP 200 with an empty deployment status for the prepared receipt. The recovery tool's contract requires explicit `deployment_cancelled`; the empty value is unknown, not evidence of cancellation. The observation and its platform/read-only limits are retained separately in `cbz/evidence/pages-prepared-status-20261110/`.

The existing nonterminal/unknown-deployment test now also supplies the observed empty string and checks that the exact prepared head remains and no writes occur. The recovery mutation campaign adds `deployment-empty-status`, which changes the real subject to admit that empty value. No production recovery code or mathematical content is changed.

On the reconstructed exact source identified in `manifest.json`, Python 3.12.14 ran 34 recovery/workflow controls without skips in each mode. A direct selected-test execution rejected the empty-status mutant with an assertion from the selected test body in each mode. The existing live-provenance recorder then ran 28 source and 10 workflow clean-baseline/selected-oracle mutation pairs per mode, rejecting three subject/helper/lifecycle assertion origins per mode. Both new mutant origin reports identify the selected test body, with one failure and no errors or skips. The full local campaign archive has SHA-256 `c2d6167543f901049d4f9b59ffeb6bbfab7fe787b741be69b1f6ce9759b7f7ae` and 254 byte-verified members.

Re-run on a complete checkout using fresh output directories:

```sh
python3 -B -S -m unittest -v tests.test_cbz_site_recovery.SiteRecoveryTests tests.test_cbz_site_recovery_workflow.RecoveryWorkflowTests
python3 -B -O -S -m unittest -v tests.test_cbz_site_recovery.SiteRecoveryTests tests.test_cbz_site_recovery_workflow.RecoveryWorkflowTests
CBZ_RECOVERY_NORMAL=$(mktemp -d)/mutations
CBZ_RECOVERY_OPTIMIZED=$(mktemp -d)/mutations
python3 -B -S tools/cbz_site_recovery_mutations.py --output "$CBZ_RECOVERY_NORMAL"
python3 -B -O -S tools/cbz_site_recovery_mutations.py --output "$CBZ_RECOVERY_OPTIMIZED"
```

These are local, source-bound engineering controls. Fresh hosted checks on the final branch composition remain necessary. No actual receipt recovery, deployment, scientific review or final G0–G11 result is implied.
