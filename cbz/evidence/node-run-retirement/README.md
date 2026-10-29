# Queued Node PR run retirement

Purpose: conserve hosted worker capacity by requesting cancellation only for queued, requested, pending or waiting non-control workflows belonging to exact superseded heads of main PR 707. Completed and in-progress runs, the current PR head, the pending diagnostic head 8aa, and every cbz-cold-start-node control run are outside the permitted set. No test or proof credit is awarded.

Source is .github/workflows/cbz-main-retire-superseded-runs.yml. It adapts the previously qualified main retirement workflow, limits action by PR number, exact head and workflow name, and runs 19 policy and receipt controls before calling the GitHub API. To execute, push a change to that workflow on branch cbz/node-retire-queued-20261029; the hosted token must have actions:write. Every candidate is re-read, the PR head is re-read immediately before cancellation, and each POST is followed by a native readback. The report is uploaded as cbz-superseded-runs-RUN_ID-ATTEMPT.

This is not an atomic lease: a queued run can start, or a PR head can change, between the last read and the cancellation POST. A cancellation request alone does not prove final cancellation. Preserve completed adverse-control runs and inspect the resulting artifact and fresh native status before reporting what happened. Maintenance owner: none; this one-shot contribution can be archived after the queue is reconciled.

Run 40954706379, job 123731659636, completed successfully after the 19 policy controls. It requested cancellation of four queued superseded PR707 runs. Native readback at 2026-10-29 21:08:00 UTC showed three completed/cancelled and one still queued; the fourth request remains unconfirmed. The complete report and decoded job log are retained beside this file.

A subsequent native readback at 2026-10-29 21:15:35 UTC observed run 40953041663 completed/cancelled; all four requested cancellations are now observed. The earlier queued readback remains in RESULTS.json as historical evidence.
