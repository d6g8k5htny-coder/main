# Main superseded-run retirement

Prepared 2026-10-29 20:53:27 UTC. This one-shot administrative workflow is limited to open main PRs #704 and #705. It rechecks run status and PR head immediately before a cancellation request, retains all current-head, ambiguous and other-PR runs, writes a pending record before every request, and records the observed response. Terminal races are recorded as already terminal; an active refusal remains a failure. Original failing or incomplete workflow evidence remains historical and no test is deleted or disabled.

The code is adapted from Math source db46797a430495c943fc45867e77d6cffc6baee2, whose hosted run 40951530009 passed 10 selection and action-order self-tests and produced a complete 74-row report. The adaptation adds an exact two-PR allowlist and three scope tests. Its own hosted run is pending; no cancellation or mathematical gate result is inferred.

Usage: push cbz/main-retire-superseded-20261029 after changing the workflow. Full JSON decisions are uploaded from the actual run. This is not a proof or regression certificate. Owner none; criteria 5, 9, 11.
