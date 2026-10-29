# Alignment review records

One Markdown record plus one JSON record per independent alignment review of
this package. The Markdown file carries the per-target verdicts and the
reviewer's exposure statement; the JSON file is the machine-validated record
described in [SCOPE.md](../SCOPE.md#review-contract) and validated by
`python3 tools/formal_gate_check.py --alignment <file>.json`.

Template: [`TEMPLATE_alignment_review.json`](TEMPLATE_alignment_review.json).
Reviewer procedure: [`../REVIEW_LANE.md`](../REVIEW_LANE.md).

No record other than the template exists at publication. The validator refuses
a record whose author and reviewer share a provider, family or agent, whose
manifest or scope digest is stale, or whose target list is not exactly the
manifest's. Passing the validator does not establish that the review happened;
a controller must retrieve and authenticate the referenced evidence.
