# Parent-bound mutable evidence repair

Predecessor: Main PR #709 at 5d35ef0700ad88069549cd186d891a3d456c5d61.
Review: https://github.com/d6g8k5htny-coder/main/pull/709#discussion_r4418119091

The comment endpoint uses a global comment ID. A cited issue or pull URL also
asserts a parent thread. ContentObserver now requires the exact canonical API
issue_url for the cited repository and issue/PR number. GitHub PR conversation
comments use the issues parent API. Missing, wrong, noncanonical or malformed
parents are unknown, never fresh. API JSON now rejects duplicate names at every
nesting level, preserving the compatible strictness present in PR #700.

Two new methods first produced 42 intended assertion failures: 40 invalid-parent
cases across both URL forms and both supported repositories, plus two duplicate
JSON cases. After repair, 110 methods per normal/optimized mode passed with five
unchanged real-record skips (105 executed). All original assertions remain;
two old fixtures gained canonical parent fields so their original timestamp and
ID assertions continue exercising those conditions rather than parent rejection.

The existing nine source mutations remain. Two additional source mutations
remove parent binding and duplicate-JSON refusal. All eleven per mode were killed
by intended named assertions (22 total); positive designated baselines passed.
These finite engineering controls are not proof of auditor or mathematical soundness.

Reproduce using the commands in ../README.md. The raw logs and this directory's
SHA256SUMS bind the new source. Historical sibling logs, SHA256SUMS, REVIEW.md and
live-20261029 report retain their original bytes and predecessor scope; they are
not current-source verification. No new network observation is claimed. Earlier
strict live observation remains non-green (one changed cited directory).

No production record, projection, original RealRecordTests method, schema or
workflow changes. No full regression/kernel/CBZ or Q-9006 closure claim. Local
full browser-capable suite was not retried after its earlier sandbox denial;
current-source hosted qualification remains necessary and belongs to the parent.

PR #700 overlap: its GitSourceState API and tests are separate from ContentObserver.
Its current helper compares type+SHA but discards mode, resolves a head per source,
and has no captured-input --check or backwards-time refusal. Reconcile compatible
helper/test consumers during later composition; neither contribution is wholly
superseded. This repair does not overwrite that branch.
