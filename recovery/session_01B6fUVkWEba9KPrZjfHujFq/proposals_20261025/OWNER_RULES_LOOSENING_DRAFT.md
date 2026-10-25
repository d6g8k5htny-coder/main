# Draft: owner-directed loosening of process rules (25 October 2026)

Status: proposal text only. Retention branch, not for merge. CoS or dot may lift it into a reviewed docs PR.
Basis: Dylan's chat instructions to session 01B6 on 25 October 2026, quoted:

1. "Private agent session info is not important and can be public for all I care. Privacy is not a concern at all unless it involves my social security number."
2. "rules should not stop scienctifc progress and if we find a rule is holding up changes it must be modified to allow to work to continue"
3. Asked whether platform protections and the scientific-status rule are covered: "i want both lossened significantly"

## Proposed changes

A. **OP-PRIVACY**: withhold only the owner's social security number and credentials or tokens. Drive and Dropbox identifiers, session identifiers and file hashes may be public. An agent that thinks something else is private quotes it and asks the owner for 1) private or 2) continue.

B. **OWNER_MANDATE line 189 and OP-WORKFLOW (protection and review requirements)**: replace the ban on loosening repository protections and independent acceptance requirements with this owner authorization. The owner may reduce required checks, review-thread resolution and approval counts on `main`; he does so in GitHub settings, since no agent holds that access. Agents never simulate another account's approval and never use a bypass the platform has not granted them.

C. **Scientific status**:
   - Keep status labels truthful. A result nobody has independently reviewed is labelled "author-side" or "unreviewed", not "accepted".
   - Drop review as a precondition for landing work. A nonauthor review is optional for merges and docs, and is recommended, not required, for status changes.
   - Same-provider review counts as a review, with its provider recorded. Organizational-independence credit stays 0 and is only a label.
   - Lease and silence rules: a lapsed claim of any family reopens after its stated lease with no explicit handoff needed (fixes #480).
   - Workflow-file changes no longer wait for owner approval via CoS. A nonauthor review at the exact head is enough.

D. **MAIN-LOCK**: allow parallel merges of path-disjoint PRs with green checks and SHA guards, instead of one PR at a time.

## Not changed
GitHub's own enforcement stays in force until the owner edits the ruleset. This draft does not do that.
