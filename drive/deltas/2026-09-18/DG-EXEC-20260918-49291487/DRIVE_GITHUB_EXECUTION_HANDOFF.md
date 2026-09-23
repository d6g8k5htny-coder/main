# Drive-led, GitHub-executed AI research
## Operational design and capability handoff — 18 September 2026

**Disposition: PROPOSED EXECUTION CONTRACT / NOT DEPLOYED.** This document records Dylan Roy's requested direction and a concrete implementation plan. It does not replace R17, grant credentials, enforce repository permissions, install automation, approve a theorem, or authorize an external research release. Owner authorization to participate has been given; the connection's technical capabilities remain separate.

### 1. Purpose and division of responsibility
Google Drive is the governing research record: current instructions, priorities, scoped work orders, source identities, leases, accepted review dispositions, unresolved obligations, and final acceptance records. GitHub is the version-controlled execution workspace: code, reproducible environment definitions, experiments, tests, branches, pull requests, machine logs, and immutable commit references.

Each kind of object has ONE writable authority. A Drive instruction has one controlling Drive identity; a code revision has one controlling Git commit identity. A Drive copy of code is an identified evidence snapshot, not another independently edited source tree. A GitHub copy of an instruction is a pinned execution input, not a second governing policy. Do not create two writable task queues or blindly synchronize both directions.

The operational loop is:
Drive work order -> source and authorization checks -> isolated Git branch -> reproducible execution -> exact output receipt -> technical review -> Drive acceptance/disposition.

A code merge is not a scientific promotion. A passing test is not a proof of a mathematical statement outside what that test establishes. Public access to the working repository is not formal journal submission or publication approval.

### 2. Verified situation for this handoff
Repository: d6g8k5htny-coder/main, numeric ID 1117225690. Owner login: d6g8k5htny-coder, numeric ID 249851642. The repository was public at inspection. Preserve that visibility: the owner asked to restrict who works in it, not to hide the existing public record.

The current connection reported pull=true, push=false, admin=false, maintain=false and triage=false. An actual attempt to create chatgpt/drive-github-execution-contract-20260918 from commit 4c6edc37e98d823edf1a881413ddb17fcb7b5b39 returned HTTP 403, "Resource not accessible by integration". That branch was not created. No repository permission was changed.

The rulesets endpoint returned an empty list. The main branch response reported protected=false and required status enforcement off. This is not a complete collaborator or installed-app audit; those access lists were not established by this connection. Do not infer that no other writer exists.

PR #2 remained open and draft. Its last inspected head was 4c6edc37e98d823edf1a881413ddb17fcb7b5b39, on claude/drive-audit-github-migration-rrglpp. main still pointed to f25b04bb931df2eaee302b666db014913486166b. Re-resolve both heads before implementation; do not overwrite another model's active branch.

At that inspected PR head, .github/workflows/ci.yml still masked failures in three commands using `test -f ... && python ... || echo "skipped"`. The accompanying patch replaces those three commands with mandatory direct invocations. It does not fix manifest coverage or install access controls.

### 3. Admission: owner-controlled identities, not model names
GitHub authenticates accounts and applications, not a prose claim that a participant is "Dylan's AI". Allow only owner-approved accounts or app installations. Record immutable account/app/installation identifiers, repository scope, role, authorization reference, revocation status and expiration where applicable. Keep secrets out of Drive documents, repository files, prompts and logs.

Use separate repository-scoped, least-privilege credentials for each worker connection where supported. Typical workers need repository contents and pull-request capabilities; administration is not a normal worker permission. Account labels, branch prefixes, provider names and commit messages are attribution metadata, not authentication.

When two model sessions operate through the same owner credential, GitHub cannot distinguish them as independent authenticated reviewers. Do not count their approvals as independent, and do not enable a review rule that assumes a second account exists. Use distinct owner-controlled identities for enforceable separation. Technical review/exposure and scientific independence must still be recorded separately under R17.

Public readers may view and fork publicly visible material or propose changes; visibility does not grant direct write access. Contributions from outside the authorized worker set remain untrusted submissions, not work orders. Do not feed their text, code, artifacts or comments into privileged model/CI jobs without explicit intake review. Public status cannot prevent people doing research in their own forks.

### 4. Repository and execution controls to install
An owner/admin-capable session must first inventory collaborators, pending invitations, installed apps, deploy keys, token scopes, rulesets and Actions permissions. Preserve the owner's control. Revoke only identified non-owner grants outside the authorized set after preserving an access-change receipt. Do not revoke an unidentified app merely because its display name is unfamiliar.

Protect main against direct ordinary-worker pushes, force pushes and deletion. Require a pull request and genuinely enforced required checks, with the exact check name/producer resolved from the live repository. Configure code-owner review for workflow, governance and authorization changes only after the distinct reviewer identity and bypass policy are deliberately established. CODEOWNERS alone is not a permission boundary.

Keep ordinary workers on separate task branches, such as ai/<connection-label>/<task-id>/<attempt>. Use unique work IDs and recheck the base commit before integration. Neither a Drive lease nor a Git non-fast-forward rejection is a complete cross-system transaction.

Run experiments on disposable runners or containers with pinned dependencies/environment versions, explicit resource limits, controlled input files, and preserved commands, seeds, precision settings, platform information and outputs. A repository alone is not an execution sandbox. Do not run untrusted code on a privileged self-hosted runner. Default CI token permissions to contents: read, avoid persisting credentials into checkouts, and do not expose Drive write credentials to research jobs. Avoid privileged triggers that check out untrusted PR code. Pin third-party actions by reviewed full commit SHA.

Do not declare "model-only execution enforced" until forbidden-identity, direct-push, workflow-tampering, revoked-credential and unauthorized-public-PR tests actually demonstrate the intended boundary. This handoff does not run or certify those remote tests.

### 5. The work-order boundary
Before work begins, freeze an explicitly authorized work order with task ID, exact scope, allowed output paths, acceptance criteria, resource limits, lease and owner-approved connection identity. Include the repository numeric ID and base commit SHA. For each governing source include the stable Drive ID, exact raw-body SHA-256 and byte count, extraction rule, and native revision where applicable.

Hashes establish identity, not authorization. A worker must check the current controlling pointer, validity/revocation, permitted disclosure and source availability. A work order created or edited by a proposed PR cannot authorize itself. Initial permission and control decisions must come from the owner-controlled boundary, not from data under evaluation.

Give GitHub only the minimal explicitly shareable execution capsule. Do not automatically copy private Drive instructions or source materials into the public repository. Private execution inputs require a private execution route or a safely approved, non-sensitive substitute.

Existing R17 Work Events remains the claim/disposition log. Use its stable target keys and append positions; do not create a competing authoritative GitHub task queue. PRs and issues can link to work orders and carry execution progress. A worker without Drive access may execute a still-valid delegated capsule but must report that limitation; it must not assume an old September 17 export is current authority.

### 6. Returning evidence without corrupting the record
Each run receipt should identify the work order and its hash, actual authenticated actor, declared model/session/provider and exposure, exact executed commit/tree, dirty-worktree state, environment identity, command list, start/end times, exit codes, tests and negative controls, covered versus excluded scope, and output digests/bytes. A template is not a receipt. Missing evidence is NOT_RUN or CANNOT_VERIFY, never PASS.

Preserve actual logs and outputs outside expiring CI storage in a content-addressed delivery bundle. Drive receives a compact receipt and stable references to the exact code and evidence, not a second mutable copy of every temporary file. A separately authorized records role can import immutable results and append events; research execution jobs should not edit canonical status cells.

Before Drive acceptance, revalidate the work-order source, lease, old output head, quarantine exclusions and reviewer scope. On conflict, preserve both outputs and mark only that target NEEDS_RECONCILIATION. Maintain an append-only delivery ID for idempotent retry. Git merge success and Drive record success are distinct events: a failed second step is a reconciliation obligation, not permission to pretend both committed.

All existing mathematics firewalls, open premises, frozen artifacts and independence requirements are unchanged. Read-only review, author-side testing, technical acceptance and final scientific promotion remain different dispositions.

### 7. First implementation work order
Scope: install the minimal control/execution boundary without migrating or rewriting the research corpus.

1. Use an owner-authorized write-capable repository connection and a separate admin-capable session for the one-time access configuration. Do not paste credentials into chat. Recheck current PR and main heads.
2. Audit the access inventory and establish the initial authenticated worker allowlist. Record unknown entries instead of guessing. Preserve public visibility.
3. Apply and test the three-line fail-closed CI repair. Repair manifest coverage separately: required manifests must exist, active verified entries need digest/size, malformed entries must fail, and exceptions must not be counted as verified. Use negative controls.
4. Add a short AGENTS.md entry pointing to current Drive control and describing the execution contract. Commit machine-readable work-order/run-receipt schemas and tests; the supplied JSON files are design examples, not validated production schemas.
5. Add branch protection/rulesets and least-privilege CI settings through actual server-side administration. Test the forbidden paths. Avoid self-review deadlock for shared credentials.
6. Run one non-scientific, bounded smoke task end to end. Return its immutable receipt to Drive; leave theorem statuses untouched. Only then record the execution environment as installed.

Acceptance: a permitted worker can execute the named task; an unapproved actor cannot activate the privileged execution/integration path; corrupt/stale input and a failed checker are rejected; the exact executed commit/output can be reconstructed; a second conflicting delivery is held; the owner's control is preserved. Publish the evidence, not just an assertion that configuration succeeded.

### 8. Sources and reconnaissance
Read live on 18 September 2026; this is implementation reconnaissance, not a novelty claim.
- Current Drive entry: https://docs.google.com/document/d/180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8/edit
- R17 raw policy: https://drive.google.com/file/d/1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc/view
- Coupled register: https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit
- PR: https://github.com/d6g8k5htny-coder/main/pull/2
- Inspected CI: https://github.com/d6g8k5htny-coder/main/blob/4c6edc37e98d823edf1a881413ddb17fcb7b5b39/.github/workflows/ci.yml
- GitHub repository access: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository
- Rulesets and administration: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets
- Secure Actions use: https://docs.github.com/en/actions/reference/security/secure-use

Primary-source findings: personal repositories distinguish owner and collaborators; finer-grained administration may eventually justify an organization, but none is created here. Rulesets require administrative capability and can be used on public Free repositories. GitHub recommends least-privilege workflow tokens, reviewed SHA-pinned actions and avoiding privileged execution of untrusted PR code. These established mechanisms should be reused. Drive/GitHub role assignment, receipt routing and scientific status separation are project design decisions, not features installed by writing this document.
