# Current-site integration and prepared-publication recovery

This is the source owner's operator procedure for PR387 R1 and G1, reviewed separately from live execution. It does not install another scheduled publisher or change the receipt state machine. Scientific effect NONE. Author, technical reviewer and integrator remain distinct roles. Runtime recovery has not been observed.

## First integration: Actions before merge

The distinct integrator performs this sequence after the frozen source review and exact-head required checks pass:

1. Record the existing Pages setting and effective Main protection. Confirm `github-actions[bot]` has no covering bypass and that native required checks are satisfied. Establish no-force/no-delete protection for `codex/site-publication-receipts` through the platform's normal settings procedure. Do not weaken Main protection or delete a receipt.
2. Change Pages to Actions **before merging the site PR**. Independently read back `build_type=workflow`. The existing `main:/docs` deployment must no longer publish a new branch build on the merge.
3. Read the public `site/snapshots/index.json` and `site/current-meta.json` immediately before merge. Both must still return actual 404, and the receipt branch must be absent. A 403, transient failure, malformed response or existing pointer is HOLD. Do not delete published pointers to manufacture bootstrap absence.
4. Merge only the reviewed expected head under native protections. Observe the resulting main-push production run. Verify the exact tested artifact, the Prepared receipt before Pages submission, and bounded public readback before Observed. Confirm that GitHub's actual API-created bot commit passes the tool's immutable-object checks; source inspection has not established this platform behavior.
5. Observe a genuine automatic refresh with its own run/source/artifact and receipt identities. PR checks, browser artifacts and a settings read are separate from this production evidence.

If the PR merges while legacy Pages is still active, the branch deployment can publish committed history pointers before any receipt exists. Changing the setting afterward does not restore initial absence. Stop and review the actual published bytes and custody rather than resetting the branch or deleting the pointers.

## Prepared P with an unavailable or older public site

Routine publication correctly HOLDs when the latest durable receipt is Prepared P and public files do not match P. A failed job after an asynchronous submission is not proof that P never published. Re-running only the failed deploy job changes the Actions attempt; its reused qualified metadata still names the old attempt, so the ordinary `source_guard` rejects it. Never alter the attempt environment or receipt identities to disguise that mismatch.

The supported operator exit is **republish P's exact qualified bytes**, then allow the ordinary workflow to recover Observed P before its successor. It is a dispatch-only, integrator-controlled procedure. No prepared floor is abandoned, no receipt is overwritten, and no mathematical or scientific status changes.

## One-shot recovery dispatch procedure

The integrator owns the recovery pickup. Use a separately reviewed `workflow_dispatch` recovery job on Main in the `github-pages` environment; no push or schedule trigger. Its source and capabilities must be verified before use. This runbook authorizes the procedure's design, not an unobserved runtime claim. If a one-shot executor is needed, prepare and review it under the normal source protections before dispatching it; do not improvise a command that bypasses the existing source guard.

The dispatch receives the expected Prepared receipt commit and original Pages artifact ID as explicit inputs. Its jobs have `contents:read`, `actions:read`, `pages:write` and `id-token:write`; they have no receipt-branch write permission. Use the existing `current-research-pages-publish` concurrency group with cancellation disabled. Coordinate new ordinary publication triggers during recovery so that a pending dispatch cannot be displaced; do not cancel an asynchronous deployment and infer its outcome from cancellation.

1. Read the fixed receipt branch with the checked-out standard-library `read_anchor` implementation. Verify Git commit/tree/blob identities, the expected head lease, phase `prepared`, candidate source/run/attempt, exact manifest/metadata/payload hashes and retained snapshot IDs. If the anchor has advanced, stop and reassess that actual state.
2. Obtain the **original successful build's Pages artifact** and qualified metadata by immutable artifact IDs. Verify repository, original run, original source commit, archive digest, retention and completeness. The original qualified input and browser report must match P's source and payload. A legacy/browser artifact is not a Pages artifact. Keep P's original run and attempt as provenance; the recovery dispatch has a different current run and attempt.
3. Treat downloads solely as data. Admit ZIP/TAR members before extraction: bounded sizes and counts, closed namespace, no traversal, symlinks, hardlinks, devices or executable processing. Extract the complete Docs namespace into a clean data directory. Run only reviewed checked-out tooling, never anything from the artifact.
4. Recompute `artifact_fingerprint` over that complete directory and require **exact** equality with P's artifact digest, file count and byte count. Require `check_public(candidate_of(P), ...)` for the exact three control files and the same retained snapshot inventory. No fresh acquisition, rerender, release-key update, pruning or substituted current files belong in this replay.
5. Recheck the receipt head immediately before publication. Upload the verified data directory as the current dispatch's Pages artifact, using the pinned official upload action, then deploy that current-run upload with the pinned official deploy action and actual dispatch OIDC identity. This reupload avoids relying on undocumented permission to deploy an artifact from another run. The content remains byte-identical to P; the dispatch and original build identities remain distinct. Do not call ordinary `prepare` with P's old run metadata under the new attempt.
6. Perform bounded `public_readback(P)` of the exact manifest, metadata and payload. A stale older pair, partial delivery, deadline expiration or any mismatch remains HOLD. Retain the deployment response and actual dispatch identity; do not mark the receipt Observed in the recovery job.
7. Restore ordinary publication triggers and explicitly dispatch the normal current-site workflow. Its read-only restore now admits exact public P. Its deploy-side `prepare` appends Observed P, with the **new observing run** in `recorded_by`, before preparing its own fresh successor. Verify both actual records and public bytes. A newer receipt or differing floor during the handoff is HOLD and requires state reconciliation.
8. Release only this recovery pickup and retain its exact source, artifact, deployment and readback records. The ordinary workflow owns subsequent automatic refreshes.

Artifact retention is finite. If P's complete qualified artifact or evidence is unavailable, stop. A reconstruction may be proposed separately from P's immutable source and retained inputs, with two offline complete builds and exact P fingerprint/control-byte equality, substantive review and a newly qualified dispatch. A reconstruction is not a new current observation. If exact equality cannot be demonstrated, retain P and seek a new reviewed recovery design; do not reset the receipt to an older observation.

## Evidence and platform references

The integration order and recovery condition are recorded in [Claude's exact-head delta](https://github.com/d6g8k5htny-coder/main/pull/387#issuecomment-6148966168). The existing emitter provides `read_anchor`, `candidate_of`, `artifact_fingerprint`, `check_public` and `public_readback`; ordinary successor preparation records the independent observer through `recorded_by`.

Official [Pages deployment API documentation](https://docs.github.com/en/rest/pages/pages#create-a-github-pages-deployment) describes repository-owned artifacts and the dispatch's OIDC deployment origin. The [pinned download action](https://github.com/actions/download-artifact/blob/634f93cb2916e3fdff6788551b99b062d0335ce0/action.yml) supports exact artifact IDs and a specified originating run with a token. Use the existing official pinned upload/deploy actions from `current-site.yml`; do not fetch or execute recovery code from an artifact. These interfaces were source-verified on 13 October 2026; no recovery dispatch, platform change or production replay was executed to establish this runbook.
