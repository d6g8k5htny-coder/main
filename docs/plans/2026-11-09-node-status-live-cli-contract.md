# Q-9006 CLI source-context contract phase

This artifact stages offline tests only. The 23 helper controls and all
production records/schema/ownership remain unchanged. No CLI production code,
live workflow, token use, source disposition or 100% claim is included.

Target: tools/node_status_live_check.py after its separately qualified helper
phase. CLI interface: optional --root (default the executed tool's checkout)
and required --output (a new external JSON report file whose real parent exists).
The caller cannot select another checkout, even if its tree is identical.
There are no API/transcript/observer inputs.

Bind the actual selected Git commit and nine listed working files directly to
HEAD blobs, not status/stat shortcuts; clear ambient Git path redirects for
identity reads without changing credential/proxy configuration. The tested
merge commit is the checked commit when that is what Actions checked out.
An unchanged trusted entrypoint must refuse changed dependency bytes before
executing them. The complete host/interpreter and entrypoint remain trusted
bootstrap assumptions; this is not a hostile-host certifier.

Native context requires GITHUB_ACTIONS=true, repository
d6g8k5htny-coder/main, full lowercase immutable GITHUB_SHA matching actual HEAD,
and positive GITHUB_RUN_ID/GITHUB_RUN_ATTEMPT. Report actual sys.flags.optimize,
not an environment string. A supplied helper-unit identity is not CLI evidence.

Ordinary parse errors return 2 with argparse diagnostics. --help returns zero
without context, token, output creation or HTTP. Structured refusals are nonzero,
emit exactly one JSON object on stdout with no traceback, and use schema
cbz.node-status-live-cli.v1, status REFUSED, error_code, scientific_effect NONE
and cbz_green false. Error codes exercised by these contracts:
RUN_CONTEXT_INVALID, CHECKED_COMMIT_MISMATCH, ROOT_MISMATCH,
SOURCE_BYTES_CHANGED, OUTPUT_INVALID and AUTH_REQUIRED.

Do not create an output for invalid context/source/root/output. Never overwrite
an existing file or follow output/parent symlinks; never write into selected
source. With otherwise valid bound source/context/output but no GITHUB_TOKEN,
write the AUTH_REQUIRED refusal to the new output and stdout identically.
Include derived identity and source_bindings for each of the nine files,
with exact git_blob and sha256, but no source_observations/fresh_checked fields.
This is a source-bound refusal, not synthetic live qualification.

tests/test_node_status_live_cli.py creates owned local Git fixtures from actual
selected source bytes. A child harness denies urllib HTTP and socket connection
attempts and strips all token input. It tests real CLI behavior; it does not
mock a report or source-identity implementation. All current data edits are
explicit adverse temporary fixture changes. No real record or assertion is
rewritten.

Parent first publishes these tests with the unchanged helper-only source and
runs:
- python -B -S -m unittest tests.test_node_status_live_cli -v
- python -B -O -S -m unittest tests.test_node_status_live_cli -v

The helper-only file has no CLI, so current ordinary invocations return zero
without the required refusal/help behavior. Expect direct test-body failures,
not import/syntax/environment errors. Preserve that actual RED before
authorizing any CLI implementation. Later actual HTTPS qualification remains a
separate read-only G5 companion. The named dormant real-record test/history and
Q-9002 through Q-9005 remain explicit and unchanged.
