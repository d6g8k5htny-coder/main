PASS_SCOPED_CORE_CORRECTIONS — all three original findings resolved at the exact reviewed hashes.

- Collector SHA256: `697f87e9184728b90caaf50e2fe9a71fbb950719c6d325987f114271c343c314`
- Core tests SHA256: `b3d94e36111154fd255193c118d87128e39aca2aea1dcd8f5cc2f72837743844`

Both files matched the supplied hashes before and after execution. This binds working-file bytes;
it does not assign them an unobserved new native commit or hosted execution.

WGI-01 (P1): corrected. Separate environment key/value transport preserves `=` in config
subsections; status verifies suppression and refuses unsafe discovery/transport. The unchanged
original fixtures observe zero filter execution in inventory and CLI for all five names, while
all five native positive controls execute. The core regression additionally exercises the
parent/submodule pair and refusal paths.

WGI-02 (P1): corrected. Exact `--resolve-git-dir` validation replaces the filename-only metadata
heuristic. The original nested repository is found through both API and CLI. The added core
regression covers deeper objects/assets directories and their symlink, with existing metadata
exclusion tests retained.

WGI-03 (P2): corrected. Field-scoped redaction preserves the two query-shaped filenames and
newline/tab/nonUTF8 bytes, including CLI JSON roundtrip. The added test retains filesystem
identities across roots, repos, worktrees, links and error targets while redacting URL and
diagnostic credential values.

Actual reviewer execution: **14/14 core tests normal,14/14 optimized**, zero failures/errors/skips;
**all seven unmodified original fixture replays** complete, CLI exit0, output0600. Original
fixture script was copied byte-for-byte into this new directory; original evidence was not
modified. Its source-label header is historical; the unchanged import path reads the collector
bound above. The raw results, logs and exact script hash are retained here. Fixtures cleaned.

No new concrete defect was established within this bounded correction scope. This is a scoped
engineering result, not a universal configuration/isolation proof or an integration approval.
Changing companion lifecycle code was excluded; parent3.11 and native CI results are not claimed
as reviewer executions. No global audit, installation, source/ref/PR/canonical-doc mutation or
mathematical/independent-family credit.

Actual reviewer: OpenAI Codex, separate nonauthor, same provider/source-test exposed,
organizational independence0, science NONE. Root684/B003 keeps serial integration. Only this
core correction-review unit is CLOSED and released; prior roles remain closed.
