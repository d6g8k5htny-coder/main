# Historical foundation and archive custody

This is source custody and a non-admitting comparison, not a proof catalogue
for the current federation. The original #695 candidate is closed and unmerged.
Its 184 historical rows and all source text remain available in
`catalog.json.gz`; decompression yields the exact Git blob
`892466e68f90a584e214f60c86c7c493d4f9c42e`, SHA-256
`6fb095df59726f6ee13b265dd88440e1a5f0c7ca7a5025643fd1bbe9acab3ec7`,
from Main commit `82d9e480a51b946b3e74525ccf32139129d37d28`. The source
catalog's adoption wording and old source cuts are preserved as historical
content, not imported as current mathematical dispositions.

The three named Q-8336, Q-8340 and Q-8429 archive tags peel to their original
unmerged PR heads. `archive-sources.tar.gz` retains all 87 explicitly named
source files with original bytes; `archive-sources-manifest.json` identifies
all 72 unique Git blobs, their paths, sizes and SHA-256 values. The annotated
tag objects and historical review text are retained in `raw-context.tar.gz`.
Their preservation does not establish new review independence or acceptance.
Two references to a README row lack an exact path in the historical catalog;
they remain unresolved. Other historical files outside the three named source
subtrees are not implied by this archive.

The exact scope, observed changed and unchanged source entries, missing
structured-location coverage, current limits, prior leaf-only attempt and
failed native fetch are in [RECOVERY_OBSERVATION.md](RECOVERY_OBSERVATION.md).
The compressed comparison keeps every row's current admission `UNASSESSED`.
Historical one-off analysis scripts in `raw-context.tar.gz` retain their original
scratch paths; they are evidence, not a portable or admitted certifier.
It does not prove all eleven fields semantically valid, fixture isolation,
complete proof coverage, current kernel or axiom results, or CBZ-0. The old
and current Q-3001 and Q-3002 bytes and their diffs are in `raw-context.tar.gz`;
the former's newer record limits its closure to the 66 exact conditional
companions and does not close the parent applications.

From the Main repository root, the ordinary local integrity and recovery
commands are:

```sh
sha256sum -c cbz/evidence/foundation-custody-20261110/SHA256SUMS
gzip -cd cbz/evidence/foundation-custody-20261110/catalog.json.gz | git hash-object --stdin
tar -tzf cbz/evidence/foundation-custody-20261110/archive-sources.tar.gz
# Extract only to a new directory outside the checkout when the bytes are needed.
```

The catalog hash command must print the Git blob named above. Archive member
names were checked as regular files with no parent traversal; the independent
source-tree/hash readback is retained in `raw-context.tar.gz`. These commands
check custody only. They are not the final G0–G11 pass or scientific review.

A local independent readback checked the decompressed catalog's Git identity,
all 87 archive member bytes against both Git and SHA-256 identities, all three
annotated tag-object bytes, and both comparison JSONs. It then ran the actual
`sha256sum -c` on a clean copy, a one-byte-corrupted copy and a missing file:
the respective exits were 0, 1 and 1. `verification.log` and the exact
one-off script are retained; that script uses the original scratch location
and is evidence rather than a portable gate.
