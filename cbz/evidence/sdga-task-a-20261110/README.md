# Initial Type-valued fixture: retained native observation

Source: PR797 head b88b87fc9bb32dbc4deba999ca3712ef79747047, tree
a9bd64d7a66b61f7b459a3cf1654bb93e0a45dc6. Original fixture Safe.lean is
Git blob814c0f3169b4e967f5b4940d205a479effebeed1; Component is
ecb422db32b1825b71833e91120de3dbc59fd7a6 and pure Python372dab47.

Actual run42743801064/attempt1/job129590978474 tested signed-valid native
97132b3b32d0d291ac101620bb7f57ea6a7d3c65, ordered parents[Main812a799f,b88b87fc],
and the exact candidate tree. CPython3.11.18 and Lean release5045d005 were
observed. Safe and Component compilation exited0 in both modes; Component
compiler stdout/stderr were empty. Inventory exited0 in both modes and emitted
one module plus26declarations (29JSONL records including header/end).

Raw exported/server/private table counts are23/23/26. Eleven genuine compiler
extras, including flip._boxed and Switch elimination/constructor helpers, remain
present in the inventory. Task1 reported UNSUPPORTED_EXTRAS and BLOCKED/NONE;
it did not erase or admit those names. An ordinary unexported source def was
actually private-mangled. There were no plain-name hidden declarations.

Both native launchers then rejected the positive-fixture assumption before
the six methods could enter their controls:6failures/0errors/skips per mode,
all39subcases not_run. No semantic-negative or mutation credit follows.
The separate pure suite passed12methods per mode. Owned Python process cleanup
preflights succeeded, with no source drift or unconfirmed command cleanup.

These source-bound observations motivate a representative positive fixture;
they must not be relabelled as a successful six-method/39-case native run.
Kernel replay, fullSDGA99-module execution, fullIR and scientific admission
remain unestablished. scientific_effect NONE; CBZ green false.

## Exact printed originals

`initial-type-inventory.jsonl`:20086UTF8bytes, SHA256
a7fbc636c99e34078eb043c7879acd57c579c67c320c60363e16872f8143316b, Git blobbdbd7c8ff857bc4b6d21f377f3e7defb6dfea78f.

`initial-type-classification.json`:542UTF8bytes, SHA256
6fca6a81db1b9232d6515ce858378e6870b57efe16c50648de7948a68251825f, Git blobe1f6e490fb78423f16ebd54351381c6b6f795e30.

Root copied the complete original runner files printed in the decoded native
job log. Independently recomputed UTF8 byte counts and SHA256 match the printed
runner digests. Normal and optimized inventories had this identical digest.
The artifact ZIP13730181829 was inspected as metadata only; no member bytes
were read. This retained JSONL/classification copy is not a ZIP extraction.
