# Eigenfloor recovery boundary — 2026-09-25

Recovered from Drive as distinct proof/replay objects:

- `EIGENFLOOR_INDEPENDENT_SIDE24_EIGENFLOOR_FROM_KERNEL_INDEPENDENT_2026-08-02.md` — Drive `1_d3IlxTpBtRGhay2BnqOLgC5GwVqaS36`, raw 10,653 B, SHA-256 `4479cd6ba9764260971185eb53118b14847d86e88b699346d1242a3efbd0ccd6`, Git blob `0d108ea3accde093578adcf7ab1cc2d0ce2b8e50`.
- `EIGENFLOOR_INDEPENDENT_verify_side24_eigenfloors_from_kernel.py` — raw verifier remains on Drive at `1Tmj62Ny7_4ffev5gojO0Rpz9PbSD1vVK`, 18,701 B, SHA-256 `17209cb60d74c4caa4bc7988e1f1c0b8615c98835d5c9ee2ddf812e534ce2d4d`. It was materialized and replayed locally, but it is **not landed in Git here** because the connector text path did not preserve its bytes exactly.
- `SIDE24_V3_3_EIGENFLOOR_ARCHIVE_INDEPENDENT_REPLAY_AND_NEGATIVE_PROVENANCE_2026-08-02.md` — Drive `168VZ0zmLExmJ8iBCKaYWc-v5yC5rdtmo`, raw 9,468 B, SHA-256 `8f8106d7fa89cc08d112f8fb5fdbdd0d2902059cd5b17613b325ec57a9127f7e`, Git blob `06ace09ee60af4f58c22ebb023df2e294945ab8b`.

Fresh local replay of the raw 18,701-byte Drive verifier completed all **29/29** checks and printed `ALL_ASSERTIONS_PASS`. This receipt records the raw Drive identity above; Git source custody is claimed only for the two Markdown proof/replay notes whose blob IDs match the materialized bytes.

## What this closes

This supplies a citable, archive-independent **mathematical reconstruction** of the exposed contact/eigenfloor mechanisms used by the current SIDE24 proof chain.

## What remains intentionally unclosed

It is **not** the historical 12,388,862-byte `SIDE24_V3_3_V5_ADJUDICATION_CHECKPOINT_2026-08-01.zip` (expected SHA-256 `4040d654bddb1fc3b22eafa87bea7d0875166c03150d12602d6c1a0cb6385cfc`). It does not prove that every undisclosed row of the historical table archive has been byte-identically reconstructed. The historical archive/table identity remains a provenance gap unless those exact bytes are recovered.

Accordingly, a theorem reviewer may evaluate whether this independent derivation is mathematically sufficient to replace the carried historical table premise, but source custody must not silently claim the old table was recovered.
