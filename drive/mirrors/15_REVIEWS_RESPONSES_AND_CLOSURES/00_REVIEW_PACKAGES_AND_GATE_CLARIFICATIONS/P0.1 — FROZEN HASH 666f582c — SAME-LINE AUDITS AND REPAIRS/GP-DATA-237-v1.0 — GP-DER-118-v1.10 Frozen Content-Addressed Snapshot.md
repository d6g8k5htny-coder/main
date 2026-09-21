# GP-DATA-237-v1.0 — GP-DER-118-v1.10 frozen content-addressed snapshot

Artifact ID: GP-DATA-237-v1.0
Date: 2026-08-02
Class: DATA — raw frozen-object identity and roundtrip receipt
Canonical impact: mechanical identity only

Source object: GP-DER-118-v1.10
Drive ID: `1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9`
Raw file size: 30,933 bytes
Raw whole-file SHA-256: `c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564`

Frozen-body extraction:

- require exactly one line `BEGIN_FROZEN_THEOREM_BODY`;
- require exactly one line `END_FROZEN_THEOREM_BODY`;
- exclude both marker lines;
- preserve body bytes with LF line endings;
- bind exactly one trailing LF.

Frozen-body size: **29,293 bytes**
Frozen-body SHA-256: **`9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014`**

Drive upload readback: PASS. The downloaded Drive base64 and local source base64 were equal byte-for-byte (41,244 base64 characters; 30,933 raw bytes).

Verifier: `verify_gp_der_118_v1_10_freeze.py` (fail-closed; normal and `python -O` transcripts byte-identical).

Predecessor GP-DER-118-v1.9 / GP-DATA-236 remains frozen and unchanged at body SHA `d894c0fe35d17be878820971df5b7b300b6182ee71d79a89efe0d25ef6fad7f5`.

This snapshot records §5 component closure but does not promote P0.1 or close RP-C/RP-S.

END GP-DATA-237-v1.0

