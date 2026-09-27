# SIDE24 exact-source recovery — 2026-09-25

This directory repairs **source custody only**. It does not create a new theorem verdict.

Recovered byte-exact from connected Google Drive:

- `SIDE24_GAP_FILL_SUPPLEMENT.md`: Drive `1D5eRshQWyiugIrIikBUhItasLyiyV1FX`, 105,980 B, SHA-256 `05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da`, Git blob `bdf5f06026bfa5b4949aad881eb2c5075cf15fe2`. It contains G.1, Theorem G.3.1, Theorem G.7.1 and G.8.
- `SIDE24_THEOREM_PACKAGE_v1.0.md`: Drive `1YL4trMlYg55uowYj9ay-suvnwkXcBJXH`, 26,045 B, SHA-256 `703eefd5297afd4541d25e350ad27d130435a20951fc8290e4bcfb0676231a52`, Git blob `e12c1fa2dd62701a2b5d8bb36071764bd6c41c4a`. It contains the T1/T2 statements, lifetime pushforward, coefficient chain and the stated side-24 correction bound.

The V3.4 replay archive is a single blob at `custody/SIDE24_V3_4_AUDIT_REPLAY_FULL_ARCHIVE.zip`: 12,465,983 B, SHA-256 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`, 180 members. The same digest is Drive `13QS9QQHxSiuLPIPSh9o5plS5HkClwSmz` and Drive `1TSH5V0f-7BwdlHlY6-VarfuDFEQT-TIf`, and it is the chunk object of that digest in `drive/deltas/2026-09-19/DG-MIGRATION-20260919`.

Two different `arithmetic_ledgers.py` members of that archive are extracted beside it. They are not the same file. Neither extraction is a theorem verdict:

- 3,915 B, SHA-256 `40ad76a1973248d9d1a48ab093b9a2b6bcebf4a6cd8c36c0a09067ea64109f4c`, at `custody/arithmetic_ledgers/40ad76a1973248d9/arithmetic_ledgers.py`. The archive has this digest at both `evidence/historical_reproduction/arithmetic_ledgers.py` and `evidence/original_v2/SIDE24_gap_fill/scripts/arithmetic_ledgers.py`.
- 4,191 B, SHA-256 `ffbc6ccb48c08f78f4ab169aa98fba1c3eb0d96ed36c5fbf451d4a45e1b8bea4`, at `custody/arithmetic_ledgers/ffbc6ccb48c08f78/arithmetic_ledgers.py`. It is the nested member `evidence/v5_forward_branch/OKComputer_Project_Gap_Closure_1_119ffbfa.zip!/SIDE24_gap_fill/scripts/arithmetic_ledgers.py`.

## Fail-closed boundary

The theorem package says the historical V3.3 eigenfloor tables are a carried dependency. The exact standalone table carrier/hash is still not exposed. The recovered supplement's G.1 supplies a qualitative spectral/eigenfloor proof mechanism, but it is **not silently substituted for the historical table object**. Issue #117 should reclassify only the objects whose exact bytes are now present.

No 2D q0 result, RN/JETMOD status, independence credit or theorem grade changes from this recovery.

## Issue 117 missing-object search — 2026-09-26

Custody only. These classes do not accept the theorem.

| Object | Class |
|---|---|
| V3.4 replay archive, 12,465,983 B, SHA-256 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480` | PRESENT_EXACT |
| `arithmetic_ledgers.py`, 3,915 B, SHA-256 `40ad76a1973248d9d1a48ab093b9a2b6bcebf4a6cd8c36c0a09067ea64109f4c` | PRESENT_EXACT |
| `arithmetic_ledgers.py`, 4,191 B, SHA-256 `ffbc6ccb48c08f78f4ab169aa98fba1c3eb0d96ed36c5fbf451d4a45e1b8bea4` | PRESENT_EXACT |
| Module I §3 extraction, 9,700 B, SHA-256 `b958c21d822ada574db14145060d7958be135412e340d1bb29535bf800b2199c` | BLOCKED_ABSENT |

The Module I hash is the source-extraction baseline recorded for LS-DER-072-v1.0 in the 31 July 2026 technical appendix. The connected Drive document `19UCMJehB-RXez4cDWk8TZ2WrehVXA7vz9HA-h8GrWrI` is a different object: its stored native export is 15,053 B, SHA-256 `68118b8a478ef24d9fe7c71590244deabf039769f5b5fc7b67b6e102e12c49ad`, and the frozen body named inside that export is 11,193 B, SHA-256 `183bb12b5c92f4490dccf63c9c7507325cd3cbe59e066dc5f9a636a7f5e5b621`. Search surfaces that did not contain the 9,700-byte digest: every local Git blob size, the migration object index, the V3.4 archive through one nested ZIP level, the pre-review packet ZIP `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`, and Drive title/full-text hits, which cite the hash or return the different document above. Vault99 was not searched.
