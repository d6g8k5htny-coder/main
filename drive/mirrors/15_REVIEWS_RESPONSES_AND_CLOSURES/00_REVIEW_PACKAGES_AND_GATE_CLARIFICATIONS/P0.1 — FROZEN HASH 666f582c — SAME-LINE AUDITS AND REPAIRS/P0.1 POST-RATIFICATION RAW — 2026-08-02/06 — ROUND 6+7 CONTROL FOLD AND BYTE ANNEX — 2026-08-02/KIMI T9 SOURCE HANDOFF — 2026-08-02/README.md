# Kimi T9 source handoff

Date: 2026-08-02  
Operator: Dylan Roy  
Purpose: close the outstanding byte/source requests recorded by KIMI-AUD-016 and AO48-AUD-051.

## Contents

`core/` contains the requested GP-DATA-214 source, result, coefficient table, GP-DER-197 body/export, and LS-DER-026/027/030 bodies/exports.

`cl_dq012/` contains the CL DQ-012 primary verdict as a current native text/plain export, its exact marker-extracted 2,123-byte frozen body, and the CL lane-completion receipt.

`optional_lineage/` contains current native text/plain exports of GP-DER-118-v1.9 and GP-DATA-236, plus the exact 28,257-byte v1.9 theorem body bound by both carriers.

`MANIFEST.csv` and `SHA256SUMS` bind every payload in this handoff. The ZIP is a transport carrier; the extracted-file hashes are controlling.

## Exact identities needed for the review

- GP-DATA-214 source: 37,512 bytes, SHA-256 `63ef800a51969903b9c3d97c74e61f77ee037afadff73038047ef7cfce040fb9`.
- GP-DATA-214 result: 10,944 bytes, SHA-256 `af2779575ce3d18eeb69499e32ecc6e7c40898181ddf0fd6e4815d35cf8a6cbf`.
- GP-DATA-214 coefficient table: 3,269 bytes, SHA-256 `b2724c9379d9a3d7cb55998908d3480818ee4ac29e918a9b7a4192c350b2224e`.
- GP-DER-197 frozen body: 13,797 bytes, SHA-256 `035d5a18018606bd8736dc5774a7adbed29de453b30a1780976e1e4f69285d4d`.
- LS-DER-026 frozen body: 11,725 bytes, SHA-256 `e4d8094ed1bfc9f9cc7491f13efd5cc5294ba3b5f4fbd4bc8bbba33dd1a2ab5d`.
- LS-DER-027 frozen body: 11,330 bytes, SHA-256 `60768783048ae9f5347735caa6a543ff074b785f2d5a1c568237c92d12f12f39`.
- LS-DER-030 frozen body: 12,794 bytes, SHA-256 `5476f11a281635ad99a7ed125a55e06e875681aa273fb682ada02c9e0035d0c1`.
- CL DQ-012 verdict body: 2,123 bytes, SHA-256 `81748d71dbc4270f1f325cd48622cf2227cf48ebcf54c639fd6d318f253ecccd`.
- GP-DER-118-v1.9 theorem body: 28,257 bytes, SHA-256 `d894c0fe35d17be878820971df5b7b300b6182ee71d79a89efe0d25ef6fad7f5`.

## Primary Drive sources

- CL DQ-012 primary verdict: `1aQK0-emveJQqUDWP_Pck32mHNjtkMtT0hn7SA9toxBk`.
- CL DQ-012 lane receipt: `1ICG-UiF1nJYzx-O9TC471MGSv9OoX-D1A7JcYYRvjic`.
- GP-DER-118-v1.9: `1mHxvCzfVIS6eER8UBqcQdEP0R0q1KF8p70VZHmpWwU0`.
- GP-DATA-236-v1.0: `1RgY1VkeXeF244h8pYtTlMipYobJFRVA0kJMN-6nVEeU`.

The remaining source IDs appear row-by-row in `MANIFEST.csv`.

## Replay

Use Python 3.12.3 with the pinned packages in `requirements.txt`:

```bash
python3 core/GP-DATA-214-v1.0_r_uniform_exact_endpoint_normalizer.py \
  --output GP-DATA-214-v1.0_result.fresh.json
sha256sum GP-DATA-214-v1.0_result.fresh.json
```

The expected result hash is `af2779575ce3d18eeb69499e32ecc6e7c40898181ddf0fd6e4815d35cf8a6cbf` and the final line is `ALL CHECKS PASS`.

## Provenance and scope notes

- The GP-DATA-214 coefficient table is a deterministic materialization from the exact source and matches the declared byte identity. It is not claimed as recovery of a historical standalone carrier.
- The DQ-012 audit body is extracted from the primary native Doc using its unique frozen-body markers and reproduces the lane receipt's declared identity exactly.
- The v1.9 body is extracted from the current source carrier and is independently bound by GP-DATA-236 at the same byte count and SHA.
- CL DQ-012 records `APPROVE WITH CLARIFICATIONS / DQ-012 EXACT-OBJECT GATE CLOSED / THEOREM B NOT RESTORED`; it creates no new provider-family credit.
- LS-DER-027 is retained as an imported off-diagonal/far source. Its earlier recombination language is superseded by LS-DER-031; this handoff does not revive predecessor status.
- No theorem or promotion status is changed by this transport bundle.
