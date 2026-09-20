# SIDE24 — Direct Kimi Carrier and d7 Reconciliation

**Artifact ID:** GP-SIDE24-THM-001-AUD-ADD-001-v1.0  
**Date:** 2026-08-02  
**Authority:** Preservation and reconciliation record; no independent promotion authority  
**Controlling theorem state:** RATIFIED-AT-STATED-SCOPE under AO48-OPR-045, unchanged  
**Parent package:** GP-SIDE24-THM-001-v1.0, frozen bytes unchanged

## 1. Outcome

The direct-carrier gap recorded in the frozen theorem package is closed. The operator-supplied archive has been authenticated, preserved as received, and separated from the controlling audit texts. Raw KIMI-AUD-006 and KIMI-AUD-006b carriers are now in Drive as byte-identical files, with distinct whole-file and embedded report-body identities. Native Google Docs mirrors are secondary reading surfaces only; the raw text files control.

No mathematical conflict was found between the direct reports and AO48-AUD-043, AO48-AUD-044, or AO48-OPR-045. The sequence remains:

1. KIMI-AUD-006: AMEND REQUIRED, pre-committed to convert without further review after amendments A–C.
2. KIMI-AUD-006b: APPROVE after A–C were verified landed.
3. AO48-OPR-045: operator ratification, closing RP-C/RP-S and accepting the theorem at its stated scope.

## 2. Controlling raw audit carriers

### KIMI-AUD-006

- Drive ID: `1HQIpE8Ey7VOiXN7vJWucJIVEAdsXgx0h`
- Bytes: 8,329
- Whole-file SHA-256: `17c8eba9fcc4a5d1475d7f9e6f3b51ff060aa5ecca78ee879617a20ea0c127fd`
- Embedded report-body SHA-256: `4b66f85e97e2338362e9086abd0f8c939a71248fbdea08e4a462d141756f5a38`
- Post-upload verification: byte-identical by direct base64 equality.

The embedded body hash is the identity abbreviated as `4b66f85e…5a38` in AO48-AUD-043. It is not the whole-file hash.

### KIMI-AUD-006b

- Drive ID: `1hfUOS2YXKfDfE8Al-sCGNf1PX-FgtSgN`
- Bytes: 4,624
- Whole-file SHA-256: `2a38f2d412b4a7b51f008fc5593889b2598fbe01197ad1d873646bb7444877ce`
- Embedded report-body SHA-256: `9e615e81d0d0569c57b66df94414085d1bf9ac5cc6e7d11e0c8ddfe160629d77`
- Post-upload verification: byte-identical by direct base64 equality.

The report verifies amendments A and B through the separate envelope carrier, not by hash-binding a newly republished facewise-v1.1 file. It verifies amendment C through the three supporting-file hashes and reports a 96-check normal/optimized replay. This archive does not contain separate Task-7 verifier sources or those replay transcripts; none are claimed here.

## 3. Amendment-C replay sources

The three sources cited by KIMI-AUD-006b are absent from the received ZIP but already exist in Drive with the exact hashes bound by the report:

- `three_hessian_conditional_moment_envelope.md`: Drive `1Ws1KlIYXdDDwYqReEWx_brm3TBxgnjQp`; SHA-256 `f19132b47dd58339d632ffe8ef5c9466dbcf9b353b406a31e22d3e35b5e5d0fb`.
- `quartic_remainder_audit_note.md`: Drive `1lcVeJ0J7jglJEjn6A684dE3MDCBwGzNo`; SHA-256 `e344ccec3945ecd1c931e7e84eb25ef2fe2a0c9b41f5768eb03619400b2b38db`.
- `verify_quartic_endpoint_counterexample.py`: Drive `12RkAxAiQD_B6Mh8emfPeVWm-odAShf-e`; SHA-256 `461959368dfc7759c78ee105e4b82ff97659ddfc3a28837bbf7e8092da2f39d9`.

The three previously absent diagnostic scripts also remain available through their already-landed raw Drive carriers:

- hybrid mixed-curvature/axis absorption: Drive `1VPYlprjcWgNmoIMiVaTL5sa6wUirseVr`; SHA-256 `0ba99085cf5b1655deb95859a63f02478057491fe1d394124d853e93bea8c481`.
- facewise collar/axis integration: Drive `1a67DJHnF8NKPxDdhdKvIDpOHi0FWCeQR`; SHA-256 `bb4fdb739636e504d2c2eeafd8d8582dbf1dba12225bca88788927b049d58e5f`.
- generic collar diagnosis: Drive `1iiyLpIDkYR0YDiwv9vGRVQ-Rel8uh0sD`; SHA-256 `abc5b8cd3f03c251a8bf47115a9d0097ef03ae62dd65debcb23082ba973176c0`.

## 4. Received snapshot identity and limitations

- Drive ID: `1D1FKF58INkBnDmIWvBDPHJPMCackBtqs`
- Bytes: 6,213,385
- SHA-256: `50bef1fc2ad9e585ea36a9c890b4759b18937be0e4af73d4cfbe517a3ccaf815`
- Structure: 184 unique path-safe files; ZIP CRC test clean; 21,880,093 uncompressed bytes.
- Exact 184-file inventory: Drive `19miRtzIugMcjGfpnlVoZ7mnm8Ce0kogD`; SHA-256 `4e317e7ef7198d5d28279bd08b5a9dc18462f9f7ca939e0cd55a5c29ee97aba8`.

This is a broad pre-ratification project snapshot, not a controlling theorem carrier. Its R2.7 text still describes a procedural HOLD pending operator ratification; AO48-OPR-045 and the post-ratification R2.8 surfaces supersede that status language.

The V5 manifest correctly binds both direct Task-7 reports and 180 of 182 audited non-self rows. Its two master-register rows are stale: each expects `b528e9da…`, while the received files hash to `18437144…` and `9c848ea6…`. `KIMI-DATA-008b_inventory_full.txt` is present in the ZIP but omitted from the V5 manifest. These are packaging defects, not contradictions in the audit reports.

## 5. Task 5 and Task 6 post-ratification fold

Two further raw Kimi carriers are now preserved and were verified byte-identical after upload:

- KIMI-AUD-007 (Task 5): Drive `1LSGQUWnoEs77qlhdg6btb2rvnzc8nLEZ`; 4,071 bytes; SHA-256 `82b0a22bf802a55406f52c00c73f38ca3e49b8aab1d3e6be5d5f2db9bf8f7d55`. Verdict VERIFIED, with measured sharpening of the already-ratified axial-suppression machinery.
- KIMI-DER-006 (Task 6): Drive `1ea_wTmC0ixYDjFVGunj91xgx4FgAzkvo`; 4,417 bytes; SHA-256 `709a4d24c0c09d2e9ccd125571fdb38f20b7cb1a081e9065959c7a00596d5331`. Verdict TRUE for the d7 mechanism.

AO48-AUD-047 (Drive `1VpWNVgpgmgX9B29PxUXqX3eNroZFGZBt`) records the completed triple-blind comparison. The three routes agree on

`Cov(f_xx,f_xxx | pins) = -d^3 + (2/5)d^5 - (67/960)d^7 + …`

and on the axial first-order slope

`c1 = -(5/12)d^7 + (59/180)d^9 + …`.

Mechanism wording is reconciled as follows: midpoint reflection and coalescing odd-sector pins force `Var(f_xxx | pins) = 6d^2 + O(d^4)`; the resulting covariance is order `d^3` and saturates its sharpened Cauchy–Schwarz bound. There is no further cancellation after that odd-sector pinning. The supporting d7 item is therefore DERIVED. This does not alter the already-ratified theorem statement or its scope.

## 6. Later Kimi reports: preservation without promotion

All thirteen Kimi report carriers from the archive were uploaded individually and are indexed by `KIMI_DIRECT_CARRIER_MANIFEST_2026-08-02.csv`.

- KIMI-DER-009 proves the selection/disintegration lemma modulo one named analytic premise P; it is not an unconditional closure.
- KIMI-DER-010 assembles the five-chart joint gluing statement sufficient for the audited SIDE24 collar integration; it is supporting evidence, not a new theorem promotion.
- KIMI-DATA-008b reports that the seven q0 bind-target carriers were absent from Kimi’s received corpus. KIMI-DER-002 is an independent reconstruction at its own scope and does not bind the actual PZ0 carrier. No q0 Boolean changes here.

## 7. Status disposition

- SIDE24 theorem package v1.0: frozen bytes unchanged.
- SIDE24 theorem status: RATIFIED-AT-STATED-SCOPE, unchanged.
- RP-C/RP-S: CLOSED, unchanged.
- Direct KIMI-AUD-006/006b preservation gap: CLOSED by this record.
- d7 mechanism: DERIVED, supporting status.
- q0/P0.1: unchanged; all existing firewall and HOLD semantics remain in force.

Any future citation should use the raw Drive IDs and whole-file hashes above, with embedded body hashes labeled explicitly as body identities.
