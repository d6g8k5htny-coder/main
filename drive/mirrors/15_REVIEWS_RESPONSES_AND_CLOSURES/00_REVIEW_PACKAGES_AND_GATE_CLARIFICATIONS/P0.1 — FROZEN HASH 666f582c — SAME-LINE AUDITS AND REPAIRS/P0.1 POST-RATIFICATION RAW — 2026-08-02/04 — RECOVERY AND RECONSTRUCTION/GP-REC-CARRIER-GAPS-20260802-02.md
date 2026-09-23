# GP-REC-CARRIER-GAPS-20260802-02 — post-ratification carrier ledger

**Date:** 2026-08-02  
**Disposition:** exact recovery where possible; explicit reconstruction otherwise  
**Package boundary:** `HOLD` remains in force pending the V3.4 independent audit of RP-C/RP-S

This ledger supersedes `GP-REC-CARRIER-GAPS-20260802-01` as a findability and provenance surface. It does not supersede any recovered original.

## 1. Recovered originals

- Exact SIDE24 R2 ZIP: 2,615,005 bytes; SHA-256 `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`; Drive `1djAbU8_Tg5a4TUsyK_7BuGTnDd7KWzt8`.
- Full V5 supplement: 105,980 bytes; SHA-256 `05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da`; Drive `1D5eRshQWyiugIrIikBUhItasLyiyV1FX`.
- Recovery archive copy: 22,269,093 bytes; SHA-256 freshly computed from the Drive raw bytes as `5393ec94ae6a05c31cc2fd22d45ff7667f9286c9f6b100c4425409786c7d1f58`; Drive `18PdDtAiB0mUK2tIM4UjhlxcU5XJHNRWF`.
- Exact V3.4 audit archive: 12,465,983 bytes; SHA-256 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`; Drive `1TSH5V0f-7BwdlHlY6-VarfuDFEQT-TIf`.

## 2. Reconstructed carriers

The later corrected G.9.1 note remains an explicitly labelled reconstruction. Its canonical reconstruction set is:

- note: 5,517 bytes; SHA-256 `46b6af0ce9c673f5263553202af721fa1610728240c404f93da35bf009ec9c3a`; Drive `1mdJ7EP1nCjhi86htN-exTaRnfO9DsjOZ`;
- fail-closed verifier: 8,563 bytes; SHA-256 `eb561bad4a35bd94ab5e3cb1e31e1462df3b94704873f9534e35544337123f85`; Drive `1Wo71xFuJJbFdRu8VqL2HKwAA_tyUlex5`;
- transcript: 1,199 bytes; SHA-256 `db2afa12f28e64e0e22e34bfad69e387be4469e7474266083f9fbd492d6692e8`; Drive `1ym4WntDNdXqy8kapX4WwTKmFVY1TTPgD`.

The missing G.9.2 source has now been reconstructed from the stated periodized-transfer claim, not represented as an original:

- note: 8,053 bytes; SHA-256 `0ba4f5030ab2e185118b1d624ec9ec62a7219ce2b27f19f993bcfbb2d833440d`; Drive `10_ibWW4GNuRoHZ5VjD_po0Qfwo3Wdwmy`;
- fail-closed verifier: 15,148 bytes; SHA-256 `18824c3ce8f3edb9f0c0ef051413c04ae9853d2649eaefd986c845d563aa36a9`; Drive `1TDT-YiosXVqW76vzE76NgCbC1H6QO7q4`;
- normal and optimized transcripts: 4,076 bytes each; identical SHA-256 `0ab9e14c938db8ec9795f202345f678527f5bb5b60cee3e9fed523e8816affbb`; Drive `1bY90vvrQeV5GJIcrtNBaXEqESnBwRRYL` and `1YuCvvwioI3fM5FsP4m9mPwKdayP_o8rI`;
- reconstruction checksum ledger: 480 bytes; SHA-256 `ba89afd892b78328d38c90da2445e6bb2c95944e7f0fd584a7bd11e82ce17043`; Drive `1mycditMZRgPM7euEgjzQ5y0Jpi61WxRJ`.

The G.9.2 verifier reports 59/59 checks in normal and `-O` modes, with byte-identical transcripts. Its forced-failure mutation exits nonzero. Scope is restricted to the normalized side-24 transfer on `0 < d <= 1/2`, `|u| = 1`; it makes no original-carrier, finite-rho, RP-C/RP-S, or promotion claim.

## 3. Exact carriers still not recovered

Full-tree, attachment-ZIP, archive-member, content, byte-count, and hash searches did not surface:

- verifier runs 15, 16, or 17;
- the R2.5 addendum;
- the original corrected G.9.2 carrier;
- the original `verify_g9_periodized_transfer_v1.py` source;
- a pre-existing post-R2.5 regenerated manifest.

The available uploaded run series stops at run 7. These missing items are not fabricated, backdated, or inferred from later receipts. `GP-MAN-POSTRAT-20260802-01.csv` is a new successor manifest for the present landing, not recovery of the missing earlier manifest.

## 4. Status firewall

AO48-OPR-042 closes the GP-DER-118 section 5 endpoint-Hessian gap and flips only N0 in LS-CTL-003. RP-C/RP-S remain audit-pending, the package remains `HOLD`, and no reconstructed carrier independently changes those states.
