# E2 Conversion Readiness and Byte-Annex Inventory

**Date:** 2026-08-02  
**Mode:** read-only Drive inventory plus fresh byte extraction  
**Disposition:** mechanically conversion-ready; **E2 remains FALSE until a qualifying conversion verdict is executed**

## 1. Controlling E2 object

The controlling body is `LCR-DER-061-v1.1`, Drive ID
`164lkFwvYVMRNDDGAKu05LLBN5yRUmOMM3SlAr2jrNGU`.

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Native Google-Doc `text/plain` export, as returned | 13,882 | `35173b2b7ab51845e4c4ab789c61412490d76b5429711416d0a8c611f5ba0340` |
| Export after UTF-8 BOM removal and CRLF/CR to LF normalization | 13,617 | `de7edbdc2d7b1d6f2e028912382797ede27042639e1ffccc9e7944bc0e208418` |
| Exact bytes strictly between `BEGIN_BOREL_FROZEN_BODY` and `END_BOREL_FROZEN_BODY` | **8,928** | **`8ec4386fc1214523c98a5ae84acf4f5f64a6601aa2c63728f4382701b82dc3b7`** |

The parallel 8,566-byte / `ce9981b5…` branch is superseded and ineligible for conversion credit. The predecessor v1.0 identity remains part of the provenance record: 8,240 bytes, SHA-256 `04efac4014b21a387d9eae7d4e78e8e37e5b149629a000e7b4dbf64a8e2ead3b`, disposition AMEND.

The conversion request is `LS-P01-REQ-002`, Drive ID
`1c_-wbmoEiwKl4lzpKQFCY5BpdKDP0iYCqPhfEUSFkdk`.

## 2. Conversion checklist V0–V7

The following is the exact scope that must be adjudicated against the 8,928-byte body above.

- **V0 — lineage and credit.** The conversion is in the same Anthropic lineage, is a conversion supplement, and earns no second independent-review credit.
- **V1 — exact current identity.** Reproduce the current `LCR-DER-061-v1.1` frozen body at 8,928 bytes and SHA-256 `8ec4386fc1214523c98a5ae84acf4f5f64a6601aa2c63728f4382701b82dc3b7` before reviewing it.
- **V2 — predecessor provenance.** Preserve the v1.0 body at 8,240 bytes / SHA-256 `04efac4014b21a387d9eae7d4e78e8e37e5b149629a000e7b4dbf64a8e2ead3b` and its AMEND history; do not rewrite the earlier verdict.
- **V3 — repaired B10.** Verify that `A_r` is Borel and that, whenever `0 < Z_r^exact < infinity`, the exact conditional-Palm quotient is well-defined. This item does not supply a companion normalizer proof.
- **V4 — selector binding.** Verify the frozen-body binding to `LCR-DER-057-v1.1`, exact body 11,144 bytes / SHA-256 `62c23788fbed764240546b9f9aa84b40620a4e0725a39f2bec1fe7678fc74e72`.
- **V5 — no mathematical drift.** Confirm that the repair does not change the mathematical content of Sections 3–9 outside the stated B10/selector correction.
- **V6 — nonblocking clarifications.** Check the relative-Borel cover, the full `C_p` orbit, and the stated reopening conditions. These are clarifications, not new theorem claims.
- **V7 — scope fence.** The conversion covers event measurability and the conditional-Palm construction only. It does **not** prove the normalizer, positive probability, P0.1/P0.2, the `O(r^3)` theorem, Theorem B, promotion, sealing, or release.

A qualifying exact-hash conversion verdict covering the repaired conditional-Palm argument and selector-version binding is still required before changing E2. This inventory is not that verdict.

## 3. Direct dependency identity

`LCR-DER-057-v1.1`, Drive ID
`1ooqFTdifp8cxHYhRxmzJ2Y-nqx-971lpMQ4criehXWM`:

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Native `text/plain` export | 16,318 | `4c8b67608d4e9f897df9907f97d9cc8451836cf0e9caa62fc66d8c58017e0ef3` |
| LF-normalized whole export | 15,713 | `5b543d268d352b4d3a8dfa73d53bf0c2e67ae18c8c1247251c3ef6854bba53f1` |
| Body between `BEGIN_LP_FROZEN_BODY` and `END_LP_FROZEN_BODY` | **11,144** | **`62c23788fbed764240546b9f9aa84b40620a4e0725a39f2bec1fe7678fc74e72`** |

## 4. Byte-annex inventory requested for the Kimi lane

These identities were recomputed from fresh Drive exports, not copied from a status surface.

### GP-DATA-214 source

Drive ID `1ohTtnu7o2sbbcwTfoNTBte9TPLbC2vFlPd9raoDutE0`.

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Native `text/plain` export | 39,786 | `47a7730755f5bd8d75e1d17dbf68e96463a2145f5e3e0f792ea86a88ea315050` |
| LF-normalized whole export | 38,651 | `2d0a9a2d50d69427b4e177809bfb3da96af5c65e95d200ab99c4d99c6175c269` |
| Exact Python body between `BEGIN_EXACT_PYTHON_SOURCE` and `END_EXACT_PYTHON_SOURCE`, with the carrier's declared blank-run-halving rule applied | **37,512** | **`63ef800a51969903b9c3d97c74e61f77ee037afadff73038047ef7cfce040fb9`** |

### GP-DATA-214 result

Drive ID `1h_fCDDQU43NdUP-0l8YMpnca3OyRHdIIunyAV_h-b6k`.

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Native `text/plain` export | 12,440 | `e4fd486ba0b1a5d1827e918ce52e92110784a6922abf77dac7d0e955d434e883` |
| LF-normalized whole export | 11,966 | `d54fef37089307f7f01bbf4243869b50fada66268863df5c5bce3165bb9f9847` |
| Exact JSON body between `BEGIN_EXACT_JSON_RESULT` and `END_EXACT_JSON_RESULT` | **10,944** | **`af2779575ce3d18eeb69499e32ecc6e7c40898181ddf0fd6e4815d35cf8a6cbf`** |

The result declares a coefficient-table payload of 3,269 bytes with SHA-256
`b2724c9379d9a3d7cb55998908d3480818ee4ac29e918a9b7a4192c350b2224e`.
No pre-existing standalone coefficient-table Drive carrier was found. The
source defines the payload as

```python
json.dumps(coefficient_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
```

but prints only its identity. A wrapper replay against the exact 37,512-byte
source has now materialized `GP-DATA-214-v1.0_coefficient_table.json` at
exactly 3,269 bytes and reproduced the declared SHA-256. It is ready for a new
Drive upload; that new carrier must not be described as a recovered historical
standalone carrier.

### GP-DER-197 E0-pattern exemplar

Drive ID `1hGQVFDZKHMcdG94_OIgVEwipTR6pvKCk_-llHyIq3uQ`.

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Native `text/plain` export | 25,495 | `1974183747617041ab84d679fa632f8c22d9cf3dd09ed97ae27ad9f7b9573f71` |
| LF-normalized whole export | 24,796 | `5c5e53de9a30601d1326f0dfd902105badf500e4f71f3cea3f178dd811e95843` |
| Body between `BEGIN_FROZEN_BODY` and `END_FROZEN_BODY` | **13,797** | **`035d5a18018606bd8736dc5774a7adbed29de453b30a1780976e1e4f69285d4d`** |

This demonstrates the exact-body pattern requested for E0. The raw Google-Doc
export is not itself the 13,797-byte body; the delimiter extraction is
essential.

### LS-DER-026 / 027 / 030

| Artifact | Drive ID | Raw export bytes / SHA-256 | LF-normalized whole bytes / SHA-256 | Exact body bytes / SHA-256 |
|---|---|---|---|---|
| `LS-DER-026` | `1kSk34R28YpUdPmFjrLam1yDfOrcghyWJhTE_WbzqDGE` | 15,281 / `861de456e777255967a243e03a84d9c4496679a49fbd6d8e816eaa549bb24df5` | 14,746 / `4b43f6535215a6f6d4247c529ffa8616f2e6c528c99b6d0b7ea3ad3e6b31cea5` | 11,725 / `e4d8094ed1bfc9f9cc7491f13efd5cc5294ba3b5f4fbd4bc8bbba33dd1a2ab5d` |
| `LS-DER-027` | `1GXhNimem1IeqbdXjhnBqK-EzwCxw5CT1uWcG04-0KiE` | 16,153 / `f9234a32aa01153e6e94da2b1e8f956edabcf3da9a4ec9ab9de8b6bfabf29823` | 15,685 / `7327ecc777638d5377679c9e75ffb910be5155e3a5f003f39765c41c1931f577` | 11,330 / `60768783048ae9f5347735caa6a543ff074b785f2d5a1c568237c92d12f12f39` |
| `LS-DER-030` | `1xou7oHKy2nTa_eml5YBDhO_UVTmbspmArtWzwUiDpRY` | 16,940 / `1aba93e4d57060df5d3e8c1f3bc1789e41b6ea151bbbe92c3d839af16c981bbe` | 16,357 / `513fa5cd0c849ebbce144626e17b5e807d1740aad64f85ce1585540b3263f81b` | 12,794 / `5476f11a281635ad99a7ed125a55e06e875681aa273fb682ada02c9e0035d0c1` |

Delimiter pairs are respectively `BEGIN_TBG4_FROZEN_BODY` / `END_TBG4_FROZEN_BODY`, `BEGIN_TBG5_FROZEN_BODY` / `END_TBG5_FROZEN_BODY`, and `BEGIN_TB_COEFFICIENT_FROZEN_BODY` / `END_TB_COEFFICIENT_FROZEN_BODY`.

`LS-DER-027`'s normalization-inconsistent recombination is superseded by
`LS-DER-031`; its frozen body remains relevant only as the retained
off-diagonal/far `O(1)` proof source. Exporting the bytes must not be read as
restoring the superseded recombination.

### GP-DER-118-v1.10 raw theorem body

Controlling raw Markdown Drive ID
`1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9`; same-byte duplicate raw carrier
`1h6S231MbFP_2biZoJ-u9XnQ2IlPfC31v`; native audit mirror
`1IPJgw4W33CLwPLbIGf3ZcBwDhnZy_St_7lMCEGHHwBc`.

| Layer | Bytes | SHA-256 |
|---|---:|---|
| Whole raw Markdown carrier | **30,933** | **`c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564`** |
| Body between `BEGIN_FROZEN_THEOREM_BODY` and `END_FROZEN_THEOREM_BODY` | **29,293** | **`9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014`** |

This resolves the earlier missing whole-file `c1d6e559…` receipt without
altering the frozen theorem body.

## 5. Reproduction procedure

For native Docs, preserve the connector-returned bytes as the raw export,
then separately compute a normalized/extracted body:

```text
1. Export as text/plain without copy/paste or rich-text conversion.
2. Hash and count the raw returned bytes.
3. Decode UTF-8; remove one leading BOM if present; replace CRLF and CR by LF.
4. Hash and count the normalized whole export.
5. Require exactly one begin marker and one end marker in the correct order.
6. Extract only the bytes declared by the carrier's delimiter convention.
7. Apply an additional carrier-declared transform only where explicitly stated
   (GP-DATA-214 source: blank-run halving).
8. Hash and count the extracted body and compare with the table above.
```

Fail closed on a missing/duplicate marker, invalid UTF-8, an unrecognized
transform, a byte-count mismatch, or a hash mismatch.

## 6. Readiness conclusion

The exact `LCR-DER-061-v1.1` and `LCR-DER-057-v1.1` bodies are reproducible at
the required identities, and the V0–V7 scope is sufficiently pinned for a
conversion review. Accordingly E2 is **conversion-ready but not converted**.
Nothing in this inventory changes LS-CTL-003 or any promotion predicate.

For the broader Kimi byte annex, every requested body above is reproducible,
and the previously absent standalone 3,269-byte GP-DATA-214 coefficient table
has been generated locally at its declared exact identity. Drive landing and
receipt creation remain separate actions.
