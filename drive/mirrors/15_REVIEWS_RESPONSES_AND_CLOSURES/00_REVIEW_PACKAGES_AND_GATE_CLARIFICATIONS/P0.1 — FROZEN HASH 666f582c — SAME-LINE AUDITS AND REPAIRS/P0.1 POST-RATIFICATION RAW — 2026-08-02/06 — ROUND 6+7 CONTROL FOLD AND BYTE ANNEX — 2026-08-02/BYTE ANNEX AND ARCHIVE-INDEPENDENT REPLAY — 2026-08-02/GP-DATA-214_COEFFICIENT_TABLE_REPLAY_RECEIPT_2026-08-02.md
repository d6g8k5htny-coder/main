# GP-DATA-214 coefficient-table replay receipt

**Date:** 2026-08-02  
**Disposition:** EXACT PAYLOAD RECOVERED AND REPLAYED  
**Scope:** GP-DATA-214-v1.0 coefficient-evidence serialization only

## Source recovery

The exact Python source was extracted from
`work/round67_annex/byte_exports/GP-DATA-214_SOURCE_native_text_export.txt`
using the carrier's own extraction rule: normalize line endings, select the
payload strictly between `BEGIN_EXACT_PYTHON_SOURCE` and
`END_EXACT_PYTHON_SOURCE`, halve the uniformly doubled blank-line runs, remove
trailing line feeds, append one line feed, and encode UTF-8 without BOM.

- Recovered source: `GP-DATA-214-v1.0_r_uniform_exact_endpoint_normalizer.py`
- Bytes: `37512`
- SHA-256: `63ef800a51969903b9c3d97c74e61f77ee037afadff73038047ef7cfce040fb9`
- Carrier-declared bytes/hash: exact match

## Independent replay

Environment:

- Python `3.12.13`
- SymPy `1.14.0`
- mpmath `1.3.0`

The frozen source was executed unchanged. It returned status zero and printed
`ALL CHECKS PASS`. The regenerated result carrier is byte-identical to the
exact JSON extracted from the companion native result carrier:

- Result bytes: `10944`
- Result SHA-256: `af2779575ce3d18eeb69499e32ecc6e7c40898181ddf0fd6e4815d35cf8a6cbf`
- `cmp` result: byte-identical

The separate fail-closed replay harness imports the frozen source without
modifying it, invokes `symbolic_blocks()` and
`checked_series_coefficients(..., cutoff=6)` for `G`, `C`, and `S`, then uses
the source's canonical serialization:

```python
json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
```

Recovered exact payload:

- File: `GP-DATA-214-v1.0_coefficient_table.json`
- Bytes: `3269`
- SHA-256: `b2724c9379d9a3d7cb55998908d3480818ee4ac29e918a9b7a4192c350b2224e`
- Expected bytes/hash: exact match
- Harness terminal status: `ALL_ASSERTIONS_PASS`

## Replay artifacts

| Artifact | Bytes / SHA-256 |
|---|---|
| `replay_gp_data_214_coefficient_table.py` | SHA-256 `fc83fdd1e8ba0b73032425e427d1b1af0c5c34ff110b2804fd999529d87cabc8` |
| `replay_gp_data_214_coefficient_table.out.txt` | SHA-256 `a28c95e161ce7c75c4c25bf74194838c8ca4299ddd22836fa66824b38fc93de9` |
| `GP-DATA-214-v1.0_replay_stdout.txt` | SHA-256 `6c7f9ac0a0db6dc0b834453f8fbda9126dba0caa4a7a20588cbb42b5b2293dd9` |

No coefficient-table bytes were guessed or hand-transcribed.
