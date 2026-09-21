# SIDE24 Eigenfloor Replay Receipt

**Date:** 2026-08-02  
**Working directory:** `/workspace/scratch/4c1cbf90475f`

## Environment

```text
Python 3.12.13
numpy 2.3.5
scipy 1.17.0
```

The exact G.9.2 transfer script is Python-standard-library-only. SymPy and
mpmath are not required by any of the three commands below.

## Commands and results

```bash
python3 work/kimi_received_20260802_main/SIDE24_gap_fill/scripts/verify_corrected_pin_floor.py \
  > work/round67_annex/eigenfloor_replay/verify_corrected_pin_floor.fresh.out.txt 2>&1
```

Exit 0; `ALL_ASSERTIONS_PASS`. Source: 3,668 bytes, SHA-256
`06a82d0f545bc4699432720a1ea23295b86ce86a7a129ba184dc413e31e0b4f3`.
Fresh transcript: 1,996 bytes, SHA-256
`8be3f094d5bf257103902ed6c2e3118ffa293646c8c28771aecce4196c219656`.

```bash
python3 work/v34_sources/diagnose_side24_generic_collar.py \
  > work/round67_annex/eigenfloor_replay/diagnose_side24_generic_collar.fresh.out.txt 2>&1
```

Exit 0; `ALL_CHECKS_PASS`. Source: 6,551 bytes, SHA-256
`abc5b8cd3f03c251a8bf47115a9d0097ef03ae62dd65debcb23082ba973176c0`.
Fresh transcript: 2,953 bytes, SHA-256
`7af42105d4b98fe6838c411fe40857159e0f954aab9cf216813589b9b4244b18`.

```bash
python3 work/g9_2_rebuild/verify_g9_2_periodized_transfer_reconstruction.py \
  > work/round67_annex/eigenfloor_replay/verify_g9_2_periodized_transfer_reconstruction.fresh.out.txt 2>&1
```

Exit 0; 59/59 checks and `ALL_ASSERTIONS_PASS`. Source: 15,148 bytes,
SHA-256
`18824c3ce8f3edb9f0c0ef051413c04ae9853d2649eaefd986c845d563aa36a9`.
Fresh transcript: 4,076 bytes, SHA-256
`0ab9e14c938db8ec9795f202345f678527f5bb5b60cee3e9fed523e8816affbb`.

## Scope

These are replay transcripts, not a reconstruction of every frozen V3.3
eigenfloor table. The corrected-pin and generic-collar scripts are explicitly
numerical diagnostics; the G.9.2 script has scope `0<d<=1/2`, `|u|=1` and
does not claim original-carrier identity, a finite-rho remainder, RP-C/RP-S
closure, or promotion.
