# SIDE24 V3.3 Eigenfloor: Archive-Independent Replay and Negative Provenance

**Date:** 2026-08-02  
**Disposition:** partial independent replay; original V3.3 archive bytes not recovered  
**Scope fence:** this report does **not** claim full recovery or re-derivation of every frozen V3.3 eigenfloor table

## 1. Archive identity and search result

The referenced V3.3 archive is described as:

- byte count: **12,388,862**;
- SHA-256: **`4040d654…`** (only this prefix is available in the present record).

No byte carrier matching that identity was found in the current local uploads,
their ZIP members, or the accessible Drive/registry search surfaces. Searches
used the strings `4040d654`, `12,388,862`, `12388862`, `V3.3`, and
`eigenfloor`, plus member-name scans of every supplied ZIP. The coupled
registry searches over Recent Activity, Artifact Index, Frozen Objects,
Identity Drift Watch, Evidence Lineage, and Run Log returned no identity row.
Drive search returned only `AO48-WO-048` citing the archive identity and its
absence, not the archive itself.

The current Kimi upload is a different object:

| File | Bytes | SHA-256 |
|---|---:|---|
| `OKComputer_Project_Gap_Closure(2).zip` | 6,213,385 | `50bef1fc2ad9e585ea36a9c890b4759b18937be0e4af73d4cfbe517a3ccaf815` |

It must not be relabeled as the 12,388,862-byte V3.3 archive. Until the exact
archive bytes are supplied and the full SHA-256 is computed, the archive
identity belongs in the negative-provenance ledger as **cited, not landed**.

## 2. What the live proof chain consumes

The facewise v1.1 layer states that V3.3 supplied uniform compact-set
eigenfloors and the Palm normalizer comparison

```text
c r^2 <= Z_r <= C r^2.
```

The later facewise estimates consume the resulting density scales and Schur
controls, including:

- the corrected raw-to-witness collar scaling `O(r^-7)`, from weights
  `(r^3, r^2, r, r)`;
- the singular-frame scale `O(s^-7)`, from weights
  `(s^-3, s^-2, s^-1, s^-1)`;
- the generic transverse contact determinant comparison
  `varrho^6 (4 X^2 + varrho^2)`;
- positive normalized Schur eigenfloors on the applicable compact charts.

`KIMI-AUD-006` explicitly recorded that these rest on the frozen V3.3-layer
tables, found them internally consistent and of the same class as its
independently proved G.3/G.7.1 normalizer work, but did **not** re-derive the
tables. This replay preserves that distinction.

The current Part G analytic source proves positivity by full Fourier support,
linear independence of the normalized derivative distributions, continuity,
and compactness after the correct blow-up has been installed. The numerical
runs below corroborate values and catch basis/scaling regressions; they do not
replace the analytic proof or reconstruct an unavailable table carrier.

## 3. Fresh replay environment

```text
Python 3.12.13
numpy 2.3.5
scipy 1.17.0
sympy unavailable (not required by these runs)
mpmath unavailable (not required by these runs)
```

The G.9.2 transfer verifier is Python-standard-library-only. The corrected-pin
and generic-collar diagnostics require NumPy.

## 4. Replay A — corrected pair-pin eigenfloor

Source:

| File | Bytes | SHA-256 |
|---|---:|---|
| `work/kimi_received_20260802_main/SIDE24_gap_fill/scripts/verify_corrected_pin_floor.py` | 3,668 | `06a82d0f545bc4699432720a1ea23295b86ce86a7a129ba184dc413e31e0b4f3` |
| committed output transcript | 1,996 | `8be3f094d5bf257103902ed6c2e3118ffa293646c8c28771aecce4196c219656` |

Command:

```bash
python3 work/kimi_received_20260802_main/SIDE24_gap_fill/scripts/verify_corrected_pin_floor.py
```

Fresh result: exit 0, `ALL_ASSERTIONS_PASS`. The lattice truncation retained
50,541 modes with radius cutoff 6. The sampled minimum eigenvalues of the
unnormalized corrected-pin covariance were:

| Direction | r=2 | r=1 | r=0.5 | r=0.2 | r=0.1 | r=0.05 | r=0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(1,0,0)` | 11.03823 | 41.84762 | 36.35271 | 34.65311 | 34.41511 | 34.35586 | 34.33929 |
| `(1,1,0)/sqrt(2)` | 11.03823 | 41.84762 | 36.35274 | 34.65316 | 34.41516 | 34.35591 | 34.33934 |
| `(1,1,1)/sqrt(3)` | 11.03823 | 41.84763 | 36.35275 | 34.65316 | 34.41517 | 34.35591 | 34.33934 |

This independently reproduces the current register's canonical corrected-pin
behavior: the trapezoid-corrected last coordinate has a stable limiting
eigenfloor near 34.4. It does not certify a uniform all-direction/all-radius
constant by sampling. The script says so explicitly: numeric corroboration
only; Part G carries the analytic proof.

## 5. Replay B — generic transverse normalized Schur face

Source:

| File | Bytes | SHA-256 |
|---|---:|---|
| `work/v34_sources/diagnose_side24_generic_collar.py` | 6,551 | `abc5b8cd3f03c251a8bf47115a9d0097ef03ae62dd65debcb23082ba973176c0` |

Command:

```bash
python3 work/v34_sources/diagnose_side24_generic_collar.py
```

Fresh result: exit 0, `ALL_CHECKS_PASS`. The run retained 50,541 side-24
lattice modes with cutoff 6. Across 24 direction/shape probes, the determinant
ratio to

```text
varrho^6 (4 X^2 + varrho^2)
```

converged to approximately `0.999931`–`0.999932` at the smallest sampled
radius. Over the full finite-radius diagnostic grid the ratio range was
`[0.78104, 1.83594]`; the minimum normalized Schur eigenvalue was
`0.0485002`.

This is a generic-compact diagnostic only. It does not cross the unnormalized
angular axis, and it is not a substitute for the separate axial blow-up or a
full compact-table proof.

## 6. Replay C — exact periodized transfer and stable planar pin floor

Source:

| File | Bytes | SHA-256 |
|---|---:|---|
| `work/g9_2_rebuild/verify_g9_2_periodized_transfer_reconstruction.py` | 15,148 | `18824c3ce8f3edb9f0c0ef051413c04ae9853d2649eaefd986c845d563aa36a9` |
| committed output transcript | 4,076 | `0ab9e14c938db8ec9795f202345f678527f5bb5b60cee3e9fed523e8816affbb` |

Command:

```bash
python3 work/g9_2_rebuild/verify_g9_2_periodized_transfer_reconstruction.py
```

Fresh result: exit 0, 59/59 checks, `ALL_ASSERTIONS_PASS`. Exact arithmetic
reproduced the stable planar pin eigenvalue floor

```text
21/64.
```

It also reproduced the certified transfer envelopes

```text
ALL_IMAGE_1D_BOUND       = 2.441406250000E-111
STABLE_JOINT_ENTRY_BOUND = 1.126562500000E-106
CONDITIONAL_GAMMA_BOUND  = 9.463125000000E-103
DETERMINANT_TRANSFER_BOUND = 4.542300000000E-101
```

and all eight high-precision probes stayed inside those analytic bounds. Its
declared scope is `0 < d <= 1/2`, `|u|=1`; it asserts no original-carrier
identity, finite-rho remainder, RP-C/RP-S closure, or promotion.

The exact `21/64` floor and the approximate `34.4` corrected-pin floor refer
to different normalized frames/objects. They are not competing numerical
estimates and must not be compared as if they were the same matrix.

## 7. Additional identity used to orient the replay

The amended facewise proof carrier used for comparison was:

| File | Bytes | SHA-256 |
|---|---:|---|
| `work/facewise_v1_1/rp_c_rp_s_facewise_closure_v1.1.md` | 36,542 | `d5677359e72243a89450c14d324051948d721aa031bb729f2ce9f76fb392804e` |

No byte of that frozen proof was changed during this replay.

## 8. Re-derived versus not re-derived

### Re-derived or freshly replayed

- positivity and small-r stabilization of the canonical corrected pair-pin
  covariance on three representative directions and seven radii;
- the generic transverse determinant-ratio behavior on the declared 24-probe
  compact diagnostic grid;
- the sampled normalized Schur minimum `0.0485002` on that grid;
- the exact stable planar pin floor `21/64`;
- the G.9.2 image, joint-entry, conditional-covariance, and determinant
  transfer bounds and eight precision probes;
- the basis warning that omitting the trapezoidal correction produces the
  wrong degenerate last coordinate, whereas the canonical corrected basis
  stabilizes near 34.4.

### Not re-derived and not claimed recovered

- the exact 12,388,862-byte archive;
- its full SHA-256 beyond prefix `4040d654…`;
- a row-by-row byte-identical reconstruction of every frozen V3.3 eigenfloor
  table;
- uniformity over faces or parameter ranges not covered by the analytic
  arguments and declared script scopes;
- the axial-face theorem by inference from the transverse diagnostic;
- any new RP-C/RP-S or theorem promotion claim.

## 9. Negative-provenance entry

Recommended ledger text:

> **V3.3 eigenfloor archive — cited, bytes absent.** Expected identity:
> 12,388,862 bytes, SHA-256 prefix `4040d654…`; full digest unknown. No
> matching local upload, ZIP member, Drive object, or registry row was found
> on 2026-08-02. The current 6,213,385-byte Kimi upload has SHA-256
> `50bef1fc2ad9e585ea36a9c890b4759b18937be0e4af73d4cfbe517a3ccaf815`
> and is not the archive. Key eigenfloor classes were replayed independently
> in this report, but the frozen V3.3 tables were not recovered. Reopen archive
> identity only on receipt of exact bytes reproducing 12,388,862 bytes and a
> full SHA-256 beginning `4040d654`.

## 10. Conclusion

The available source is enough to independently corroborate the principal
eigenfloor mechanisms and several values consumed by the current proof chain.
It is not enough to make the stronger historical claim that the frozen V3.3
table archive itself has been recovered or exhaustively rebuilt. The honest
status is therefore: **mathematical corroboration refreshed; historical
carrier unresolved and negatively recorded**.
