# Inventable JETMOD probes (REFUSED only)

Fail-closed inventable attempts. The first four receipts are the sibling-sweep
CLOSED EMPTY named walls. Do not invent lemmas or new walls.

Three tip-aligned shortcut refusals record already-stated false-progress paths.
They are not in the sibling-sweep four.

- Invent a 24-jet roster and `p_J` without a Drive enumeration — **REFUSED_NOT_24JET**. Not PARTIAL. No roster is invented. Order>2 API missing (`STATUS_JETMOD` STOP).
- Promote the display residual or struct κ as a certified enclosure — display ≠ certified.
- Merge draft PR #12 as a math_status catch-up, or claim RUNG2/3 discharges JETMOD. PR #12 stays unmerged. Green ≠ discharge.

Sibling `inventable_*` receipts that sit in this directory and were unnamed above:

- `inventable_eval_F_G12box_REFUSED_receipt.json` — `REFUSED`
- `inventable_interval_schur_ainv_REFUSED_receipt.json` — `REFUSED_IA_STRADDLES`
- `inventable_joint_ry_cancel_EMPTY_receipt.json` — `EMPTY`
- `inventable_phi_bridge_ABSENT_receipt.json` — `ABSENT`
- `inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json` — `REFUSED`
- `inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json` — `REFUSED`

Those tokens are honesty labels, not discharge. Naming the merge receipt does not merge draft PR #12.

```bash
python3 docs/math_status_probes/inventable_jetmod_probes.py
python3 tools/math_status_check.py   # also validates these receipts
```

`discharges_OBL_H5_JETMOD` stays false. `discharges_lemma` stays false.
`lemma_closed` stays false. `certified_C_H` stays false. `freeze` stays false.
`inventable_attempt_accepted` stays false. OBL-H5-JETMOD stays OPEN.
PACKET disposition stays OPEN_HOLD. Green ≠ discharge.

## Instrumentation STATUS vocabulary (PARTIAL / REFUSED_NOT_24JET)

Fills former `?` inventory rows under local `code_prototypes` (first_band*, g12_ext_named, multi_jet_band) with honest STATUS only.

```bash
python3 docs/math_status_probes/inventable_jetmod_instrumentation_status.py
python3 tools/math_status_check.py
```

Does not invent a 24-jet roster. Does not promote display/κ. Does not merge PR #12.
`discharges_OBL_H5_JETMOD` stays false. `lemma_closed` stays false. `certified_C_H` stays false.
`prizes_solved` stays 0. OBL-H5-JETMOD stays OPEN. Green ≠ discharge.

## Vault paths and quarantine tooling (non-activating)

Committed inventable receipts in this directory name no vault id and no
quarantine path. `INVENTABLE_PROBES_INDEX.json` and
`INVENTABLE_INSTRUMENTATION_STATUS_INDEX.json` carry no vault id, no exclusion
digest, and no current-tip re-run. Receipt bytes and `aligned_to_base_tip`
stay as already recorded. `quarantine/EXCLUSIONS.json` stays the tip pin.

A vault id or a quarantine path, named from this lane, stays inactive. It
does not become an inventable source of truth. Quarantine is not a source
of truth.

Two checkers divide the vault-path and exclusion hygiene. A third checker,
`tools/registers_check.py`, owns the class-table pass on the same export.
None of the three writes an inventable receipt. None sets
`inventable_attempt_accepted`.

| Checker | What it owns on this tip | What stays closed for an inventable attempt |
|---|---|---|
| `tools/quarantine_check.py` | `quarantine/EXCLUSIONS.json` agrees with `registers/json/quarantine_index.json`. Archive-member digests. Invariant 3 compares records that carry `payload_sha256` (sixteen compared, six `digest_not_compared`, including folder exclusion `Q-R17-VAULT`). Bound-member annotations. `vault_rows` refuses a stored manifest row whose `drive_path` contains `99_DO_NOT_OPEN`. | `Q-R17-VAULT` is uncompared because the exclusion names a folder. That record is not an opening and not acceptance of a vault id. A stored vault id whose `drive_path` omits `99_DO_NOT_OPEN` is outside `vault_rows`. |
| `tools/vault_hygiene_check.py` | The map in [`quarantine/PATHS.md`](../../quarantine/PATHS.md): `drive/vault_tree.txt` against `drive/inventory.jsonl`; a stored row whose Drive id is a vault id even when `drive_path` omits `99_DO_NOT_OPEN`; a vault or `90_QUARANTINE_AND_TRIAGE` path copied into `engine/`, `research/`, `packages/`, or `claims/`. | That active-lane scan does not read this directory. Silence on an inventable receipt is not activation. |
| `tools/registers_check.py` | The class column of `registers/json/quarantine_index.json` against the OP-PROT-019 §6 table, among the other structural invariants. Rows 14–16 are `Q-R17-LOCAL-TB`, `Q-R17-LOCAL-P01`, and `Q-R17-VAULT`, class `EXISTING_CONTAINER`. That class is absent from the table, so the run prints three KNOWN lines. The allowlist text in `registers/KNOWN_FINDINGS.json` says "Accepted as-is". | "Accepted as-is" records that source-workbook class. `inventable_attempt_accepted` stays false. A green run (those KNOWN lines included, new problems at zero) leaves the vault id inactive and leaves OBL-H5-JETMOD OPEN. Agreement of `quarantine/EXCLUSIONS.json` with the export stays on `tools/quarantine_check.py`. |

A green run of any of these three checkers is engineering hygiene. It leaves
`discharges_OBL_H5_JETMOD` false, `lemma_closed` false, `prizes_solved` at 0,
`certified_C_H` false, `freeze` false, and `inventable_attempt_accepted` false.
OBL-H5-JETMOD stays OPEN. Engineering hygiene is not mathematical discharge.

Exclusion classes stay in [`quarantine/README.md`](../../quarantine/README.md).
The path map stays in [`quarantine/PATHS.md`](../../quarantine/PATHS.md).

## SIDE24 prep is not an inventable source of truth

The local SIDE24 prep path is navigation only:
[RN_SIDE24.md](../RN_SIDE24.md), then
[RN_SIDE24_DENSITY.md](../RN_SIDE24_DENSITY.md), then
[RN_SIDE24_CELL.md](../RN_SIDE24_CELL.md). The commands sit in
[RESEARCH_EXECUTION.md](../RESEARCH_EXECUTION.md). The reader route is
[RESEARCH_MAP.md](../RESEARCH_MAP.md) §3. Each note already labels
**ABSENT**, **REFUSED**, and **OPEN**.

SIDE24 has no new source-of-truth carrier after 2026-08-06. Later rows in
the exported file catalog that mention SIDE24 are metadata-only routing,
not that carrier. Objects those notes mark **ABSENT** stay **ABSENT**. A
quarantine path does not activate a carrier. Quarantine is not a source of
truth. A green run of `tools/rn_side24_check.py`,
`tools/rn_side24_density_check.py`, or `tools/rn_side24_spatial_check.py`
is engineering hygiene. It leaves `inventable_attempt_accepted` false and
leaves OBL-H5-JETMOD OPEN. It is not discharge. The marked-cylinder delivery
under `research/campaigns/math_push_20260924/` is not that carrier.

## Honesty labels and tip provenance

`REFUSED` ≠ discharge. `REFUSED`, `REFUSED_IA_STRADDLES`, `EMPTY`, `ABSENT`, `PARTIAL`, and `REFUSED_NOT_24JET` are honesty labels. They are not discharge, not a source of truth, and not FREEZE. They are not an RN source of truth. `STATUS_RN_UNIF.md` keeps that lane. Instrumentation STATUS is `PARTIAL` / `REFUSED_NOT_24JET` only. Sibling and shortcut receipts labeled `REFUSED`, `REFUSED_IA_STRADDLES`, `EMPTY`, or `ABSENT` are not instrumentation STATUS.

`aligned_to_base_tip` on `INVENTABLE_PROBES_INDEX.json`, and on the three shortcut receipts that already carry it (`inventable_24jet_roster_without_Drive_list_REFUSED_NOT_24JET_receipt.json`, `inventable_promote_display_residual_struct_kappa_REFUSED_receipt.json`, `inventable_merge_PR12_or_rung_discharge_REFUSED_receipt.json`), is `1ea0ae8183fb0459c6678243946295518fded1ba`. That SHA is generation provenance (`HISTORICAL_NONCURRENT`). It is not a re-run on the hardening tip observed at this edit, `fcad72366743f5244eb26a0164e106419de78137`. The probe runners were not re-executed on that tip. The prior observation chain includes branch tip `f244312db901cf6ede88e46af663398d496eb989` after #108; it is not a re-run on that tip either. The prior observation chain includes branch tip `848aea2a874cbaa879ea2ece13accd94d1fca20d` after #102; it is not a re-run on that tip either. The prior observation chain includes branch tip `eeebb28eeab2452edc1103817d02ef7e3abbf5f2` after #99/#100; it is not a re-run on that tip either. The prior observation chain includes branch tip `388a22caff13c449f0e3264b0aff314d456d6659` after #97; it is not a re-run on that tip either. The prior observation chain includes branch tip `077464ef5e2859ce98cbb9307799d5867a820eaf` after #89; it is not a re-run on that tip either. The prior observation chain includes branch tip `0adeb651d92fb024d7b21994c7fc60375060fe44` after #85; it is not a re-run on that tip either. The prior observation chain includes branch tip `02cfbfdf3ddc3120ddde6b67b0ff927115de6c4c` after #84; it is not a re-run on that tip either. An earlier observation named hardening tip `3a29f526da5108df173edc390a8ca2d1f3d887c9` after #83; it is not a re-run on that tip either. An earlier observation named hardening tip `3b3860da7336528a9517ae198f09c83f68fa137a`; it is not a re-run on that tip either. An earlier observation named hardening tip `8e359e5bf879f524e11cbeece9b36bd9996d2587`; it is not a re-run on that tip either. An earlier observation named hardening tip `542e6ec2f462d6202f5bc5b3a044e71ae7a1a96c`; it is not a re-run on that tip either. An earlier observation named hardening tip `b89448da439d963a404212ae024534169aa22297`; it is not a re-run on that tip either. An earlier observation named hardening tip `8bd1f03cc2bb10c59b08b852ca2775dac27e28e9`; it is not a re-run on that tip either. An earlier observation named hardening tip `c82c9357db381e8fd60d939a7243dab4cc863118`; it is not a re-run on that tip either. An earlier observation named hardening LOCK `b3da6688a55d34681bb27f17ba6c6c5e16ad534c` (short `b3da668`); it is not a re-run on that LOCK either. Sibling receipts and instrumentation receipts that omit the field stay without it. Receipt bytes are not re-hashed here.

Advancing the tip does not re-execute `inventable_jetmod_probes.py` or `inventable_jetmod_instrumentation_status.py`. `REFUSED`, `EMPTY`, `ABSENT`, `PARTIAL`, and `REFUSED_NOT_24JET` stay honesty labels. It does not upgrade a refusal, `EMPTY`, `ABSENT`, `PARTIAL`, or `REFUSED_NOT_24JET` into `PRESENT` or `SUCCESS`.

24-jet STOP in `docs/math_status/STATUS_JETMOD.md`: the source-named subset stays 8 (6 MS-diag + `kappa_c2` + `s_f_fx`). Inventing toward 24 without a Drive/PROMOTE enumeration stays **REFUSED_NOT_24JET**.

`scientific_status_changed` stays false. OBL-H5-JETMOD stays OPEN. Green ≠ discharge.
