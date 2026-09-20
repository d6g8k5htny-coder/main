# Research source transfer — 2026-09-19 continuation

This dated, additive snapshot accounts for **4,924 objects**: **4,112 stored
files**, **758 folder metadata records**, and **54 held files**. There are no
unresolved fetches or inventory hash mismatches in the final ledger. One native
export failed transiently and succeeded on retry.

| Disposition | Source identities |
|---|---:|
| Inventory-exact bytes, newly fetched or shared by exact digest | 1,730 |
| Inventory-exact bytes already present elsewhere in this repository | 313 |
| Native Google exports, each with its own export digest | 1,969 |
| Newly discovered raw-file snapshots, without an older inventory digest | 100 |
| Folder metadata only | 758 |
| DO_NOT_OPEN content held | 40 |
| Personal content held | 14 |

The frozen 4,456-object inventory is unchanged. `scope.json` adds 16 folders and
452 file identities. Document search omitted some archive, code and data MIME
types; a separate listing reconciled 744 permitted folders, each below the
1,000-item limit. Native revision metadata was subsequently observed for all
1,969 exports. Revision timestamps and export metadata timestamps are retained
separately: these exports were **not requested by revision ID**. This is a
bounded, dated content snapshot, not a transactional backup, complete revision
history, or guarantee of every Google-specific feature.

## Verify, find and reconstruct

From the repository root, with Python 3.11:

```sh
python tools/drive_coverage.py --json
python tools/drive_coverage.py --search 'Hermite'
python tools/drive_coverage.py --extract-id DRIVE_ID --output work/original-file.ext
```

The extractor verifies all scoped identities before writing, refuses an
existing output, and does not execute source content. `coverage.jsonl` maps each
Drive ID to an existing repository file or a SHA-256 object. `objects.json`
orders that object's chunks and maps each chunk to an inert ZIP pack. Chunks
are at most 1 MiB; packs are built from at most 4 MiB of uncompressed chunks.
Repeated content shares storage while all source identities and contexts remain
separate. Standard ZIP tools plus the JSON index suffice to reconstruct bytes.

The package manifest pins every pack and control file. Its identity is added
to the existing CI manifest-coverage configuration; the coverage checker also
reconstructs every logical object and compares source hashes and byte counts.
A missing source, reading copy, folder, or hold cannot count as verified raw
source bytes. `eligible_sources_accounted_for` refers only to this declared
scope. The checker reports unresolved rows explicitly if a later snapshot has
any; it does not certify global Drive discovery completeness.

Four byte-identical historical containers fail a bounded decode check and one
reaches the inspection depth limit. These findings are retained in the ledger;
the original bytes are preserved, not repaired or executed. A copied archive
is not automatically a usable archive or mathematical evidence.

## Authority and mathematics

Every source retains its original context. Legacy, quarantine and accessibility
material acquires no evidentiary authority from this transfer. Protected and
personal contents are not stored. No exported register, frozen body, claim,
premise, obligation, gate or grade is changed.

The separate [Hermite–Gaussian candidate](../../../../docs/HERMITE_GAUSSIAN_ENVELOPE.md)
supplies an exact interval evaluator and a written global two-axis envelope.
Its 66 reproduced derivative cases through total order 10 have conditional
lattice tails below 10^-60 on the explicitly specified axial displacement box.
The envelope remains unreviewed, its review flag remains false, and all
independence credit is zero. It supplies no full 24-jet/six-pin/band certificate.
