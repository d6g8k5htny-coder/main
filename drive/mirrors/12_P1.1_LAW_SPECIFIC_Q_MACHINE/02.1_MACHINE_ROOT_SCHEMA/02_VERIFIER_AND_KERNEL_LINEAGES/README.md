# `02.1_MACHINE_ROOT_SCHEMA/02_VERIFIER_AND_KERNEL_LINEAGES`

Drive folder id `1lJL3lOCjCerz1I3jQafUNoF-QHxF6W-P`. The 2026-09-17 inventory gives this
folder **7 items**, all `text/x-python-script`, and all 7 are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `gate kernel v1_1.py` | byte-exact | 27,279 |
| `gate kernel v1_2.py` | byte-exact | 35,686 |
| `gate_kernel_v2_0.py` | byte-exact | 95,017 |
| `q0_llm_verifier .py` | byte-exact | 15,366 |
| `q0_llm_verifier_v2.py` | byte-exact | 20,334 |
| `q0_llm_verifier_v4.py` | byte-exact | 36,551 |
| `q0_llm_verifier_v5.py` | byte-exact | 68,019 |

The odd filenames — the two with a space in place of an underscore, and the one with a
space before the extension — are the Drive titles as the inventory records them, kept
unchanged. `q0_llm_verifier_v3.py` is not a Drive object in this lane; the lineage skips it.

## The status banners, verbatim

`gate kernel v1_1.py` states its own scope before anything else:

> GATE KERNEL v1.1 — executable enforcement of the Master Gate File (v1.1) SHELL.

> This kernel checks the mechanical SHELL of every gate:

`gate_kernel_v2_0.py` announces itself as a breaking successor rather than a patch:

> GATE KERNEL 2.0

> Standalone executable shell for Gate Framework Master v1.2 (reconciled).

> BREAKING CHANGES FROM 1.x

> * Claim status, warrant, assumption role, and reasoning mode are separate.

> * Content hashes are recomputed from canonical claim content.

> * Every dependency edge pins the target's canonical SHA-256.

`q0_llm_verifier_v5.py` records its own dependency chain:

> v5 is a strict extension of q0_llm_verifier_v4:

> * v3 remains the numerical and risk-ledger dependency of v4.

## What this does not establish

Seven executable files are stored here and **none of them was run**. Nothing was imported,
no battery was reproduced, and no number any of them prints was checked. The words *kernel*,
*enforcement*, *verifier* and *validation* in these filenames and docstrings are the
sources' own vocabulary for what the code is meant to do; a passing gate inside one of them
is a statement by that program about its own inputs, not a certificate, and this repository
neither ran it nor adopted it. `C047R-VAL-002` elsewhere in this lane records that several
of these files are not standalone-runnable without co-located project modules, which is a
fact about reproducing them, not a defect. Storing the bytes fixes their identity against
the 2026-09-17 inventory and establishes nothing else.
