# q0 / SIDE24 — research program repository

This branch is a **front door, not the work.** It carries this page, the
contributor and agent notes beside it, and the licence. Everything else lives
on the working branches below.

Read that literally. `main` holds three files. If you are looking for the
registers, the claim graph, the certified arithmetic or the Drive mirrors, you
are on the wrong branch and the table says which one you want.

---

## Where the work is

| branch | holds | what it is |
|---|---|---|
| `claude/drive-audit-github-migration-rrglpp` | 3,879 files | the Drive research program ported to git: 44 register tabs, the 4,456-item source map, the machine-checked claim graph, certified interval arithmetic, per-lane drivers, the runner and its receipts |
| `chatgpt/drive-github-hardening-20260919` | 4,235 files | the active integration lane, forked from the migration branch; most open pull requests target it |
| `main` | 3 files | this page |

Both working branches carry a full `README.md`, `CLAUDE.md`, `docs/RESEARCH_MAP.md`
and `docs/OPEN_PROBLEMS.md`. Those are the real documentation. This page
deliberately does not restate them, because a summary that drifts from its
source is the defect this repository exists to avoid.

---

## What this program does not claim

The single most important thing about this repository is that **every status
label in it is transcribed from a source register, never decided in the
repository.** Some consequences, current as of 2026-09-23 and carried verbatim
from the registers rather than assessed here:

* The five validity premises of **Theorem D1 v2.2(2)** are **OPEN**.
* Both pieces of **`D3-LEMMA-RN-UNIF`** are **OPEN**.
* **`OBL-H5-JETMOD`**, **`OBL-H5-ZBAND`** and **`OBL-H5-REMOTE-THRESHOLD`** are **OPEN**.
* **Theorem B** is **`RETRACTED_TO_CANDIDATE`**, retracted since `GP-AUD-187` (2026-07-24).
* **No original prize problem is solved.** Every prize claim carries `original_prize_closed: false`.
* **No independence credit has been awarded.** Every review on file sits at zero,
  and the gates that require organizational independence remain open.
* The 2D upper, 2D lower and 3D lifetime tracks are **composed nowhere**.
* The proposed Drive–GitHub execution contract is **PROPOSED / NOT DEPLOYED**.

A green CI run is a record that a computation ran. It is not a discharge, a
closure or a certificate, and nothing in this repository can promote a claim —
that is an operator decision recorded under `governance/`, and Dylan Roy is the
single final authority for canonical promotion, external release, permanent
deletion and machine-root replacement.

---

## What used to be on this page

Until 2026-09-23 `main` carried a 2025 single-file README for *A Reconstruction
of Physics from Multiscale Retrodiction Complexity and Gauge Representation
Minimization*, together with `body`, a 75 KB JavaScript manuscript generator.
That page presented a results table marking predictions "✅ Confirmed" and
"✅ Validated" — a grading style the present program does not use, applied to a
different body of work.

`body` is removed from this branch and the README replaced. **Neither file is
lost.** Both are preserved, byte-for-byte and with zero evidentiary authority,
on the migration branch:

```
legacy/complexity-physics-framework/README.md
legacy/complexity-physics-framework/manuscript_generator_docx.js
```

They also remain in this branch's own history at `4fc1d7c` and earlier. Nothing
here reassesses that work; it is moved out of the front door because it
describes a repository this is no longer, not because anyone adjudicated it.

---

## A note on the branch topology

`main` is not the trunk, and that is deliberate rather than an accident of
neglect. The migration landed on `main` on 2026-09-23 and was reverted the same
hour (PR #32, `cos/revert-pr2-never-main`); the integration lane has continued
on `chatgpt/drive-github-hardening-20260919` since. Anyone — human or agent —
opening a pull request should read [`AGENTS.md`](AGENTS.md) first and pick a
base deliberately. Targeting the wrong one produces a diff of hundreds of
unrelated files, which has already happened once and was caught by another
agent rather than by a check.

`main` currently defines no CI workflow, so nothing merged here is gated by
tests. The working branches define and run their own.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
