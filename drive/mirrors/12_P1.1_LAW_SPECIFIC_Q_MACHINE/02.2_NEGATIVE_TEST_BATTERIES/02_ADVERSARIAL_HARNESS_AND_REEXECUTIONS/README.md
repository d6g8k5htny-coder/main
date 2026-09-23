# `02.2_NEGATIVE_TEST_BATTERIES/02_ADVERSARIAL_HARNESS_AND_REEXECUTIONS`

Drive folder id `1PXS_2_3fR0y2XvuUMimpe51pwetD55gQ`. The 2026-09-17 inventory gives this
folder **2 items**, both native Google Docs, both held here as text exports — reading
copies, not the objects.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `GP-DATA-058-v1.0 — Adversarial harness for GP-DATA-054 qualification defects.export.txt` | reading copy | 4,566 | 3,767 |
| `GP-DATA-060-v1.0 — Adversarial harness reexecution against canonical 7f061dd7 source.export.txt` | reading copy | 4,923 | 4,460 |

## The status banners, verbatim

`GP-DATA-058-v1.0` records

> Status: EXECUTED — 3 OF 3 DEFECT CONTROLS CONFIRMED

> Canonical impact: NONE

and warns about its own publication surface:

> Google Docs may normalize formatting. The hash above refers to the locally executed harness bytes. The code is reproduced below for review and independent reimplementation.

`GP-DATA-060-v1.0` records

> Status: EXECUTED AGAINST SHA256 7f061dd71032c76e785e259a115d3dad50e6adf5274dc8b3cf7104b9c2bd4c22 — 3 OF 3 DEFECT CONTROLS CONFIRMED

> Authority: none

and its result section reads

> Source reconstructability: PASS.

> Frozen self-tests: PASS, 13/13.

> Adversarial controls: 3/3 qualification defects confirmed.

> Production eligibility: FAIL-CLOSED.

> Independent-review requirement: OPEN.

## What this does not establish

Both documents describe runs performed by their author's line, on that line's machine, and
both say so. Neither was executed here. What they establish *in their own terms* is a set
of confirmed defects in a different instrument — a negative result about that instrument's
qualification, which is exactly the kind of result the 02.2 leaf card asks for, and which
moves nothing forward. Their own labels are the ones to read: production eligibility
fail-closed, independent review open. Note also that `GP-DATA-058`'s Google Docs warning
applies word for word to what is stored here: the export normalises formatting, so the
listing it contains is a reading copy of a source, not the source.
