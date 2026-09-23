# GP-LB-STAT-004A -- W8 RECOVERY CORRECTION v1.0

Date: 2026-08-05/06

Program: q0 / SIDE24 LB-RATE lower campaign

Status: **ADDITIVE FACTUAL CORRECTION / W8 REMAINS NONCLOSED / NO CONTROL EFFECT**

This note corrects one custody sentence in `GP-LB-STAT-004` without mutating that frozen raw carrier.

## Correction

`GP-LB-STAT-004` stated that the W8 script was absent from the delivery. A complete outer-archive comparison found two relevant W8 source files outside `K3_SIDE24_LB/`:

- `verify_lambda_grid_v2.py`: 49,020 B; SHA-256 `579ef6cd88d543ec14db45558e5e7c1899078095e0df295c7d46fd1810273831`; Drive ID `1hnd7jbFwcCQIuOcbrzilMQoymsOzFQZb`.
- `mutate_lambda_grid.py`: 3,320 B; SHA-256 `3c76fcb0235a4ad5319b77b00dadec85c27f89a43f900dcfc6b4e4c104f01794`; Drive ID `1WdndQBVooDf39dfbi5QATdleNpiKkgkc`.

Two additional incomplete forensic traces were also found and landed:

- `finish.sh`: 819 B; SHA-256 `38a24eae290b5cc9085736ba3d03cb7e60cd507ed2d170f3152c9e5a7f5cd70b`.
- `mini2.log`: 2,301 B; SHA-256 `2556c3ddedf36e180704c59071c30e47834cb1aaab7497957e166428c888ab83`.

The recovery subfolder is Drive ID `1v-JgJKl9q5HIlss10tBZJwXmsQSBWG5J`.

## What remains absent

The outer archive contains no W8 `transcript_O.txt`, identity stamp, S6 excerpt, or W8 `HASHES.txt`. The K3 W8 package contains only the report, mutation receipts, and an incomplete normal transcript; its transcript stops before the complete O/F, assembly-constant, and hash/identity receipt sequence.

## Mathematical disposition unchanged

The recovered source does not prove the sampled `H-B3` premise, does not turn sampled finite differences into an analytic cellwise third-derivative supremum, and does not supply a positive unconditional Lambda-side constant. W8 remains **FORMAL NONCLOSURE / H-B3 UNPROVED / RECEIPTS INCOMPLETE**. `K3-THM-001` remains **REFUTED AS WRITTEN / NONCONTROLLING**. The two-dimensional lower campaign remains OPEN.

This work changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status, ratification status, or q0 package status. Any such change requires separate operator adjudication.
