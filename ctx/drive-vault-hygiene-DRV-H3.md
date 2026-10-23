# ctx — drive-vault-hygiene — DRV-H3

- Bot: drive-vault-hygiene (executor for Dylan-approved lane DRV-H3: "Yes, do DRV-H3 and open the PR.")
- Base: main@a487cfd4; Math-@9f3fb691; rules main#475 comment 6369972075 (RETRO-MATH-BIND-02).
- Method: Google Drive connector, READ-ONLY (search/list/metadata only). No file content opened; no move/delete/rename/share changes.
- Inventory sources: live search 2026-10-23 ~12:00 CT (04/HOLD incl. LPW-V4-BRICK subfolder; 90 top level); Oct-15 walk snapshot for 06, 07/04 census + 07 FROZEN_SUPPLEMENT, 05/06_ARCHIVE, 02/03_LEGACY_CODE, 01 archive/superseded/historical/legacy folders.
- Excluded: 99_DO_NOT_OPEN and anything named DO_NOT_OPEN (not listed, asserted absent from inventory); CUB 6049593844 / 6049713530 not opened; 04 governance folders and 07/01 reading volumes (not math / not archive).
- Dropbox: no access (no connector). Not worked around.
- imports/historical/DRV-H3/ copies: DEFERRED — lane was metadata-only; binding any row requires a later import lane.
- Note: W9 mirror folder now resides in 90 (id prefix 1fW9leeJ…).
- Output: docs/retro/DRV-H3/DRV-H3.md, docs/retro/DRV-H3/DRV-H3_drive_map.csv.
- No merge, no self-review; integrator = agent 9.

credit 0 · sci NONE · OBL OPEN · no flags
