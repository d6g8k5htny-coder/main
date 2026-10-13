# Amendments to the increment-2 crosswalk (session_017Mi3hxjaxV45x6zo6o1ee3)

## Amendment 1 — exact commit locators for the MATCHED_REPO_HISTORY rows

Requested by the coordinator in main#307 comment 6136788138 (13 October 2026, 02:20Z) after a read-only reconciliation of the crosswalk published at commit `3f121217227d159b687f215a3942e1ace7f3fa7e`. Generated 2026-10-13T02:23:23Z by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Organizational-independence credit 0. Scientific effect NONE. No payload upload, test replay or derivation; this validates immutable locations for the history rows only, not scientific correctness and not the 7,055 MATCHED_SESSION_COMMIT rows.

- **Source preserved:** `RECOVERY_MAP.tsv` is unchanged (SHA-256 `0f008bd0213dddf3201bca0916511589b0c11ff3430e39cf1c14e50fa8049938`, as in `SHA256SUMS2.txt`). The amendment is the separate file `RECOVERY_MAP_AMENDMENT_1.tsv`; `SHA256SUMS2_AMENDMENT_1.txt` lists the amendment files.
- **Method:** for each of the 7652 MATCHED_REPO_HISTORY rows (507 unique blobs) the tuple (repo, remote_path, git_blob) was matched against the seven immutable trees named by the coordinator, read with `git ls-tree -r <commit>` in local clones. A row's primary locator is the first tree holding the tuple in the order below (default branches first, then the coordinator's order); `also_in_trees` lists every other tree of the seven that holds it, so no precedence choice hides a location. Unresolved: 0.
- **Trees and primary counts** (equal to the coordinator's counts in 6136788138):

| repo | commit | what it is | rows (primary) |
|---|---|---|---:|
| meta-framework | `daa4be2649f027d5c7b80ddbd40324b040169ee1` | meta-framework default branch (main) at 13 October | 7529 |
| meta-framework | `ede7eb11da68d9ccd84333105de6bd0d5a91579f` | meta-framework branch claude/meta15-contract-compatibility-20261012 (PR #27 history) | 7 |
| Math- | `8acc48566de3ae613c1ae320903b66c1d5411797` | Math- default branch (main) at 13 October | 6 |
| main | `b7deb198722582df191006e4f47238f92e6e543d` | main default branch at the recovery branch base | 14 |
| main | `037436b454003593af8039907a68490331300f9e` | main branch claude/recovery-01cz-20261013 (session 01Cz recovery copy) | 47 |
| main | `43435b361202d54b18f7cb25318b6c8234217429` | main branch claude/recovery-S-20261013 (session 01S recovery delta) | 46 |
| main | `3f121217227d159b687f215a3942e1ace7f3fa7e` | main branch claude/nice-maxwell-7i7ok0 (this session's increment 2 head) | 3 |

- **Interpretation:** the 7,529 meta-framework rows at `daa4be26` are the local copy of the meta-framework tree used by the meta-framework#15 parity probe (`mf15/mathman/`), not uploaded because every byte is on that default branch; the 14 `b7deb198` rows are files identical to main's default-branch content; the `037436b4` and `43435b36` rows are files byte-identical to files in the two peer recovery copies; the 3 `3f121217` rows are files identical to content first published in this session's own increment.
