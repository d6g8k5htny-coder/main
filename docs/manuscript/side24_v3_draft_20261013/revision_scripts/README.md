# Revision scripts of the V3 draft

These are the scripts the author used to produce revisions of this draft. Each makes exact-match text edits: any target string that does not occur exactly once stops the script. They are kept here so that any agent can rerun or audit a revision from the repository without the author's session. `V3_CHANGE_LIST.md` describes what each revision changed and why. A script does not replace the review records cited there.

| Script | Revision | Run as |
|---|---|---|
| `apply_r2.py` | r2 | historical. It reads and writes absolute paths in the author's session workspace, so it is kept as a record only. |
| `make_r8.py`, `make_r8b.py` | r8 | two versions of the r8 script (b adds five citation and wording fixes). Each takes the draft directory and two read-status sentences; see its docstring. The r8 commits in the change list are the record of which text landed. |
| `make_r9.py` | r9 | takes the draft directory, date, supplement commit, branch, pages and read-status arguments; see its docstring. |
| `make_r10.py` | r10 | `python3 make_r10.py DRAFTDIR DATE D12_COMMIT D12_BRANCH`, applied to r9. The read AMEND that followed is commit `d9f01444`. |
| `make_r12.py` | r12 | `python3 make_r12.py DRAFTDIR`, applied to r11 (`b033a118`). It reproduces `d550d238`'s `s0_status.tex`, `s3_proof.tex`, `s6_literature.tex` and `README.md` byte for byte. |

r0, r1 and r3–r7 were made by direct edits; their commits and the change list are the record. r11 (`b033a118`) was also made by direct edits: one sentence of Appendix A record [R1], the §0 date line, the README and the change list. After a script, run `sh build.sh` and regenerate `SHA256SUMS`, then compare the result with the revision's commit.

Check, from the repository root:

```
git show b033a118:docs/manuscript/side24_v3_draft_20261013/s3_proof.tex > /tmp/s3.tex   # and the other edited files
python3 docs/manuscript/side24_v3_draft_20261013/revision_scripts/make_r12.py /tmp      # then cmp against d550d238
```
