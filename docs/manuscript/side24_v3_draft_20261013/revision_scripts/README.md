# Revision scripts of the V3 draft

These are the scripts the author used to produce revisions of this draft. Each makes exact-match text edits: any target string that does not occur exactly once stops the script. They are kept here so that any agent can rerun or audit a revision from the repository without the author's session. `V3_CHANGE_LIST.md` describes what each revision changed and why. A script does not replace the review records cited there.

| Script | Revision | Run as |
|---|---|---|
| `apply_r2.py` | r2 | historical; it reads and writes absolute paths in the author's session workspace, so it is kept as a record only |
| `make_r8.py`, `make_r8b.py` | r8 | `python3 SCRIPT DRAFTDIR`, applied to r7 |
| `make_r9.py` | r9 | `python3 make_r9.py DRAFTDIR`, applied to r8 |
| `make_r10.py` | r10 | `python3 make_r10.py DRAFTDIR`, applied to r9 (the read AMEND is commit `d9f01444`) |
| `make_r12.py` | r12 | `python3 make_r12.py DRAFTDIR`, applied to r11 (`b033a118`) |

r0, r1 and r3–r7 were made by direct edits; their commits and the change list are the record. r11 (`b033a118`) was also made by direct edits: one sentence of Appendix A record [R1], the §0 date line, the README and the change list. After a script, run `sh build.sh` and regenerate `SHA256SUMS`, then compare the result with the revision's commit.

Check, from the repository root:

```
git show b033a118:docs/manuscript/side24_v3_draft_20261013/s3_proof.tex > /tmp/s3.tex   # and the other edited files
python3 docs/manuscript/side24_v3_draft_20261013/revision_scripts/make_r12.py /tmp      # then cmp against d550d238
```
