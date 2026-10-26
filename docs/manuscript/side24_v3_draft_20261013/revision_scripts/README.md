# Revision scripts of the V3 draft

These are the scripts the author used to produce revisions of this draft. Each makes exact-match text edits: any target string that does not occur exactly once stops the script. They are kept here so that any agent can rerun or audit a revision from the repository without the author's session. `V3_CHANGE_LIST.md` describes what each revision changed and why. A script does not replace the review records cited there.

| Script | Revision | Run as |
|---|---|---|
| `apply_r2.py` | r2 | historical. It reads and writes absolute paths in the author's session workspace, so it is kept as a record only. |
| `make_r8.py`, `make_r8b.py` | r8 | two versions of the r8 script (b adds five citation and wording fixes). Each takes the draft directory and two read-status sentences; see its docstring. The r8 commits in the change list are the record of which text landed. |
| `make_r9.py` | r9 | takes the draft directory, date, supplement commit, branch, pages and read-status arguments; see its docstring. |
| `make_r10.py` | r10 | `python3 make_r10.py DRAFTDIR DATE D12_COMMIT D12_BRANCH`, applied to r9. The read AMEND that followed is commit `d9f01444`. |
| `make_r12.py` | r12 | `python3 make_r12.py DRAFTDIR`, applied to r11 (`b033a118`). It reproduces `d550d238`'s `s0_status.tex`, `s3_proof.tex`, `s6_literature.tex` and `README.md` byte for byte. |
| `make_r13.py` | r13 | `python3 make_r13.py DRAFTDIR`, applied to `8130d3a1` (r12 plus its documentary additions). It edits `refs.bib`, `s6_literature.tex`, `s7_disclosure.tex`, `s0_status.tex` and `README.md`; the change list, PDF and `SHA256SUMS` were then written as for r12. |
| `make_r14.py` | r14 | `python3 make_r14.py DRAFTDIR`, applied to `4994f890`. It edits `refs.bib`, `main.tex`, `s0_status.tex` and `README.md`. |
| `make_r15.py` | r15 | `python3 make_r15.py DRAFTDIR`, applied to `ba1b904d`. It edits `refs.bib`, `s0_status.tex` and `README.md`. |
| `make_r16.py` | r16 | `python3 make_r16.py DRAFTDIR`, applied to `18543a20`. It edits `appA_records.tex`, `s0_status.tex` and `README.md`. |
| `make_r17.py` | r17 | `python3 make_r17.py DRAFTDIR`, applied to `fcc83ab5`. It edits `appA_records.tex`, `s0_status.tex` and `README.md`. |
| `make_r18.py` | r18 | `python3 make_r18.py DRAFTDIR`, applied to `29ca667d` after `build.sh` was rewritten by hand. It edits `s0_status.tex`, `README.md` and `SUBMISSION_READINESS.md`; then `sh build.sh --update-sums`. |

r0, r1 and r3–r7 were made by direct edits; their commits and the change list are the record. r11 (`b033a118`) was also made by direct edits: one sentence of Appendix A record [R1], the §0 date line, the README and the change list. After a script, run `sh build.sh --update-sums` (since r18 the build is reproducible and the script copies the PDF and rewrites the manifest), then compare the result with the revision's commit.

Check, from the repository root:

```
mkdir -p /tmp/r12check
for f in s0_status.tex s3_proof.tex s6_literature.tex README.md; do
  git show b033a118:docs/manuscript/side24_v3_draft_20261013/$f > /tmp/r12check/$f
done
python3 docs/manuscript/side24_v3_draft_20261013/revision_scripts/make_r12.py /tmp/r12check
for f in s0_status.tex s3_proof.tex s6_literature.tex README.md; do
  git show d550d238:docs/manuscript/side24_v3_draft_20261013/$f | cmp - /tmp/r12check/$f
done
```
