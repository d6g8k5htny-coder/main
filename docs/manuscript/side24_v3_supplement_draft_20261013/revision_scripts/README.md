# Revision scripts of the supplement

These are the scripts the author used to produce checkpoint draft 13 from draft 12 (`59d20df5`). They are kept here so that any agent can rerun or audit the revision from the repository without the author's session. The record of what changed is in `CONCORDANCE.md` ("Checkpoint draft 13"), and the review records are in main#389.

| Script | What it does | Run as |
|---|---|---|
| `make_d13.py` | Edits the TeX: adds the proof of Proposition S12.24 and the dependency sentences in S12.22, S12.23 and Lemma S5.20. Every edit is an exact match and must occur once. | `python3 make_d13.py SUPPDIR` on draft 12 |
| `make_d13_docs.py` + `cfg_d13.json` | Edits the status section, the README and the concordance. | `python3 make_d13_docs.py SUPPDIR cfg_d13.json` |

`checks/dep_graph_check.py` (in the supplement directory) was written directly and is committed with draft 13.

`make_d13.py` on draft 12 reproduces `supp_s5_selection.tex` and `supp_s12_parent.tex` at `77d3f804` byte for byte, and `make_d13_docs.py` with this `cfg_d13.json` reproduces `supp_s0_status.tex` there. The README and `CONCORDANCE.md` at `77d3f804` were edited after the script: both say 162 pages, not the config's 161, and the S5.40 sentence names the window and the covariances through S5.21. The read sentence in `cfg_d13.json` is the one in that concordance. The read's two AMENDs were then applied by hand as `e7573e34` (the S5.20 sentence and the concordance count) and `d28e6ab4` (the §S0 count). After the scripts, run `build.sh` and regenerate `SHA256SUMS`.

**Draft 14.** `make_d14.py`, run on draft 13 (`f8b71cab`), removes the provenance wording from the mathematical sections and writes the status-section, README and concordance entries. The ordering fix in `checks/dep_graph_check.py` and the regenerated `checks/dep_graph_check.out.txt` were made directly, and the concordance records them.

**Draft 15.** `make_d15.py`, run on draft 14 (`860d82b4`), adds the radius and parameter matching sentences and writes the status-section, README and concordance entries. The hardening of `checks/dep_graph_check.py`, the new `checks/test_dep_graph_check.py` and the regenerated `checks/dep_graph_check.out.txt` were written directly; the concordance records them.
