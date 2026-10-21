# Revision scripts of the supplement

These are the scripts the author used to produce checkpoint draft 13 from draft 12 (`59d20df5`). They are kept here so that any agent can rerun or audit the revision from the repository without the author's session. The record of what changed is in `CONCORDANCE.md` ("Checkpoint draft 13"), and the review records are in main#389.

| Script | What it does | Run as |
|---|---|---|
| `make_d13.py` | Edits the TeX: adds the proof of Proposition S12.24 and the dependency sentences in S12.22, S12.23 and Lemma S5.20. Every edit is an exact match and must occur once. | `python3 make_d13.py SUPPDIR` on draft 12 |
| `make_d13_docs.py` + `cfg_d13.json` | Edits the status section, the README and the concordance. | `python3 make_d13_docs.py SUPPDIR cfg_d13.json` |

`checks/dep_graph_check.py` (in the supplement directory) was written directly and is committed with draft 13.

The output of these scripts is commit `77d3f804`. The read's two AMENDs were then applied by hand as `e7573e34` (the S5.20 sentence and the concordance count) and `d28e6ab4` (the §S0 count). `cfg_d13.json` is the configuration as run, so its page count (161) and read sentence are those of `77d3f804`, before the AMENDs. After the scripts, run `build.sh` and regenerate `SHA256SUMS`.
