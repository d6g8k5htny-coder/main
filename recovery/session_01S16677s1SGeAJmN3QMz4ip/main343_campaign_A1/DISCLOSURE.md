# Performer disclosure for these run records

Every `RUN.json` in this directory says `"performer": "OpenAI/Codex population certificate runner"` and `"exposure": "Source-exposed same provider/team"`. The runner `experiments/universality/two_parameter_cusp_population/run_population_certificate.py` writes those two strings as constants. They are false for these runs.

The actual performer is Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip` (Claude S), acting for Dylan Roy (delegated AI work), on 13 October 2026 about 19:53-19:58Z, Linux x86-64. Organizational-independence credit 0. Scientific effect NONE.

The files are kept byte-exact as the runner wrote them. Source: commit `fb43953fec02ba94455218a765f085a7801d0900` of `d6g8k5htny-coder/main` (the A1 successor of main#343), called through `runA1.py`, which invokes the runner's own `run()` with a fresh output directory and the default budgets.

| directory | interpreter | flags |
| --- | --- | --- |
| `out_fb43953f` | CPython 3.13.16 | `-B -S` |
| `out_fb43953f_O` | CPython 3.13.16 | `-B -S -O` |
| `out_fb43953f_python3.12` | CPython 3.12.3 | `-B -S` |
| `out_fb43953f_python3.11` | CPython 3.11.17 | `-B -S` |
