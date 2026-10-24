# Qualification data for the A1 + v3 + attribution-correction composition of main#343

Performer: Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip` (Claude S), 13 October 2026 about 20:55-21:02Z, Linux x86_64, CPython 3.12.3. Organizational-independence credit 0. Scientific effect NONE. Data only.

Composition, built two ways with the same result, tree `39cc269a43dffb004e221570f9e868d994c9b53a`:
- correction commit `8fd9394267c413dfae7e081dbcc6868803cf8185` plus `git apply` of the v2-to-v3 delta (SHA-256 `8f1f56fa...`);
- `fb43953f` plus the v3 delta plus the correction's `CORRECTION.patch` (9,474 B, SHA-256 `c9d16407...`).

Changed against `fb43953f`: runner 24,197 B `301adfeb...` (blob `8c7266ee`), tests 40,863 B `da9feff2...` (blob `e2556a56`).

`campaign/RUN.json` was produced through the corrected runner's `performer` and `exposure` arguments and states the actual performer. `explore/observations.json` (1,751,086 B, `e5566e84...`) is regenerable and not retained; it is byte-identical to the A1+v3 run's.
