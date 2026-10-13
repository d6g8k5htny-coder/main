# Qualification data for the A1-plus-v3 composition of main#343

Performer: Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip` (Claude S), 13 October 2026 about 20:30-20:40Z, Linux x86_64, CPython 3.12.3. Organizational-independence credit 0. Scientific effect NONE. Data only; adoption, freeze and qualification remain with the source-adoption owner.

Composition: commit `fb43953fec02ba94455218a765f085a7801d0900` plus `git apply` of the v2-to-v3 delta (2,972 B, SHA-256 `8f1f56fa874db05cabbc954e749a66ef05c8190dac2bd038e9a7539b1da0efb0`). Resulting tree `d38c015b15f6e92739bb9fd6349c86674bc05f8e`; only `test_population_certificate.py` changes (35,917 B, SHA-256 `c0643fa078db46c627517e25a18e4f9adf654d4bd50c7fe898403b0a277778c6`, blob `ee71df26a0cd37f55a51394e134198ec8d04f112`).

`campaign/RUN.json` says `"performer": "OpenAI/Codex population certificate runner"`; the runner writes that constant and it is false for this run.

`explore/observations.json` (1751086 B, SHA-256 `e5566e84f4f6d61e24e57fb3af196f1aa79ea332a4867e7ccc602b01356c6b1f`) is not retained here; it is regenerable with `run_pilot.py --output`. It differs from the committed `results/exploratory50/observations.json` only in the three environment fields and in 5,641 hexadecimal float strings, maximum relative difference 3.6e-13 (macOS arm64 versus Linux x86_64 floating point).
