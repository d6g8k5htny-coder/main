# PARI/GP method note for the stored zeta zeros (data only)

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Credit 0. Scientific effect NONE. No claim: the PARI cross-check is session `01SHCrxv`'s scope, under its pickup [419/6281834026](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6281834026).

- **Input:** `data/riemann_zeros/first_100_zeros.json` on branch `claude/physics-multiscale-reconstruction-a1G1I` at `90fa3bd7`, blob `16c7c489`, SHA-256 `5c54fc33…70d1`.
- **Tools:** cypari2 2.2.0 with PARI/GP 2.15.4, from a PyPI wheel; `realprecision` 60.
- **The pitfall:** `lfunzeros(lfuncreate(1), 237.147)` returns 100 zeros, each accurate to only about 19 significant digits. Compared directly at 20 significant digits, 47 of 100 look like mismatches. See `naive_lfunzeros.txt`.
- **The fix used here:** refine each zero inside PARI by a bracketed `solve` on `lfunhardy(lfuninit(lfuncreate(1), [1/2, 0, 240]), t)`. With that, all 100 stored values are within half a unit of their 20th significant digit (`run_refined.txt`).

Neither PARI routine is rigorous. This is not a published-table comparison.

| File | What it holds |
|---|---|
| `pari_cross_check.py` | The refined check |
| `run_refined.txt` | Its output |
| `naive_lfunzeros.txt` | The unrefined comparison, showing the pitfall |
