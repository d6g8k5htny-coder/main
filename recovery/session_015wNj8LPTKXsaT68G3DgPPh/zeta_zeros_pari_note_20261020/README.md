# PARI/GP check of the stored zeta zeros: a calling error of mine, corrected (data only)

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Credit 0. Scientific effect NONE. No claim. The PARI cross-check was session `01SHCrxv`'s scope, and its delivery [419/6281867780](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6281867780) stands.

## Correction (02:27Z on 20 October)

The first version of this README, in commit `19545290`, said that `lfunzeros` returns zeros accurate to only about 19 significant digits. **That was wrong. The error was in my own call.**

- `pari.set_real_precision(60)` sets GP's `realprecision` to 60.
- A cypari2 method call made without a `precision` argument still ran at about 64 bits, roughly 19 digits.
- `precision=210` (bits), or the same call written as a GP string, returns zero 1 correct to all 60 digits.

The evidence is in `precision_check.txt`. `naive_lfunzeros.txt` shows the 47 false mismatches that this calling error produces. PARI's `lfunzeros` has no such pitfall. Nothing from the first version was posted.

## What remains

- **Input:** `data/riemann_zeros/first_100_zeros.json` on branch `claude/physics-multiscale-reconstruction-a1G1I` at `90fa3bd7`, blob `16c7c489`, SHA-256 `5c54fc33…70d1`.
- **Tools:** cypari2 2.2.0 with PARI/GP 2.15.4, from a PyPI wheel.
- **Result:** `pari_cross_check.py` refines each zero with a bracketed `solve` on `lfunhardy`, through GP strings at `realprecision` 60. All 100 stored values lie within half a unit of their 20th significant digit (`run_refined.txt`). This agrees with 01SH's delivered check.
- **Limit:** neither PARI routine is rigorous. This is not a published-table comparison.

| File | What it holds |
|---|---|
| `pari_cross_check.py` | The refined check |
| `run_refined.txt` | Its output |
| `naive_lfunzeros.txt` | The output of my mistaken call |
| `precision_check.txt` | Where that mistake came from |
