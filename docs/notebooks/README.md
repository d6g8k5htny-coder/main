# Public research notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d6g8k5htny-coder/main/blob/main/docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb)

Open [SIDE24_PUBLIC_COEFFICIENTS.ipynb](SIDE24_PUBLIC_COEFFICIENTS.ipynb) on GitHub or in Colab, then run the cells in order. It fetches one public JSON file at a fixed 40-character Math- commit and verifies its size and SHA256 before parsing.

The notebook preserves the published decimal endpoint strings and original `scientific_acceptance: false`. Its optional two-bar chart uses approximate midpoints and is **NON-CERTIFYING**. No random field or lifetime solver runs, and the notebook cannot change repository status. The exact table needs only Python's standard library; the chart uses Matplotlib if available.

Input: [`Math-@9d7b6802424fb4715b31999066aafca8ee2f3cca:coefficients/side24_v1/ENCLOSURE.json`](https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/ENCLOSURE.json), 1090 bytes, SHA256 `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`.

See the [scoped coefficient review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) and [current program status](https://github.com/d6g8k5htny-coder/main/blob/main/STATUS.md) separately. Viewing or replaying these numbers does not accept a parent theorem. The public inventory remains [`../public-math/`](../public-math/sources.json); these files create no separate catalog.
