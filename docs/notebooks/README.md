# Public research notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d6g8k5htny-coder/main/blob/main/docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb)

## Quick start in your browser

1. [Read the saved notebook output on GitHub](SIDE24_PUBLIC_COEFFICIENTS.ipynb), or use **Open in Colab** above to run it without a local installation.
2. In Colab, choose **Runtime → Run all** and follow any connection prompt. Runtime startup and the source download can take time.
3. Check the printed source commit, byte count and SHA256, then read the exact coefficient endpoints for dimensions 2 and 3. The optional chart follows the table.

The notebook fetches one public JSON file at a fixed 40-character Math- commit and verifies its size and SHA256 before parsing. A fresh run needs internet access to `raw.githubusercontent.com`; a fetch or identity failure stops the run.

The notebook preserves the published decimal endpoint strings and original `scientific_acceptance: false`. Its optional two-bar chart uses approximate midpoints and is **NON-CERTIFYING**. No random field or lifetime solver runs, and the notebook cannot change repository status. The exact table needs only Python's standard library; the chart uses Matplotlib if available.

## Run locally

Use Python 3.11 or later. The notebook's dependency groups are:

| Part | Dependencies |
|---|---|
| Fetch, hash verification and exact table | Python standard library: `hashlib`, `json`, `urllib.request`, `decimal` |
| Local notebook interface | JupyterLab and its installed dependencies |
| Optional approximate chart | Matplotlib and its installed dependencies; the notebook skips the chart when Matplotlib is absent |

From the root of a checkout of this `main` repository, create a separate environment. These commands use a macOS/Linux shell and put the environment beside the checkout:

```sh
python3 -m venv ../universal-law-notebook-env
. ../universal-law-notebook-env/bin/activate
python -m pip install jupyterlab
python -m jupyterlab docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb
```

Select the environment's Python kernel and run the cells in order. For the optional chart, install Matplotlib in the same activated environment before starting JupyterLab:

```sh
python -m pip install matplotlib
```

On Windows, create the environment with `py -3 -m venv ..\universal-law-notebook-env` and activate it with `..\universal-law-notebook-env\Scripts\Activate.ps1` in PowerShell; the remaining `python` commands are the same. If you already have a notebook interface and Python kernel, use those directly.

These are notebook setup dependencies, not a dependency specification for the separate research branch or query package. The commands install compatible available package versions rather than a locked environment. Record your Python, JupyterLab and Matplotlib versions when sharing a replay. See the [JupyterLab installation guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) and [Matplotlib installation guide](https://matplotlib.org/stable/install/index.html) for platform-specific setup.

## Source and scope

Input: [`Math-@9d7b6802424fb4715b31999066aafca8ee2f3cca:coefficients/side24_v1/ENCLOSURE.json`](https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/ENCLOSURE.json), 1090 bytes, SHA256 `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`.

See the [scoped coefficient review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) and [current program status](https://github.com/d6g8k5htny-coder/main/blob/main/STATUS.md) separately. Viewing or replaying these numbers does not accept a parent theorem. The public inventory remains [`../public-math/`](../public-math/sources.json); these files create no separate catalog.
