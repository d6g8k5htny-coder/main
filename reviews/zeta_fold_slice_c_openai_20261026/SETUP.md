# Observed setup and execution provenance

Performer: OpenAI / Codex, foreground `/root` in workspace `57388d5ddbb4`.
The interface exposes no portable model-version or platform-session identifier.
This is a new execution, not a predecessor's receipt. Source-exposed, not blind
or human; different provider from the Anthropic packet author, shared project
principal, organizational-independence credit 0. Scientific effect NONE.

The following commands were executed in this workspace before the numerical run:

```sh
git clone --depth=1 --no-tags https://github.com/d6g8k5htny-coder/main.git ul-main
cd ul-main
git fetch --no-tags origin 8f1d846daeb03990a84a92fa6e42c9903575c412 0a7e5145b55f020e02e7782fed318266d13c2447 78976ae7d412427ac15d2ce01786ca1c24d6b066
git worktree add --detach ../zeta-run 0a7e5145b55f020e02e7782fed318266d13c2447
git switch -c codex/zeta-openai-execution-20261026
```

The worktree command succeeded and reported:

```text
Preparing worktree (detached HEAD 0a7e514)
HEAD is now at 0a7e514 Merge pull request #442 from d6g8k5htny-coder/codex/sg-hf-source-composition-20261018-root01a11cbf
```

The execution checkout and evidence checkout are separate. The wrapper checks
the execution HEAD and clean tracked/untracked status before execution. It does
not inventory ignored raw files. The new worktree creation, coefficient-fitting
lines in `logs/rs_validate.stdout`, generator commands, and raw-file modification
times are evidence for regeneration; no separate pre-run ignored-file inventory
was captured. The private fresh check separately refuses an existing coefficient
file and records the refitted coefficient hash.

Environment preparation, from the workspace parent:

```sh
uv python install 3.13.16
uv venv --python 3.13.16 zeta-venv
uv pip install --python zeta-venv/bin/python numpy==2.5.3 mpmath==1.4.1
```

The install completed with those two exact package versions. `environment.json`
is the numerical interpreter's own runtime report, including executable hash,
mpmath backend, thread pins and NumPy build/BLAS configuration. NumPy's optional
`pyyaml` recommendation is preserved in `logs/environment.stderr`; it is not a
numerical-job error. Pillow was absent and no optional PNG previews were requested.

The exact frozen driver, comparator and timing-rule bytes were extracted with
`git show` from `8f1d846daeb03990a84a92fa6e42c9903575c412`, never reconstructed.
Their source directory tree is `6138261110c848d0d4d56f53aadc27864ebe237f`.
The execution packet tree is `bca06db86581b2ff878fa321c4040c7b8f7bd55d`.

`run_replay.py`'s executed bytes are preserved. It records the environment;
the subsequent `build_evidence.py` verifies the recorded dependency/backend/thread
pins, interpreter, working directories, command multiset and both parent-lane
exit codes. The frozen driver records stdout SHA-256 and stderr size at capture.
The final manifest hashes all retained stdout and stderr bytes; capture-time
stderr SHA-256 was not separately recorded by that driver.

The auxiliary fresh-check shell process exited 0 (exec session `95937`). Its
exact command and stream identities are in `fresh_execution_receipt.json`.
Overall start timestamp and total wall duration were not separately captured
for that auxiliary process; per-window durations remain in `fresh/result.json`.
No execution was repeated to fill that metadata gap.
