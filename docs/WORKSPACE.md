# Workspace and tools

## Current working locations

The [hardening research branch](https://github.com/d6g8k5htny-coder/main/tree/chatgpt/drive-github-hardening-20260919)
contains the active mathematical software and evidence. The
[Claude migration branch](https://github.com/d6g8k5htny-coder/main/tree/claude/drive-audit-github-migration-rrglpp)
holds additional work and proposed repairs. Inspect actual differences and current
heads before combining them: a small advertised change can otherwise bring along
unrelated ancestry. No branch name makes work correct or permanently unmergeable.

Use [pull requests](https://github.com/d6g8k5htny-coder/main/pulls) and
[Actions](https://github.com/d6g8k5htny-coder/main/actions) for live collaboration and
execution results. The [research execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md)
describes the research-side commands; read it in the checkout actually being used.

The [Drive Research Home](https://docs.google.com/document/d/180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8)
and [coupled research registers](https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no)
provide research memory and Work Events. These links require the relevant Drive
access; a public GitHub page does not make linked Drive files public. Historical
permission wording in an older mirror does not revive revoked owner restrictions.

## Tools and execution

All participating models have Dylan's permission to download, install, create, and
use useful tools. Choose the actual runtime for the task rather than assuming the
2025 package described real installed software. For example, a clean checkout of
the current research branch starts with:

```sh
git clone --filter=blob:none --single-branch --branch chatgpt/drive-github-hardening-20260919 https://github.com/d6g8k5htny-coder/main.git research-workspace
cd research-workspace
git rev-parse HEAD
```

Read the checkout's environment files and workflows before installing dependencies.
This session prefers project-local environments, identifiable upstream sources, and
recorded versions so experiments can be replayed without disrupting another worker.
Tools do not need to be installed merely to demonstrate that permission exists.
Disclose actual capability or credential failures rather than inventing a successful
installation or routing routine permission back to Dylan.

## What this main-branch check covers

The landing workflow's `verify` job checks the declared local documentation links,
exact custody of the two relocated historical files, and its own test cases. It uses
Python's standard library and requires no project package installation.

```sh
python3 tools/workspace_landing_check.py
python3 -m unittest discover -s tests -p test_workspace_landing.py -v
```

This check does not execute or certify the research on another branch, inspect remote
link contents, validate Markdown anchor targets, or test every future file. Integrating
research code into `main` should bring its relevant verification with it; a green
landing check is not full research CI. Agents may replace or expand this workflow as
the workspace changes.

[Current authority](../governance/OP-AUTONOMY-20260923-v2.1.md) ·
[Home](../README.md) · [Historical material](../history/2025/README.md)
