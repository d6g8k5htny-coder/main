# Workspace and branch guide

[Home](../README.md) · [Research guide](RESEARCH_INDEX.md) · [Run the checks](REPRODUCE.md)

## Choose the correct checkout

| Location | What is actually there |
|---|---|
| [main default branch](https://github.com/d6g8k5htny-coder/main) | Research navigation, collaboration entry, integration decisions and landing checks |
| [Math- default branch](https://github.com/d6g8k5htny-coder/Math-) | New mathematical candidates, calculations and tests, directly readable without ZIP extraction |
| [main hardening research branch](https://github.com/d6g8k5htny-coder/main/tree/chatgpt/drive-github-hardening-20260919) | The larger legacy research software/evidence tree and its source-specific numerical obligations |
| [Claude migration branch](https://github.com/d6g8k5htny-coder/main/tree/claude/drive-audit-github-migration-rrglpp) | Additional migration history and proposed work; compare exact heads before integration |

The name of a branch does not decide mathematical correctness or make it permanently unmergeable. A successful check of the default landing tree does not test the larger hardening stack. Historical `never-main`, date-cutoff and provider-specific 403 notes must be read in their exact scope, not applied as blanket claims about all newer sources or current connections.

## Drive and source identity

[Drive Research Home](https://docs.google.com/document/d/180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8) and the [research registers / Work Events](https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no) preserve source records and collaboration history. These require the relevant Drive access. A GitHub link does not change their sharing.

[google-drive](https://github.com/d6g8k5htny-coder/google-drive) contains selected public replicas only. The [source catalog](https://github.com/d6g8k5htny-coder/meta-framework) and [lookup tool](https://github.com/d6g8k5htny-coder/query-) bind public artifacts to exact paths/commits/hashes, not scientific acceptance. Private experiments stay private.

## Continue useful work

Read the actual source and its latest review, then the relevant claim discussion in [campaign #61](https://github.com/d6g8k5htny-coder/main/issues/61). Use a separate branch, coordinate overlapping paths, inspect the real diff and affected tests, and read back the result. The [owner delegation](../governance/OP-AUTONOMY-20260923-v2.1.md) authorizes ordinary project work without another permission loop.

[Open work](RESEARCH_INDEX.md#open-work) lists specific review and research targets. The existing hourly loop follows the campaign and successors; adding a document does not start another model or create a scheduler.

## Legacy setup

For the larger research tree rather than the landing pages:

```sh
git clone --filter=blob:none --single-branch --branch chatgpt/drive-github-hardening-20260919 https://github.com/d6g8k5htny-coder/main.git research-workspace
cd research-workspace
git rev-parse HEAD
```

Read that checkout's environment files and [execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md) before installing dependencies or running programs. Do not copy assumptions from a different branch, the old 2025 presentation, or another provider's runtime.

[Historical material](../history/2025/README.md) remains available as history. Its preserved original bytes are still covered by the landing-custody check.
