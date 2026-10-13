# Seven historical Pages sources: provenance receipt

Recorded 11 October 2026 for the seven `github-pages` artifacts listed in the
8 October [original digest receipt][digests] and [source follow-up][followup].
This consolidates those existing public records; it does not recover an archive.

**Citation rule: cite the run, full commit SHA and `docs/` tree, not an
unavailable ZIP.** Each digest and size below is a historical transcription
reported by Grok Bot agent 8 as GitHub API metadata. Those digests were **never
checked against archive bytes** by that reporter or this receipt's author.

## Exact sources and reported artifacts

All rows belong to `d6g8k5htny-coder/main`. Native run metadata read on
11 October 2026 identifies attempt **1**, conclusion **success**, and workflow
`dynamic/pages/pages-build-deployment` for every row. Success is the recorded
run conclusion, not an independent archive-content or deployment audit.

| Run / attempt | Historical artifact ID | Full run head SHA | Full `docs/` tree | Historical reported bytes | Historical reported archive SHA-256 — UNVERIFIED |
| --- | --- | --- | --- | ---: | --- |
| [37570707637](https://github.com/d6g8k5htny-coder/main/actions/runs/37570707637) / 1 | 11460278692 | `21231f12e4b3896a397757ce68e97c2580134e04` | `abcff6fb259879cc46dc664f18462952686bfec7` | 810200 | `8fcad2bf9b3e2568c80f64618f4f588ebccc72ee533004d299ebbac3b3a8c492` |
| [37578268103](https://github.com/d6g8k5htny-coder/main/actions/runs/37578268103) / 1 | 11462594525 | `70d1cdf1ae639fa709379577beb40e1d0d74a85f` | `abcff6fb259879cc46dc664f18462952686bfec7` | 810194 | `379017be58ca156610cb1faf824ede0724af9e38d2302d99b45f2be74a89193a` |
| [37629296768](https://github.com/d6g8k5htny-coder/main/actions/runs/37629296768) / 1 | 11485841860 | `7c4cef8c6a2ca4b6f984e5e45108037c2caec4e1` | `8bd9dd5fa98ef18fa65dc2e6c211f9e77766f146` | 810390 | `e75f9eee4b93831d25a825ec15e415b5a8278573befdcf9d62a1c139d541930d` |
| [37665223438](https://github.com/d6g8k5htny-coder/main/actions/runs/37665223438) / 1 | 11503245230 | `3bcf0e34b15dc38270b163263d62b690884e68df` | `8bd9dd5fa98ef18fa65dc2e6c211f9e77766f146` | 810389 | `fd9e0ab66464d0abb99f981491413c676563e6dcdf755181dc6436b9fa06f870` |
| [37681717146](https://github.com/d6g8k5htny-coder/main/actions/runs/37681717146) / 1 | 11509895334 | `8d210d165d2dbace42aa09d6c9627d56c289aaaa` | `bc0ab60477370700a5c614b073eb505903b6c6b0` | 815423 | `113ecf77107debe7acf7302885fac75c02ddd9fd3fdf70ee405764a0645d2313` |
| [37699735670](https://github.com/d6g8k5htny-coder/main/actions/runs/37699735670) / 1 | 11517097228 | `b673100e7cceb995703afbd9c01df82611296355` | `94e49a6a158fb8c96191cff1194887ed663f1b9a` | 819926 | `cb89bfd022c909b0d296a18afbcd33d281a68765ee78ad59f63a981480edaa0b` |
| [37700971942](https://github.com/d6g8k5htny-coder/main/actions/runs/37700971942) / 1 | 11517119215 | `1653fe6740e9e2c8d36a47101bf7bb50c7aae273` | `94e49a6a158fb8c96191cff1194887ed663f1b9a` | 819947 | `37706750da2792ba734f99ae756070cafc8fc8b343829e62abed4bc69e80efa0` |

## What was checked

At the 11 October 2026 observation ending **21:08:17 UTC**, native GitHub
run responses supplied all seven full heads. The Git commit and root-tree APIs
resolved each head's `docs` tree entry. Independently fetched Git objects gave
the same seven trees via `git rev-parse <full-head>:docs`; each resolved object
had type `tree`. The table's artifact IDs, reported sizes and reported digests
were transcribed from the original receipt's seven P3 rows, not recovered from
original ZIP bytes.

Each run's current artifacts-list response returned `total_count: 0` and
`artifacts: []`. That establishes absence from these lists at this observation.
The [task handoff][task] reports the seven artifacts expired on 8 October.
No ZIP endpoint, original job log, archive extraction, local rebuild or old
Pages rerun was used for this receipt.

The [earlier source audit][followup] reports all seven heads reachable from
main. This receipt pins the full identities, with working base
`1e7a2aa0290ddc48d3f706bb4fcee569e81fb44d`
(root tree `ecdda8ea1225a83a4c2b8b7d1ce209a4a6219c96`).
The durable Git objects identify available repository source; they do not
identify the complete contents of an unavailable historical artifact.

## Limits and correction

There are **three** equal-`docs/`-tree pairs: rows 1–2, 3–4 and 6–7; row 5 is
a singleton. The follow-up's opening “two pairs” sentence disagrees with its
own three examples and table. Its historical text remains unchanged.

Each pair has different **reported** archive digests. Equal source trees and
different reported digests do not by themselves prove archive-member equality,
a packaging-only difference, the exact packaging cause, or byte-for-byte
reproducibility. The follow-up's stronger rebuilding and packaging explanations
remain attributed historical conclusions, not newly verified findings here.
The current live site cannot establish these seven past artifact contents.
Do not rerun an old Pages deployment to fill this custody gap.

## Authorship and review route

Author: OpenAI / Codex `/root`, session
`01a0bbb5-2fcb-77f0-b78b-4d220ddd7ab2`, delegated for Dylan Roy.
Historical digest/source reporter: Grok Bot agent 8, as identified in the
linked original comments. Source-exposed AI work; no human review is claimed.
Same-provider technical review carries zero organizational independence.

The [pickup][pickup] binds R17 claim
`c2676267-7be8-4200-bb22-c26720261011`, Work Events 1898.
The requested different-provider nonauthor read is tracked in the resulting
pull request; this file does not assert that it has occurred. This is an
engineering provenance receipt with **no scientific status effect**, no
original archive-byte custody, and no new deployment evidence.

[digests]: https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6050620161
[followup]: https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6050655588
[task]: https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6113485242
[pickup]: https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6113762993
