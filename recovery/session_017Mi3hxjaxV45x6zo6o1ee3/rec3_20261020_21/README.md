# Recovery increment 9: 20–21 October records (retention only)

This folder holds the working records behind work by Anthropic Claude, session
`session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017), on 20–21 October 2026: drafts
of the posts it made, verification scripts and logs. They are kept here so
that another agent can re-run the checks without that session's context,
following Dylan Roy's standing direction on living project knowledge
([main#419/6321304701](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6321304701)).

This folder is retention only, not for merge. Credit 0. Scientific effect NONE.
The posted GitHub comments are the canonical records; a draft here may differ
from what was posted in whitespace or in a footer added by the server.
Copies of other agents' posts and of repository files are left out. In these
copies, model names are replaced with session IDs; the posted comments keep
their original wording.

| Folder | Tag | Work | Canonical records |
|---|---|---|---|
| `m453/` | integration | main#453 landed as `bb1a8654` | conditions [453/6282247376](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6282247376); landed + release [453/6282370349](https://github.com/d6g8k5htny-coder/main/pull/453#issuecomment-6282370349) |
| `m455/` | integration | main#455 landed as `d43a7c63` | conditions [455/6282046257](https://github.com/d6g8k5htny-coder/main/pull/455#issuecomment-6282046257); landed + release [455/6282149640](https://github.com/d6g8k5htny-coder/main/pull/455#issuecomment-6282149640) |
| `r2/` | replay check | main#451 R2 replay: Cursor summons, and the check of record `78976ae7` | check [458/6282353648](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6282353648) |
| `q46/`, `q46b/` | integration (held) | query-#46, first attempt: HOLD, then a late release | HOLD [q46/6285437907](https://github.com/d6g8k5htny-coder/query-/pull/46#issuecomment-6285437907); release [q46/6289345829](https://github.com/d6g8k5htny-coder/query-/pull/46#issuecomment-6289345829) |
| `q46r2/` | integration | query-#46, second attempt, under the usage-limit route; also Dylan's main#465 decision record | `verify_landed.py` and the drafts here |
| `quota/` | status | stop notice for the OpenAI lane's quota | draft here |
| `r81/` | review | CL-route paper read of R81 V1 K1–K2 and K19–K21, plus the Q14 floor proof | IDs in `pickup_id.txt` and `delivery_id.txt` |
| `scan1135/` | scan | helper for the eligible-work scan | `heads.py` |
| `b301/` | coordination | board pointer for #470, #466, #467 and the Codex review status | [419/6301493678](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6301493678) |
| `coord/` | summoner check | main#467: the xAI Cursor read, then a check of its re-run claims | [467/6301597331](https://github.com/d6g8k5htny-coder/main/pull/467#issuecomment-6301597331); addendum [467/6301616135](https://github.com/d6g8k5htny-coder/main/pull/467#issuecomment-6301616135) |
| `m493/` | summoner check | Math-#493: the xAI Cursor review, then a check of its re-run claims (`runchecks.sh`, `real_shell.py`) | [Math-#493/6301824223](https://github.com/d6g8k5htny-coder/Math-/pull/493#issuecomment-6301824223) |
| `m456/` | review | main#456: PASS_SCOPED at `d9ed0101` | [456/6301775858](https://github.com/d6g8k5htny-coder/main/pull/456#issuecomment-6301775858) |
| `m466/` | integration | main#466 landed as `bf60f9d2` (`verify_landed.py`) | conditions [466/6302256580](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6302256580); landed + release [466/6302572648](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6302572648) |
| `m470/` | owner decision | Math-#470: Dylan ended GROK-H02's claim, and the integration stayed open | [Math-#470/6320441095](https://github.com/d6g8k5htny-coder/Math-/pull/470#issuecomment-6320441095); pointer [419/6320445103](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6320445103) |
| `dot/` | coordination | check-in with dot at Dylan's request (`watch_dot.sh`) | [419/6321647892](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6321647892) |

`SHA256SUMS` lists every other file in this folder. To check it, run
`sha256sum -c SHA256SUMS` from inside the folder.

Scripts were written for the session's scratchpad layout. To re-run one,
change its input paths, and run it against a fresh clone at the commit it names.
