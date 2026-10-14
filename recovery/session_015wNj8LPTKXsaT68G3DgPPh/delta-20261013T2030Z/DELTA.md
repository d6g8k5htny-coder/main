# C8 recovery delta, 13 October 2026, about 20:30Z

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), for Dylan Roy. Scientific effect NONE.

This adds to the C8 recovery copy made at 15:03Z (commit `eceaf907`; see `../MANIFEST.json` and `../README.md`, both unchanged). It answers the owner's standing request [main#307/6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988): still-local work is preserved so that an interruption does not force anyone to re-derive it.

## Files

| Path | Bytes | SHA-256 | What it is |
|---|---:|---|---|
| `site_run_readback.py` | 4,358 | `1705d6c5c358b9b64b3294d6c7f35ae75c8c995f57cbcbb1103039559cf3b90a` | C8's independent exact-run readback for the current-site workflow |
| `validation/38432795042/READBACK.json` | 4,236 | `3f3cc88fae9dffdeb98c243157578e41d335c3aaa257324fc59572530b9d14ee` | The script's validation output on the first production run |
| `validation/38493627274/READBACK.json` | 5,078 | `99c1bd0e2bfd2abe6ef9424951d9d80c05f5ab0668ae5cbdc44c62dd68658029` | The script's output on the first genuine scheduled run that succeeded (added 14 October, 01:15Z) |
| `watch.sh` | 3,427 | `631131a629f4e3219a966798c0373daae6abbb6eb635554c7baf4a26fcaf3afd` | C8's read-only watcher (version 16 since 23:49Z on 13 October; the row was updated) |
| `push_run_evidence.py` | 3,282 | `9611505da52cf1cf7d8d9ce23c6f32a9438c3fa5f0c21d26f4d2ca0d6964ee2b` | Builds the evidence record for push run `38489017622` from its fetched artifacts and receipts (added 14 October, 01:38Z) |
| `assembled_docs_digest.py` | 1,227 | `abdf787e813b8d5a9d81faa6e9985993485ab84e860a422096cb7a74a8fe2209` | Recomputes the workflow's assembled-docs digest from a `github-pages` artifact without extracting it (added 14 October, 01:38Z) |
| `validation/38489017622/QUALIFICATION.json` | 5,928 | `0d381aa3020c58b8230c41bff2058645af26d6b4e2ba292d4789fc60775b856e` | The evidence record behind the main#384 push-run qualification (added 14 October, 01:38Z) |
| `run_bind.py` | 3,949 | `d924fbf3347e661e1607d1327f290950bf25dac8cfddde53efee9be81f1590b1` | A generic version of the binding check that works for any `current-site.yml` run (added 14 October, 03:18Z) |
| `validation/38503039478/QUALIFICATION.json` | 4,632 | `ac300e2cbbd159f3baf79b79f451aadc56086391f2eb52364a572614f8b16ee4` | The evidence record behind the main#397 push-run qualification (added 14 October, 03:18Z) |
| `validation/38503039478/READBACK.json` | 5,079 | `9138efc2a4885bceca2cd794927b18d47e16d2be3adae0129d5847715575c282` | The final live readback for that run, 03:15:58 to 03:16:02Z (added 14 October, 03:18Z) |
| `validation/38549074505/QUALIFICATION.json` | 4,632 | `c886a1bbb3f2cb0edd0454321f5ae62ed36d9f4473236df86c54e8d4fcc4747b` | The evidence record behind the main#386 push-run qualification (added 14 October, 10:34Z) |
| `validation/38549074505/READBACK.json` | 5,064 | `44d91e9222ba54bd98bdf88d84129da9480a09129ac333a6fbb5379ed71a7a83` | The final live readback for that run, 10:32:15 to 10:32:19Z (added 14 October, 10:34Z) |

`SHA256SUMS` in this directory lists the same hashes.

### `site_run_readback.py`

- **Usage:** `python3 -I site_run_readback.py RUN_ID [OUTDIR]`. It needs `gh api` read access to `d6g8k5htny-coder/main`.
- **What it reads:**
  - the one announced run;
  - the receipts branch `codex/site-publication-receipts`: its head, three parents and `latest.json`;
  - the three public files: `snapshots/index.json`, `current-meta.json` and `current.json` under `https://d6g8k5htny-coder.github.io/main/site/`.
- **What it checks:** each public file's bytes and SHA-256 against the receipt. It writes `READBACK.json` and copies of the three files into `OUTDIR`.
- **What it never does:** list runs, dispatch, rerun or write anything remote. The sole observer, `01a11d37`, owns run discovery.

### `validation/38432795042/READBACK.json`

- Run `38432795042` was an ordinary `push` run that concluded `success`. The receipt head at the time was `142a06aa`.
- All three public files matched the receipt. The read ran from 18:32:00Z to 18:32:03Z.
- This is evidence that the script works. It is not the current publication: later runs (for example push run `38455779711` for main#390) have since published.

### `watch.sh`

- It polls the coordination threads and C8's open items every 60 seconds, read-only.
- Its comment patterns and state paths are specific to this session.

## Open obligations at this checkpoint

1. **Site.** When the sole observer `01a11d37` announces the first genuine `event=schedule` run of `.github/workflows/current-site.yml`, run `site_run_readback.py` on it and post the result on [main#387](https://github.com/d6g8k5htny-coder/main/pull/387). As of this checkpoint, no scheduled run has been announced.
2. **Math-#454** (C8 is the author). Head `d517287c` has a non-Claude delta review PASS ([5508838177](https://github.com/d6g8k5htny-coder/Math-/pull/454#pullrequestreview-5508838177)). It waits for a new, distinct non-Claude integrator. After it merges, C8 does the author landing check. C8 never merges it.

## Done since 15:03Z, all recorded on GitHub

- DOC358 readback: PASS ([main#358 review 5508651060](https://github.com/d6g8k5htny-coder/main/pull/358#pullrequestreview-5508651060)).
- BQ38 r3: delivered in [main#275/6153450094](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153450094), and the readback passed ([6153678987](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153678987)).
- Math-#371: merged as `9844f785` on the owner's action-time approval.
- Math-#454 amendment: `d517287c`.

The session's drafts of those posts are not copied here, because the posted versions are the record.

## Excluded

- No credentials, personal data or private account or host information.
- No copies of public site files or repository files, which are recoverable from their own sources.

## Update at 22:03Z

- **Math-#454 is merged.** Muse Spark (Meta) merged it at 21:47:01Z as `a6116d8a`. The parents are `b64e70c9` and `3bd0d6cf`, and the tree is `239307f0`, the tested composition. C8's author landing check is [6155954331](https://github.com/d6g8k5htny-coder/Math-/pull/454#issuecomment-6155954331), and the three post-merge checks passed. Obligation 2 above is closed.
- **The site.**
  - The first genuine scheduled run, [38473133881](https://github.com/d6g8k5htny-coder/main/actions/runs/38473133881), fired at 21:46:54Z, so the cron works. It failed at 21:52:46Z, before deployment.
  - The cause is `tests/test_reader_links.py:1738`. That check looks for the bare string `331` in raw page HTML, and the string occurs inside the Math- commit ID `a6116d8a…89331d15…`.
  - The repair belongs to the source owner `01a13536`, and 017 reviews it ([main#396](https://github.com/d6g8k5htny-coder/main/pull/396), 6155995009 / 6156024487 / 6156037550).
  - Obligation 1 now reads: run `site_run_readback.py` on the first SUCCESSFUL genuine scheduled run after that repair lands, then post the result on main#387.
- `watch.sh` is updated to version 12. It watches `main` #387, #390 and #358, and the #454 state check was moved to `3bd0d6cf`.

## Update at 01:15Z on 14 October

- **The site obligation is closed.** Obligation 1 above is done.
  - Run [38493627274](https://github.com/d6g8k5htny-coder/main/actions/runs/38493627274) is the first genuine scheduled `current-site.yml` run on corrected `main`: attempt 1, head `8bc089f8`, a descendant of main#398's merge `465f99ec`. It ended in `success` at 01:08:32Z.
  - The successor observer, Claude 01N, handed it to C8 at [387/6158645208](https://github.com/d6g8k5htny-coder/main/pull/387#issuecomment-6158645208). C8's independent readback passed at [387/6158672118](https://github.com/d6g8k5htny-coder/main/pull/387#issuecomment-6158672118):
    - the run's own receipt pair names it: prepared `4a8a795c`, then observed `334e2fe2`;
    - all three public files are byte-identical to that receipt.
  - The output is `validation/38493627274/READBACK.json`. The board pointer is [307/6158676853](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6158676853).
- **Work for other lanes since 22:03Z.** Each item was posted only after a check for new posts, and each was read back after posting.
  - **Math-#329.** On the coordinator's match and handoff, C8 did one current-base carry (`7cd5eda1`, tree `d0a1c4f4`) and one guarded merge (`46f2a289`, 23:59:42Z). Fresh checks passed before and after. The receipts are [Math-#329/6157717516](https://github.com/d6g8k5htny-coder/Math-/pull/329#issuecomment-6157717516) and [6157813953](https://github.com/d6g8k5htny-coder/Math-/pull/329#issuecomment-6157813953).
  - **Math-#336.** Eligibility HOLD, because an Anthropic session has read it ([6157525280](https://github.com/d6g8k5htny-coder/Math-/pull/336#issuecomment-6157525280)).
  - **Math-#433 custody.** C8 posted the evidence ([307/6158232366](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6158232366)). It then ran the fetch leg: all four digests matched and the inspection was clean. Because every repository here is public, C8 has no private handoff route, and it recorded that blocker ([307/6158582996](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6158582996)). C8 holds no custody role.
  - **Data notes.** The Math-#329 refusal record ([307/6156951464](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6156951464)), the Math-#380 carry mechanism ([Math-#380/6157147928](https://github.com/d6g8k5htny-coder/Math-/pull/380#issuecomment-6157147928)), and C8's status and offer ([307/6157232875](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6157232875)).
- **Still open.** C8 holds no claim and owes no obligation. Its offer stands.

## Update at 01:38Z on 14 October

- **The main#384 push-run qualification is done and released.**
  - dot authorized the scope through Claude 01Cz ([384/6158764717](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6158764717)). C8 accepted at 01:23:31Z ([6158774360](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6158774360)), and 01Cz activated it at 01:25:15Z ([6158794792](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6158794792)).
  - C8 posted the checkpoint at 01:28:56Z ([6158837825](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6158837825)) and the delivery and release at 01:36:59Z ([6158932334](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6158932334)).
- **The result.** Push run [38489017622](https://github.com/d6g8k5htny-coder/main/actions/runs/38489017622) on head `8bc089f8` is QUALIFIED at evidence scope.
  - Its four artifacts agree three ways: C8's fresh downloads, GitHub's records and the run's own upload logs.
  - C8 recomputed the receipt's artifact digest `579ec894…` from the deployed `github-pages` tar: 330 files and 11,266,444 B.
  - The receipt pair is `7e90e748` (prepared) and `1736e449` (observed).
  - Its publication was superseded at 01:08:32Z by scheduled run `38493627274`.
- **Output.** `validation/38489017622/QUALIFICATION.json` holds public digests and identities only. `push_run_evidence.py` produced it, and `assembled_docs_digest.py` is the stand-alone digest check.
- **Expiry.** The run's `github-pages` artifact expires at 2026-10-15T00:18:27Z, and the other three artifacts on 2026-11-13. After that the tar-level recomputation cannot be repeated from GitHub, and this record and the delivery comment remain.
- **Still open.** C8 holds no claim and owes no obligation. Its offer stands.

## Update at 03:18Z on 14 October

- **Both preservation gaps are closed by another lane.** The OpenAI dot custody agent preserved the original artifacts in an owner-only archive and verified each by reading it back.
  - main#384's `github-pages` zip: [384/6159751191](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6159751191).
  - Math-#433's four originals: [Math-#433/6159873821](https://github.com/d6g8k5htny-coder/Math-/pull/433#issuecomment-6159873821).
  - Every reported digest equals C8's record. C8's report on its own capability is [384/6159605504](https://github.com/d6g8k5htny-coder/main/pull/384#issuecomment-6159605504): its upload leg is blocked, and it can keep copies only on a best-effort basis.
- **The main#397 push-run qualification is done and released.**
  - dot's offer: [397/6159978557](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6159978557). C8 accepted at 03:09:30Z ([6160003497](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6160003497)), and dot activated at 03:12:14Z ([6160034274](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6160034274)).
  - Checkpoint: [6160057005](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6160057005), at 03:14:16Z. Delivery and release: [6160092520](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6160092520), at 03:17:21Z.
- **The result.** Run [38503039478](https://github.com/d6g8k5htny-coder/main/actions/runs/38503039478) on head `9c63f11b` is QUALIFIED at evidence scope.
  - Its four artifacts agree with GitHub's records and the run's upload logs.
  - The assembled-docs digest `790c7827…` was recomputed from the deployed tar.
  - Receipts: `21198697` (prepared), then `0e987fb6` (observed). The publication was being served at 03:16:02Z.
  - The site pipeline's blobs are unchanged from `8bc089f8`.
  - Disclosed: the Math- input is `46f2a289`, C8's own Math-#329 merge.
- **Still open.** C8 holds no claim. Its offer stands.

## Update at 10:34Z on 14 October

- **main#386 landed.** dot's OpenAI review agent published its manual review: `PASS_ENGINEERING_SCOPED` on head `ff2cff11` ([386/5516156993](https://github.com/d6g8k5htny-coder/main/pull/386#pullrequestreview-5516156993)). Muse Spark (Meta) then merged it at 10:22:04Z as `c7ed3f8d` ([386/6166181790](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166181790)).
- **The main#386 push-run qualification is done and released.**
  - dot's offer: [386/6166219270](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166219270). C8 accepted at 10:26:21Z ([6166248522](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166248522)), and dot activated at 10:30:16Z ([6166315233](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166315233)).
  - Checkpoint: [6166330813](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166330813), at 10:31:08Z. Delivery and release: [6166365820](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166365820), at 10:33:13Z.
- **The result.** Run [38549074505](https://github.com/d6g8k5htny-coder/main/actions/runs/38549074505) on head `c7ed3f8d` is QUALIFIED at evidence scope.
  - Its four artifacts agree with GitHub's records and the run's upload logs.
  - The assembled-docs digest `1891ec00…` was recomputed from the deployed tar.
  - Receipts: `57312ba4` (prepared), then `cfc20e5d` (observed). The publication was being served at 10:32:19Z.
  - The site pipeline's blobs are unchanged.
  - Disclosed: the Math- input is still `46f2a289`, C8's own Math-#329 merge.
  - The `github-pages` artifact expires at 2026-10-15T10:26:31Z.
- **Still open.** C8 holds no claim. Its offer stands.

## Update at 11:01Z on 14 October

- **All three push-run `github-pages` zips are now in durable custody.** The OpenAI dot custody agent preserved them in the owner's private evidence destination and verified each by reading it back. Every reported digest equals C8's record.
  - main#397: [397/6160374746](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6160374746), at 03:42:03Z.
  - main#386: [386/6166752844](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6166752844), at 10:55:31Z. The zip is `7378566f…` (2,482,720 B), its tar `bbfd89ba…` (11,642,880 B), with 330 files.
  - main#384 was already covered in the 03:18Z update.
- **main#388, the atlas follow-up issue, is now the active site follow-through.** dot routed it at 10:59:01Z ([388/6166810512](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6166810512)):
  1. packet access and publication route: General or Source, or another holder of their exact packet;
  2. frozen-candidate review: Claude `017`;
  3. guarded integration: Muse Spark (Meta).

  C8 is not named. It holds no copy of the original General commit or tree: neither object is in any of this session's ten clones. So C8 has no capability to offer on step 1, and it takes no role on silence.
- **Watcher.** From 11:02Z, C8's watcher reports every main#388 post, not only C8 mentions. `watch.sh` is unchanged; only its local `allposts` list grew. #388 is an issue, so the script's PR-state check does not apply to it.
- **11:32Z check.** There has been no new post on #388 since 017's at 11:02:16Z, and none on #307 since 11:00Z. No new PR has opened on `main`.
- **Still open.** C8 holds no claim. Its offer stands.

## Update at 16:21Z on 14 October

- **Outage.** This session's container was down from shortly after 12:57Z until about 16:07Z. On restart, C8 read every watched thread for the gap and re-armed its watcher.
- **12:56Z check-in.** All 12 local artifact copies (main#384, #397 and #386) and all three `QUALIFICATION.json` records still match their posted digests. Nothing was posted.
- **Math-#455: no role.** At 13:23Z dot asked C8 for integration capability, having mistaken C8 for an OpenAI agent. dot withdrew the offer at 13:25Z, before C8 saw it ([Math-#455/6169417865](https://github.com/d6g8k5htny-coder/Math-/pull/455#issuecomment-6169417865)): "no #455 role is assigned to C8". An OpenAI agent, DOT455-I1, now owns that integration.
- **main#393: one open scope, accepted and awaiting activation.** dot reports that #393 merged at 15:48:15Z as `aff60c50` (tree `63d4be9d`, parents `c7ed3f8d` and `f84dc39c`).
  - dot offered C8 the publication qualification of push run 38592549661 at 15:49:39Z ([393/6172365636](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6172365636)).
  - C8 accepted at 16:13:37Z ([393/6172847586](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6172847586)), saying the answer was late because of the outage.
  - Disclosed exposure: C8 is the same family as #393's reviewer, Claude 01Fi, so organizational independence is 0. C8 has no content contact with #393.
  - Window: the checkpoint is due within 10 minutes of dot's separate activation, and the expiry is 55 minutes after it. C8 reads nothing under the scope before then.
  - At 16:20Z there was no activation yet. The watcher now reports every main#393 post, and the next check-in is at 19:14Z.

## Update at 16:31Z on 14 October

- **The main#393 push-run qualification is done and released.**
  - dot activated it at 16:23:18Z ([393/6173037864](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6173037864), `PUB393-C8-20261014T1622Z`).
  - Checkpoint: [6173084950](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6173084950), at 16:25:38Z. Delivery and release: [6173152947](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6173152947), at 16:29:10Z.
- **The result.** Run [38592549661](https://github.com/d6g8k5htny-coder/main/actions/runs/38592549661) on head `aff60c50` is QUALIFIED at evidence scope.
  - Its four artifacts agree with GitHub's records and the run's upload logs.
  - The assembled-docs digest `c06d0eb6…` was recomputed from the deployed tar.
  - Receipts: `b87dbdf8` (prepared), then `786de40f` (observed). The publication was being served at 16:27:35Z.
  - The site pipeline's blobs are unchanged.
  - The receipt just before it belongs to run `38566583355` (`9c61b106`). C8 recorded it by identity only.
  - Disclosed: the Math- input is still `46f2a289`, C8's own Math-#329 merge, and C8 is the same family as #393's reviewer.
  - The `github-pages` artifact expires at 2026-10-15T15:52:48Z.
- **Evidence.** `validation/38592549661/` holds `QUALIFICATION.json` and `READBACK.json`. C8's fresh artifact copies stay session-local in `m393art`.
- **Still open.** C8 holds no claim. Its offer stands.

## Update at 18:25Z on 14 October

- **#393 custody and close.** dot closed the #393 qualification at [393/6173211616](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6173211616). DOT393-CUSTODY-1 preserved its `github-pages` zip at [393/6173330036](https://github.com/d6g8k5htny-coder/main/pull/393#issuecomment-6173330036), and its digests equal C8's record. All four push-run zips (#384, #397, #386, #393) are now in custody.
- **#388 is now a NEW successor.**
  - The design was frozen at [388/6173589417](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6173589417).
  - DOT388-AUTHOR-1 (OpenAI) is writing it as draft [main#399](https://github.com/d6g8k5htny-coder/main/pull/399).
  - Browser evidence comes from the hosted `browser-smoke` job. Claude 017 found that this environment cannot run the browser contract unmodified, and C8 confirmed the same three limits locally.
- **The owner's 18:17Z directive** ("Continue to advance the project. Resolve any issues. Help any and all agents in whatever ways possible.") reached this session. Claude 01N also relayed it, at [307/6175251175](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6175251175).
- **C8's availability:** [307/6175271821](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6175271821).
  - **Withdrawn:** item 1, the #400 integration, at [307/6175320035](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6175320035), after 01Cz disclosed reading part of #400.
  - **Still offered:** the #399 post-merge publication qualification, the ISS-006 R1 hash readback as Muse's fallback, and guarded integration, provenance and run readbacks.
- **Math-#329 merge-tail data for Muse** (in the same post).
  - The post-merge run is `38487555792`. Its artifacts are 11860610337 (expires 2026-11-13) and 11860610404 (expires 2027-01-11).
  - C8 has no Drive copy and its upload leg is blocked, so a custody lane must own the transfer.
- **#227 is done.** dot verified the owner's four setting clicks at 02:04:58Z ([227/6159267531](https://github.com/d6g8k5htny-coder/main/issues/227#issuecomment-6159267531)), and the issue is closed.
- **Watcher v17.** It adds a repo-wide scan of `main` for comments that name C8, so an offer on any PR is seen. `allposts` now includes 393 and 399.
- **Still open.** C8 holds no claim.
