# Recover a removed reading list

Scientific effect: NONE. This is a local reading-workspace repair, not mathematical review or source authentication.

One Remove or Clear previously discarded saved sources and searches, including in peer tabs. The Library now offers one-step **Undo removal** in the same page. It captures actual removed entries inside the existing storage lock and merges them into the latest validated list; it does not replace the list with an old snapshot. Peer additions survive. Deduplication, source-metadata conflict checks,50-item and128KiB limits remain unchanged. A refusal keeps recovery available for retry. A successful Undo consumes it; a later actual removal replaces it, while queued no-ops do not. Recovery is not persisted across reloads. Existing storage/lock failures remain explicit tab-only behavior.

A native button uses the existing toolbar, focus handling and status announcement. Successful keyboard Undo returns focus to the reading-list summary; delayed completion preserves newer outside focus. The existing author button display rule required a scoped hidden-control CSS rule, verified by a retained failing first candidate. No new dependency, schema, service, source/classification/statement, Atlas control, certificate or workflow changes.

## Evidence

- Exact base: Main `a15d2fb682fdb397584787143fffcf7290753ecf`. Actual independently downloaded prior complete-source archive matches all four intended source preimages byte-for-byte before edits; see `preimages.json`.
- Baseline: six actual Remove/Clear regressions fail at the missing Undo control at1440light/390dark/320light, zero page errors. Independent read-only audit also reproduced durable two-tab loss of a real saved source and search after reload/Cite→Back.
- First candidate:36 of48 pass;12 visibility assertions fail because the existing button display overrides the native hidden attribute. Original failures retained. Scoped CSS fixes the displayed control without changing recovery data behavior.
- Final:80 real-browser cases pass at1440light/dark,390light/dark and320light, zero page errors. Exact fixture metadata restoration, peer additions/dedup, metadata conflicts,50-item/128KiB refusal and retry, latest/no-op recovery, queued focus, peer-only changes, reload, unavailable storage, refused lock/write and unreadable-peer storage preservation are covered. Synthetic imported identities are reader notes, never acquired source custody.
- Existing browser checks:49 continuity,13 reading-list and3 source→Cite cases pass, zero page errors.
- All13 prescribed/focused local command groups exit0. Counts include repetitions:159 Python and457 Node passes; two expected standalone museum fixture skips were followed by45/45 fixture passes. See exact commands/results in `checks.json` and `check-summary.json`.

Browser: sandboxed headless Chromium154.0.8037.57 on Linux. Widths/themes are emulated. This does not qualify physical touch, Safari, spoken assistive technology, full WCAG, large-graph performance or mathematical certification. The separate current Atlas speech/control lane is excluded.

Executions occurred on the working-tree candidate before its source commit. The browser reports' `git_head` identifies the then-current base; `local_product_sha256` identifies tested product bytes. `SOURCE.json` binds these bytes and tools. No execution is retroactively claimed at a later commit. Author: OpenAI Codex /root. Test operator: separate same-family OpenAI Codex `/root/undo_local_verification`. Separate whole-branch engineering review is pending when this source report is frozen; neither author tests nor the test operator supply scientific or organizational independence.

## Reproduce

```sh
python3 -B tools/research_desk_undo_browser_check.py --output /tmp/ul-undo
python3 -B tools/research_desk_continuity_browser_check.py --browser /usr/bin/chromium --output /tmp/ul-continuity
python3 -B tools/research_desk_browser_check.py --browser /usr/bin/chromium --output /tmp/ul-reading-list
python3 -B tools/reading_list_citation_browser_check.py --browser /usr/bin/chromium --output /tmp/ul-citation
```

The new driver chooses installed `chromium` or accepts `--browser`; install Python Playwright and a sandbox-capable Chromium in a fresh environment. Local mode serves the checkout and reuses committed hash-checked catalog fixtures; unavailable unrelated remote sources stay explicit. `--origin https://d6g8k5htny-coder.github.io/main/site/` tests the actual public site without local source fixtures. Nothing is saved automatically or sent to a worker service. The complete prescribed verification commands remain in `docs/site/README.md`.

## Coordination and preservation

Canonical author claim: main#388/6412854991; board mirror619/6412855323; post-pickup overlap update388/6412955270. Complete56-PR/1119-path inventory is retained by exact head, with new653's complete path set subsequently checked. Cumulative501/399 workspace key overlaps and653's disjoint README verification paragraph are explicit; no foreign branch/role is taken over. 653 Atlas live controls, historical source/acquisition, formal and scientific roles remain separate.

Original source/Drive archives and historical failures are preserved. Integration, current native checks, actual publication and delivery custody require their later source-bound receipts. This report does not predeclare those outcomes or whole-vision completion.
