# OWNER DECISION — public-surface disposition of `google-drive`, `trial` and `sandbox`, 2026-10-06

Effective owner delegation, recorded 2026-10-06 by Anthropic Claude (Claude Code,
session `017Mi3hx…`) on Dylan Roy's instruction, given directly in that session,
for [main#227](https://github.com/d6g8k5htny-coder/main/issues/227). Scientific
effect NONE. Nothing here changes a status, flag, premise, prize or register.

## Exact owner instruction

> For 227 do what's best for the project overall. If it advances the mathematical
> work without any negative effects in the future or will mess up our project you
> have my blessing to resolve this however is needed

## Facts that decide it (live reads, 2026-10-06 02:00–02:10Z)

1. All three repositories are public and not archived. No agent lane can change
   repository settings: this session's GitHub proxy refuses repository-settings
   writes, and the OpenAI lanes reported the same boundary (main#227 6003255968,
   6005800925). The settings clicks therefore stay with the owner.
2. `google-drive` holds 12 deliberately selected public replicas, each with a
   `SOURCE.json` identity and a verification workflow. It is not a Drive backup,
   and no credential exposure has been established (Codex, 1 October; Astra,
   5 October). Its About text, "Replica of Google Drive", is the misleading part.
   Every proof it labels "Drive-primary" now has a second public copy with
   identical bytes: the marked-cylinder cap and the matrix/lifetime proof under
   Math- `imports/lifetime_parent_20260925/` (blobs `0633aca3`, `dfed3b8d`), P15-B
   under Math- `imports/hardening_ebedb780/P15-B/` (blob `deab15eb`), and the
   demanded-palette optimum in meta-framework `replicas/consecutive-palette-v1/`
   (blob `1aa9db5b`). The public surfaces link to the three repositories in
   eleven places (this file's companion edit lists them in the pull request).
3. `trial` is the historical integration lab. Eleven workflows are still enabled
   there, and `watch-main-alignment.yml` was still running on its hourly schedule
   on 5 October (last run 22:59Z) despite the
   [27 September stop](OWNER_STOP_20260927.md). The public reproduction notes
   link its `federation/replay.py`; Math- `AGENTS.md` links its access guide.
4. `sandbox` is the bounded exploratory workspace. Its default branch last
   changed on 28 September; nothing on the public surfaces depends on it (the one
   test that names it uses it as a negative control).

## Decision

1. **Keep all three repositories public.** Making them private would break the
   eleven public links and the Drive-identity provenance, and would remove no
   secret. Privacy concerns are handled by OP-PRIVACY, not by hiding research
   replicas.
2. **Archive `trial` and `sandbox`** (read-only, still public). Archiving ends
   every workflow in `trial`, including the hourly watch the 27 September stop
   should already have ended, and freezes `sandbox` without hiding it.
3. **`google-drive`: correct the About text, then archive.** The About text
   becomes: "Selective, source-bound public replicas of 12 research outputs; not
   a Google Drive backup. See README." The README gains a dated qualification of
   its former Drive-primary and routing sentences (pull request in that
   repository) and the repository is archived once that lands. No further
   replica is planned: since 27 September, Drive and Dropbox are not sources for
   new GitHub material (OP-PRIVACY).
4. The repository map in `docs/RESEARCH_INDEX.md` describes the three as
   archived, read-only auxiliary repositories and keeps their links (this pull
   request).
5. Everything here is reversible with one click (unarchive); no history is
   rewritten and no byte is removed.

## The owner's remaining actions (about two minutes)

1. `trial` → Settings → General → Danger Zone → **Archive this repository**.
2. `sandbox` → Settings → General → Danger Zone → **Archive this repository**.
3. `google-drive` → About (gear icon) → Description → paste the text in
   Decision 3 → Save.
4. After the `google-drive` README pull request has merged: `google-drive` →
   Settings → General → Danger Zone → **Archive this repository**.

main#227 stays open until the three `archived` flags are observed; the recording
lane then closes it with the observed timestamps.

## Unchanged

Scientific acceptance conditions, `formal/` and register custody,
[OP-PRIVACY](OP-PRIVACY-20260927.md) and the Drive/Dropbox rules keep their
meaning. The [5 October decision](OWNER_DECISION_20261005_CURSOR.md) on Cursor
agents stands.
