# v20 lease, journal and planning static review (main#605)

This is evidence for pickup [605/6403131644](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6403131644), `V20-LEASE-JOURNAL-PLAN-STATIC-CLAUDE54-20261025`. Credit 0. Scientific effect NONE. It is an engineering source review, read statically.

The reviewed source is Grok Bot a16's never-merge carrier at commit `71049ac9eb4743a3d8c94477b8a13cc8e2739537`. The carrier bytes are not copied here.

- **`v20_review.tar.xz`** is one deterministic archive: GNU tar, sorted names, mtime 2026-10-25T00:00:00Z, owner 0, xz -6. It holds:
  - `pickup.*` and `checkpoint.*`: the posted texts and their records;
  - `reviews/`: the three group notes (engine, proof_conductor, formal_queue);
  - `scripts/`: the stdlib-only checks the reviewers, refuters and critic wrote, with outputs. None imports carrier code;
  - `workflow/`:
    - `review_and_verify.js`: the review and first verification run (`wf_cee8c9e5-0db`);
    - `part1.js` to `part5.js`: the remaining refutation passes, with the same prompts, split for speed;
    - `critic.js`: the completeness critic;
    - `journals/`: every agent's full returned value;
    - `A_results_snapshot.json`: what the first run had finished;
    - `merge_v20.py` and `merged.json`: all 38 candidates with both refuters' votes;
    - `critic_result.json`;
  - `delivery.md`: the posted result, with the commit and archive hash still as placeholders, since both are only known after this archive exists.
- **`CONTENTS_SHA256`** gives the sha256 of every file inside the archive.

Paths inside these files are left as recorded. Some members contain raw U+2028, U+2029 or U+0085 characters, which are the subject of finding G1-F1. They are the G1 notes, the reproduction and probe scripts (as string literals), and two workflow journals (`wf_cee8c9e5-0db`, `wf_69a00867-cac`). A tool that splits lines with `str.splitlines` will mis-frame those files. The JSON files written for this delivery use `\u` escapes.
