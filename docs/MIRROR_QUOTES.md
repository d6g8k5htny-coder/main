# Quoting the Drive: what "verbatim" has to mean here

Each directory under [`drive/mirrors/`](../drive/mirrors) and
[`drive/deltas/`](../drive/deltas) holds byte-exact copies of Drive objects and a `README.md` that
says what they are. Those READMEs carry sections headed **"the status banners, verbatim"**. The
quotations in them are the point of the mirror: they are how a reader learns, without opening the
Drive, what status each mirrored object claims *for itself* — `STATUS: PROPOSED`, `AUTHORITY: none`,
`DISPOSITION: FAIL`, `zero organizational-independence credit`.

A quotation under that heading is a factual claim about bytes this repository holds. If it is wrong,
the repository is misrepresenting the Drive in the one place a reader has no way to check.

## It was wrong, repeatedly

Every departure below was found in this repository, in text that had already been committed:

| shape | what the README said | what the file says |
|---|---|---|
| invented separator | `STATUS: PROPOSED · AUTHORITY: none` | the two fields are on separate lines |
| a second invented separator | `STATUS: PROPOSED / AUTHORITY: none` | likewise — and inconsistent with the first |
| dropped clause | `…weakened by any item**.` | `…weakened by any item**;` and the sentence continues |
| added terminal period | `AUTHORITY: none.` | `AUTHORITY: none` |
| comma for period | `Old control documents are history,` | `**Old control documents are history.**` |
| straight for curly | `not "ask Dylan."` | `not “ask Dylan.”` |
| transliterated LaTeX | `` `C_Q0 < ∞` `` | `C_{Q0}<\infty` |
| added emphasis | `**Presence here is not a mathematical falsity verdict.**` | the source does not bold it |
| dropped field | `{ "required": 4, "found": 4, "result": "PASS" }` | the object also carries a `note` |
| line-wrapped at a slash | `entry/budget/ provider locks` | `entry/budget/provider locks` |
| false generalisation | "each GP-MIG receipt ends …" | true of two of the four |

None of them changed a status word, and none inflated a research claim. That is exactly why they
survived review by eye: each one reads better than the source, and each one is a small lie about a
document the reader cannot see.

## The rule the checker applies

`tools/mirror_quotes_check.py` extracts every **fragment** — a run of 12 or more characters inside a
pair of quotation marks, straight or curly, split on ellipsis so an elided quote is checked
piecewise — from every README under the scanned roots, and requires each to appear in the stored
bytes. Blockquote `>` prefixes are stripped. A contiguous blockquote carrying no quotation marks at
all is itself one fragment, because that is the other way this repository presents a banner. Table
rows, fenced code blocks and inline code spans are skipped: a `"` inside a shell command opens no
quotation. Backticks are still stripped from both sides when matching, so a quotation that merely
contains a backticked token still lines up with its source.

Matching normalises Unicode to NFC and collapses every run of whitespace to one space. So markdown
soft-wrapping is forgiven, because it is invisible to a reader — and a source *line break* rendered
as ` · ` or ` / ` is **not** forgiven, because the middot is a character the Drive object does not
carry.

The corpus a README may quote from is the bytes this repository actually holds:

* every non-README file under the scanned roots, meaning the stored payloads and the manifests
  that describe them. A fragment matching **only** a manifest is the repository quoting its own
  metadata rather than the Drive. That is a true statement and it passes, but it is not a
  transcription, so the summary line counts it as `manifest_only`;
* every cell of `registers/json/*.json`;
* every `title`, `name`, `drive_path` and `path` in `drive/inventory.jsonl`;
* three repository documents the lane READMEs quote **by name**: `CLAUDE.md` for the rules they
  operate under, `claims/graph.json` where a mirrored banner and the claim graph disagree about a
  lemma's status, and `governance/GIT_ADAPTATION.md` where a lane records that a Drive construct
  has no repository equivalent. Each is named in the sentence that quotes it, so the quotation is a
  checkable statement about a file here rather than a transcription of the Drive. Nothing else
  under `governance/` is in the corpus.

**There is no allowlist, deliberately.** If a README quotes an object that was read but not stored,
the honest repair is to say so in prose without quotation marks, or to store the reading copy. A
checker with an exemption list for the very thing it checks stops working the first time somebody is
in a hurry. There is exactly one exemption, described next, and it is a rule about a documented
convention rather than a list of passages.

```bash
python3 tools/mirror_quotes_check.py           # recheck; nonzero and one line per departure
python3 tools/mirror_quotes_check.py --list    # print every fragment it checked, then the summary
python3 tools/mirror_quotes_check.py --min 12  # a stricter floor, for a one-off look
```

## The one exemption, and why it is not an allowlist

This repository's correction convention is that a fixed passage says what it used to say:

> Until 2026-09-20 this sentence transliterated the LaTeX away, and dropped the law the theorem is
> stated under, reading "there exists a finite constant `C_Q0 < ∞` such that …".

Those quotation marks are around text that is deliberately in no stored byte. That is the point of
them. So a fragment sitting after an `Until <date>` marker, **in the same paragraph**, is exempt and
counted: the summary line reports `prior_wording=N` beside `problems=N`, and `--list` tags each one,
so the exemption is visible rather than silent. It is a small minority of the corpus, and rather than
print a figure here that would rot, a control in `tests/test_mirror_quotes.py` fails if the exempted
fragments ever exceed a quarter of those checked. Run the checker for the current pair.

It is a rule, not a list. It names no file and no passage; it recognises a documented convention, and
controls pin its reach — a bare "until" with no date exempts nothing, a marker does not reach a quote
earlier in its own paragraph, and it does not reach into the next paragraph at all.

**A disclosure quote's own fidelity is checked, but by a different tool and at a different moment.**
`tools/disclosure_check.py` closes the gap this exemption opens, in both directions:

* every exempted fragment must be verbatim in the previous committed version of the same file, so a
  note cannot invent a history;
* every quotation the committed version carried whose wording is gone from the working tree must
  appear in a file that records corrections, so a correction cannot erase one in silence.

The second is a report rather than a refusal when the file does record corrections, and a refusal
when it records none. That asymmetry is deliberate. Some corrections *remove* quotation marks on
purpose, because the object was read and never stored and quoting it at all was the defect; such a
correction cannot disclose itself by quoting the old wording, and demanding that it do so would push
the author straight back into the dishonest form.

It runs in the pre-commit block in [`CLAUDE.md`](../CLAUDE.md), not in CI, because a disclosure
written at commit N describes the text at commit N−1: once the correction is committed, the named
revision carries the note itself, the quoted former wording is trivially found inside it, and the
check can no longer fail. A control commits a fabricated disclosure and shows the tool stop seeing
it, so nobody mistakes a green post-commit run for a standing guarantee.

## Where the floor sits, and why

The floor is 12 characters, and it is measured rather than chosen. It is the lowest value at which
this repository is clean, and getting there was itself the work: at 24 the tree passed while three
quoted phrases below the floor were wrong, including a register-backed sentence quoted with its
initial capital silently lowered, repeated in four READMEs. Dropping to 12 raised the fragments
checked from 726 to 823.

Below 12 a quoted run is short enough to turn up in the corpus by coincidence, which weakens the
check rather than strengthening it. A control asserts the tree is clean at 12, and a second one
records what the floor still forgives by corrupting a short phrase and showing `--min 4` catch what
the default does not. If a future change makes 12 fail, the fix is the quotation, not the floor; if
8 also comes back clean, the floor can come down, deliberately.

## Why it stops at `drive/`

The scanned roots are `drive/mirrors/` and `drive/deltas/` and nothing else, and that is a real limit,
not an oversight. Inside them, a quotation mark under a heading that says *verbatim* is a transcription
claim about a Drive object, so the mechanical rule is exactly right. Outside them, this repository also
uses quotation marks idiomatically — `"packaging is not premise discharge"`, `"do not compose 2D and
3D"`, `"the exact differentiation formula"` — where the words are its own and no transcription is
claimed.

Measured against commit `173831f`: applying the same rule to the 60 Markdown files outside `drive/`,
with every other file in the repository as corpus, left 101 unmatched fragments, the largest groups in
`docs/FINDINGS_2026-09-18.md` (22), `docs/RESEARCH_MAP.md` (17) and
`reviews/records/REV-OPS-R17-001.md` (11). Most were that idiom. Some were not: a few quote register
cells and should match. Separating the two needs a convention — checking only sections headed
*verbatim*, or a per-file opt-in — and that is work this checker does not do. Nothing re-measures that
figure, so treat it as the size of the gap when the gap was last looked at, not as a current count.
Saying so is the point: those fragments are unchecked.

## What a green run does not establish

Matching proves a string is present *somewhere* in the stored bytes. It does not prove:

* that the quotation is attributed to the right file — the checker does not read the sentence around
  it, so quoting file A's banner and crediting file B passes;
* that a generalisation over several files ("each notice ends with…") holds of all of them;
* that the surrounding description is accurate, that a count is current, or that a coverage table
  still matches the tree;
* that added emphasis is faithful — markdown `**` and backticks are stripped from both sides, so
  bolding a sentence the source leaves plain passes;
* that whitespace is faithful — a reflowed JSON block passes.

Those need a reader, and the lane verification records are where this repository keeps them. Nothing
here reads, grades or moves any claim, premise or obligation. A green run says the quotation marks in
these READMEs are honest about the bytes. It says nothing whatever about the mathematics the bytes
discuss, and no status, promotion, review verdict or licensing predicate follows from it.
