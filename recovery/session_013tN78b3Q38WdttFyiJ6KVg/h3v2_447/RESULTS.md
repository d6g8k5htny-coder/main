# main#447 rounding-soundness read: how to rerun it

These are the scripts, patches and raw outputs behind
[447/6246359073](https://github.com/d6g8k5htny-coder/main/pull/447#issuecomment-6246359073).
They were run on 18 October 2026.

- **Tested source:** #447 head `6651242aaa83656ec2b8ee04a9df4809e7777600`, `experiments/h3_band_v2/h3_band_v2.py`,
  sha256 `56734fc69487f3d9834b9ee95bbe74530968ce65eea95e382fc82a32b353c4b9`.
- **Environment:** CPython 3.11.15 and mpmath 1.4.1, installed with
  `pip install --require-hashes --only-binary=:all: -r experiments/h3_band_v2/requirements.txt`.
- **Status:** same-provider data with credit 0. It is not a review verdict.

Run every command from a checkout of that head. `SCRIPTS` is this directory's `scripts/`.

## 1. Directed-rounding replay

```
python3 -I -B "$SCRIPTS/rounding_run.py" experiments/h3_band_v2/h3_band_v2.py {n|f|c} {ceil|floor} OUT.txt
```

- `'n'` reproduces `results/{ceil,floor}.transcript.txt` byte for byte.
- `'f'` and `'c'` give the four `results/transcript_*_mp_*.txt` here. Each differs from the committed transcript
  only in its CB7 line.

## 2. Law containment on every cell

First apply `diffs/probe_hook.diff` to a scratch copy of the program, say `probe.py`. The diff only adds
`PROBE_HOOK`. Then run:

```
python3 -I -B "$SCRIPTS/law_probe2.py" probe.py experiments/h3_band_v2/h2 law_probe2.jsonl
```

It takes about 2 minutes and writes 52 rows; `results/law_probe2.jsonl` is that output. Every row has
`inside: true` and `resolved: true`.

## 3. Monte Carlo, NON-CERTIFYING

```
python3 -I -B "$SCRIPTS/mc_probe.py" experiments/h3_band_v2/h2 NUM DEN 1000000 90210
```

The runs at r = 1/32768, 1/1000, 3/400 and 1/20 are `results/mc_*.json`.

## R1 and R2

- **R1:** `diffs/r1_branch_counter.diff` instruments the unguarded `Maj.tail_from` branch and the `tails()` guard.
  Its stderr lines, filtered, are `results/r1_branch_counter_{ceil,floor}.txt`. They show 0 unguarded executions and
  4 guard runs per envelope.
- **R2:** `diffs/fix_r2.diff` is the proposed two-line fix. With it applied, `ceil` and `floor` both pass, and their
  transcripts are byte-identical to `results/{ceil,floor}.transcript.txt` at the tested head (sha256 `ddb2ade0…`,
  `b243f4be…`).

## Posted texts

`posts/` holds the pickup, the result and the #419 routing pointer, as posted.
