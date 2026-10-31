# Validation commands for the 20-target #272 record

Scientific effect: NONE. Lean was not executed. The formal tree used for `gate.py` is Math-#272 head `270167900b71539d0e2135131a3a96f1f04289f6`, checked out locally from `d6g8k5htny-coder/Math-`. Publication of these files is on `d6g8k5htny-coder/main` branch `cursor/lean-probability-alignment-272-20-676d`. Push of that branch to Math- returned `Permission to d6g8k5htny-coder/Math-.git denied to cursor[bot]` (HTTP 403). #272, #273, and #274 refs were not updated.

## 20-target record

Command, with `formal/gate.py` taken from the #272 tree and the JSON taken from this publication:

```
python3 -B -S formal/gate.py --alignment /workspace/reviews/lean_probability_alignment_20261005/alignment_272_20.json
```

Stdout:

```
SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): 845d372a459d32f6b4ffb0a4dceb69363ffe27c4ca6da8838238ce8979d9244d
```

Exit code: 0. On this `gate.py`, exit 0 after `--alignment` means `check_alignment` accepted the record.

Evidence file at commit `644d06ebe06c748e7f62988942fddaeabddd1147`:

```
5c1b30066fde9cf920ee5ce4534ec3a43770e8209897edd1b20bd14c7005d7ee  reviews/lean_probability_alignment_20261005/REVIEW_272_20.md
```

## Reused 29-target JSON, rejected

The 29-target `alignment.json` from Math- `8cf6b74472d77a62f4775a5e3568e79d4407adff` was passed unchanged to the same #272-tree gate:

```
python3 -B -S formal/gate.py --alignment /tmp/alignment_29.json
```

Stderr: `FORMAL_GATE_FAIL: stale alignment manifest`. Exit code: 1.

## 29-target record on its own tree

At Math- `8cf6b74472d77a62f4775a5e3568e79d4407adff`:

```
python3 -B -S formal/gate.py --alignment reviews/lean_probability_alignment_20261005/alignment.json
```

Stdout:

```
SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): d843ba7b96457a95e6485a4bc30678a8a252acf405d369b26dab81832c596c39
```

Exit code: 0.
