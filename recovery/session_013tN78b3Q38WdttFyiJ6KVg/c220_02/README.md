# C220-02 screen-reader evidence (main#332)

The expanded tree of this directory is at commits `ea4637b9` and `32bc328c` on this branch. The delivery post [332/6400207942](https://github.com/d6g8k5htny-coder/main/issues/332#issuecomment-6400207942) links to the first of these.

At the branch head, the same files are packed into one archive, `c220_02.tar`. That keeps #441's changed-file count under the 500 that `tools/public_intake_check.py` accepts. The archive is deterministic: GNU format, sorted names, mtime 2026-10-25T00:00:00Z, owner 0.

- `CONTENTS_SHA256` lists the SHA-256 of every file inside the archive, by path relative to `c220_02/`.
- To unpack: `tar -xf c220_02.tar -C <empty dir>`, then `cd <empty dir>/c220_02 && sha256sum -c <path to CONTENTS_SHA256>`.
- `delivery_332_6400207942.md` is the posted text, read back from GitHub. It is also inside the archive.
