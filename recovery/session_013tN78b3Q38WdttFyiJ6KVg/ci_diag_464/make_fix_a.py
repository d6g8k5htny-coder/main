"""Candidate D1 relocation: move FORMAL_EVIDENCE_TEMP from jobs.controls.env to the env of every step that reads it.
Usage: python3 -I make_fix_a.py IN.yml OUT.yml"""
import sys, yaml
src = open(sys.argv[1], encoding="utf-8").read().split("\n")
job_env = ["    env:", "      FORMAL_EVIDENCE_TEMP: ${{ runner.temp }}/formal-retained-volume-${{ github.run_id }}-${{ github.run_attempt }}/mount"]
i = src.index(job_env[0]); assert src[i + 1] == job_env[1]
value = job_env[1].split(": ", 1)[1]
del src[i:i + 2]
# step boundaries: lines starting with "      - " inside jobs.controls.steps
starts = [k for k, l in enumerate(src) if l.startswith("      - ")]
out, ends = [], starts[1:] + [len(src)]
blocks = [src[:starts[0]]] + [src[a:b] for a, b in zip(starts, ends)]
for n, b in enumerate(blocks):
    if n and "FORMAL_EVIDENCE_TEMP" in "\n".join(b):
        envline = [k for k, l in enumerate(b) if l == "        env:"]
        if envline:
            b.insert(envline[0] + 1, "          FORMAL_EVIDENCE_TEMP: " + value)
        else:
            b.insert(1, "        env:"); b.insert(2, "          FORMAL_EVIDENCE_TEMP: " + value)
    out += b
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
