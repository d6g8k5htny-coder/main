"""Extract ORIGINAL (independent-derivation.md) from Math-#193 comment 6242350213 and print its identity.

Input: the JSON returned by GET repos/d6g8k5htny-coder/Math-/issues/comments/6242350213, saved to a file.
Rule: split the comment body on LF; keep everything strictly between the line "```markdown" and the next
line "```"; join with LF and append one LF. Standard library only.
Usage: python3 -I -B extract_original.py comment.json out.md
"""
import hashlib
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))["body"]
assert "\r" not in body
lines = body.split("\n")
start = lines.index("```markdown")
end = lines.index("```", start + 1)
raw = ("\n".join(lines[start + 1:end]) + "\n").encode("utf-8")
open(sys.argv[2], "wb").write(raw)
print("opening fence at line", start + 1, "closing fence at line", end + 1, "(1-indexed)")
print("bytes", len(raw), "lines", raw.count(b"\n"))
print("sha256", hashlib.sha256(raw).hexdigest())
print("git blob", hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest())
