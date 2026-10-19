"""Reproduce the ORIGINAL bytes from the REST JSON of Math-#193 comment 6243478311.

Usage: python3 -I extract_original.py comment.json out.md
Rule: the text strictly after the line '--- BEGIN EXACT COMPLETE ORIGINAL DERIVATION ---'
and before '--- END EXACT COMPLETE ORIGINAL DERIVATION ---', minus the single newline the
relay adds before the END marker. Expected: 24,027 bytes, 629 lines,
SHA-256 f38c085ab5b15368e6989a1fd6298490f3b671f5a6c3a01bcdc81b516022bc33.
"""
import hashlib
import json
import sys

body = json.load(open(sys.argv[1], encoding='utf-8'))['body']
begin = '--- BEGIN EXACT COMPLETE ORIGINAL DERIVATION ---\n'
end = '--- END EXACT COMPLETE ORIGINAL DERIVATION ---'
seg = body[body.index(begin) + len(begin):body.index(end)]
assert seg.endswith('\n\n'), 'unexpected relay framing'
data = seg[:-1].encode('utf-8')
open(sys.argv[2], 'wb').write(data)
print(len(data), data.count(b'\n'), hashlib.sha256(data).hexdigest())
