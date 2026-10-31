#!/usr/bin/env python3
"""Substitute @@PLACEHOLDER@@ tokens in files from one or more JSON maps; refuse to leave any.

Usage: apply_fills.py FILLS.json [MORE.json...] -- FILE...
"""
import json
import pathlib
import re
import sys

args = sys.argv[1:]
sep = args.index('--')
fills = {}
for p in args[:sep]:
    fills.update(json.loads(pathlib.Path(p).read_text()))
status = 0
for f in args[sep + 1:]:
    path = pathlib.Path(f)
    text = path.read_text()
    for key, value in fills.items():
        text = text.replace(key, value)
    left = sorted(set(re.findall(r'@@[A-Z0-9_]+@@', text)))
    if left:
        print(f, 'unfilled:', ', '.join(left))
        status = 1
    path.write_text(text)
    print(f, 'filled')
sys.exit(status)
