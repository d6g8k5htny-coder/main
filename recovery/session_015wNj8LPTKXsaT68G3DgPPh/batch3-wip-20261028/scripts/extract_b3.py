#!/usr/bin/env python3
"""List the session transcript's tool calls that touched the batch-3 work tree, in order.

Usage: extract_b3.py TRANSCRIPT OUTDIR

Writes OUTDIR/calls.json: every tool_use whose input mentions c8wt3 (Edit, Write,
Bash), each with its tool_result text, in transcript order. Nothing is executed.
"""
import json
import pathlib
import sys


def text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if isinstance(item, dict) and item.get('type') == 'text':
            parts.append(item.get('text', ''))
    return ''.join(parts)


def main():
    transcript, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    uses, results, order = {}, {}, []
    for line in transcript.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        content = (rec.get('message') or {}).get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get('type') == 'tool_use':
                uses[item['id']] = {'name': item.get('name'), 'input': item.get('input')}
                order.append(item['id'])
            elif item.get('type') == 'tool_result':
                results[item.get('tool_use_id')] = text_of(item.get('content'))
    calls = []
    for n, uid in enumerate(order):
        use = uses[uid]
        blob = json.dumps(use['input'])
        if 'c8wt3' not in blob and 'merge_contracts_b3' not in blob and 'real_replay_b3' not in blob:
            continue
        calls.append({'n': n, 'id': uid, 'name': use['name'], 'input': use['input'], 'result': results.get(uid)})
    (outdir / 'calls.json').write_text(json.dumps(calls, indent=1))
    for c in calls:
        inp = c['input']
        summary = inp.get('file_path') or inp.get('description') or ''
        print(c['n'], c['name'], summary[:110].replace('\n', ' '))


if __name__ == '__main__':
    main()
