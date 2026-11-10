#!/usr/bin/env python3
"""T11c mislabel screen, read-only. For triage-ledger rows labelled WORKING or ENGINEERING, count strong mathematical markers in
the blob (Lean theorem/lemma declarations, Theorem/Lemma/Proposition/Corollary headings, proof markers, LaTeX theorem
environments). Rows at or above THRESHOLD are written for an adversarial re-check.
Usage: python3 -I -B mislabel_screen.py MAIN LEDGER OUT [THRESHOLD]"""
import json, re, subprocess, sys
main, led_p, out = sys.argv[1:4]
th = int(sys.argv[4]) if len(sys.argv) > 4 else 3
led = json.load(open(led_p))
rows = [r for r in led if r['label'] in ('WORKING', 'ENGINEERING')]
PAT = {
    'lean_decl': re.compile(r'^\s*(theorem|lemma)\s+\S+', re.M),
    'heading': re.compile(r'^\s*(#+\s*|\*\*)?(Theorem|Lemma|Proposition|Corollary)\b[^\n]{0,80}', re.M),
    'proof': re.compile(r'(^\s*(\*\*|_)?Proof[.:]|∎|\\qed|\bQ\.E\.D\b)', re.M),
    'latex_env': re.compile(r'\\begin\{(theorem|lemma|proposition|proof)\}'),
}
p = subprocess.Popen(['git', '-C', main, 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
flag = []
for r in rows:
    p.stdin.write(r['git_blob'].encode() + b'\n'); p.stdin.flush()
    hdr = p.stdout.readline().split()
    n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
    t = data[:400000].decode('utf-8', 'replace')
    c = {k: len(v.findall(t)) for k, v in PAT.items()}
    score = 3 * c['lean_decl'] + c['heading'] + c['proof'] + c['latex_env']
    if score >= th:
        flag.append({**{k: r[k] for k in ('git_blob', 'path', 'ref', 'commit', 'label', 'evidence', 'note', 'triage_batch')}, 'markers': c, 'score': score})
p.stdin.close(); p.wait()
json.dump(sorted(flag, key=lambda x: -x['score']), open(out, 'x'), indent=1, ensure_ascii=False)
print(len(rows), 'WORKING/ENGINEERING rows;', len(flag), 'flagged at score >=', th)
