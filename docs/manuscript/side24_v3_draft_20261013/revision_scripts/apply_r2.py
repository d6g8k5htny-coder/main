"""Apply one exact AMEND step to the V3 draft: replace strings in the build copy, rebuild, copy to the worktree, update SHA256SUMS.
Usage: python3 apply_r2.py STEP   (STEP in B, C)"""
import pathlib, subprocess, hashlib, sys, shutil
SP = pathlib.Path('/tmp/claude-0/-home-claude/b41b316f-232c-5cf7-a0ee-80f0de603a6f/scratchpad/v3')
TEX = SP / 'tex'; WT = pathlib.Path('/home/claude/wt-v3/docs/manuscript/side24_v3_draft_20261013')
STEPS = {
 'B': [('s5_discussion.tex', r'A simulation over $8\cdot10^7$ correlation lengths', r'A simulation over $8.4\cdot10^7$ correlation lengths')],
 'C': [('appA_records.tex', r'(\texttt{main}~PR~67)', r'(\texttt{main} issue~67)'),
       ('s0_status.tex', r'every link merged with OpenAI/Codex nonauthor reviews and a Grok Bot read (R$\cdot$1P+GB); $c_1/c$, $c_2$ certified at the stated scopes',
        r'the links in \rec{chain}, and $c_1$ and the closed form of $c_2$, each merged with an OpenAI/Codex nonauthor review and a Grok Bot read (R$\cdot$1P+GB), and certified at those stated scopes; the torus transfer of $c_2$ has an OpenAI \texttt{ACCEPT\_SCOPED} of slices A--C and an Anthropic acceptance of slices A and C, and no Grok Bot read'),
       ('s7_disclosure.tex', r"every link of Theorem~\ref{thm:three-term} has an OpenAI/Codex nonauthor review and an additional nonauthor read by Grok Bot agents \recs{chain}{consts}; the results of Remark~\ref{rem:beyond}, and the $c_3\ell^{2/3}$ term and the rate $\delta$ in Remark~\ref{rem:cand}, were read in bounded nonauthor slices \rec{twothirds}, while the rest of Remark~\ref{rem:cand} rests on records with one or two providers' nonauthor reviews \recs{unres}{parity}",
        r"the links of Theorem~\ref{thm:three-term} in \rec{chain}, and the certificates of $c_1$ and of the closed form of $c_2$, each have an OpenAI/Codex nonauthor review and an additional nonauthor read by Grok Bot agents; the finite-torus transfer of $c_2$ has an OpenAI \texttt{ACCEPT\_SCOPED} of slices A--C and an Anthropic nonauthor acceptance of slices A and C, and no Grok Bot read, \recs{chain}{consts}; the results of Remark~\ref{rem:beyond}, and the $c_3\ell^{2/3}$ term and the rate $\delta$ in Remark~\ref{rem:cand}, were read in bounded nonauthor slices \rec{twothirds}; Theorems U, K and P and Theorem Z in that remark rest on records with one or two providers' nonauthor reviews \recs{unres}{parity}; and $B_{3,24}\approx 10.2$ is exploration only \rec{eqh}")],
}
step = sys.argv[1]
changed = []
for f, old, new in STEPS[step]:
    for base in (TEX, WT):
        p = base / f; t = p.read_text()
        assert t.count(old) == 1, (base, f)
        p.write_text(t.replace(old, new))
    assert (TEX / f).read_bytes() == (WT / f).read_bytes(), f
    changed.append(f)
# rebuild in the build copy
for cmd in (['pdflatex', '-interaction=nonstopmode', 'main.tex'], ['bibtex', 'main'], ['pdflatex', '-interaction=nonstopmode', 'main.tex'], ['pdflatex', '-interaction=nonstopmode', 'main.tex']):
    r = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True)
log = (TEX / 'main.log').read_text(errors='replace')
assert '! ' not in log.replace('! LaTeX Font', ''), [l for l in log.splitlines() if l.startswith('!')][:5]
assert 'undefined' not in ' '.join(l for l in log.splitlines() if 'Warning' in l and ('Reference' in l or 'Citation' in l)), 'undefined refs'
over = [l for l in log.splitlines() if l.startswith('Overfull')]
pages = [l for l in log.splitlines() if 'Output written' in l]
print('overfull:', over); print(pages)
shutil.copy(TEX / 'main.pdf', WT / 'SIDE24_V3_DRAFT_20261013.pdf')
# SHA256SUMS: same file list, recomputed
names = [l.split()[1] for l in (WT / 'SHA256SUMS').read_text().splitlines()]
(WT / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (hashlib.sha256((WT / n).read_bytes()).hexdigest(), n) for n in names))
print('changed', changed, 'pdf', (WT / 'SIDE24_V3_DRAFT_20261013.pdf').stat().st_size, hashlib.sha256((WT / 'SIDE24_V3_DRAFT_20261013.pdf').read_bytes()).hexdigest())
