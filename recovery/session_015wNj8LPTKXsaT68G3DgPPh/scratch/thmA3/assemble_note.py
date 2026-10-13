D='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/thmA3'
P='/home/user/wt_d3/frontiers/c8_theorem_a_constants_d3_20261001'
head=open(D+'/NOTE_draft.md').read()
stmt=open(D+'/NOTE_stmt.md').read()
body=open(D+'/NOTE_body.md').read()
tail=open(D+'/NOTE_tail.md').read()
old="""  The second slope is the chord of `min(bar-g, g_0 exp(c lambda))`, valid because `g(lambda) <= g(0) exp((b + epsilon) lambda / sigma^2)`
  for `lambda >= 0`. It matters at larger `b`, where `g` rises on the failure window."""
new=old+""" When `g_0 >= bar-g`, only the first
  bound is used."""
assert old in body
body=body.replace(old,new)
mc=open(D+'/NOTE_mc.md').read() if __import__('os').path.exists(D+'/NOTE_mc.md') else '@@MC@@'
tail=tail.replace('@@MC@@', mc.rstrip('\n'))
note=head.replace('@@STATEMENT@@\n', stmt+'\n'+body+'\n'+tail)
open(P+'/NOTE.md','w').write(note)
print(len(note.encode()))
