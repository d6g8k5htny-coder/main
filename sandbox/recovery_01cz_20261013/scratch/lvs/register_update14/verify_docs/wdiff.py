import sys, subprocess, difflib, re
repo = '<SCRATCH>/repos/math_r5'
path = sys.argv[1]
old = subprocess.run(['git','-C',repo,'show','9e305ea5:'+path],capture_output=True,check=True).stdout.decode()
new = open(repo+'/'+path,encoding='utf-8').read()
ol, nl = old.split('\n'), new.split('\n')
sm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
def toks(s): return re.findall(r'\S+|\s+', s)
for tag,i1,i2,j1,j2 in sm.get_opcodes():
    if tag=='equal': continue
    print(f'=== {tag} old[{i1+1}:{i2}] new[{j1+1}:{j2}]')
    a='\n'.join(ol[i1:i2]); b='\n'.join(nl[j1:j2])
    ta, tb = toks(a), toks(b)
    s2 = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    out=[]
    for t,a1,a2,b1,b2 in s2.get_opcodes():
        if t=='equal':
            seg=''.join(ta[a1:a2])
            if len(seg)>160: seg=seg[:70]+' ..... '+seg[-70:]
            out.append(seg)
        else:
            out.append('[-'+''.join(ta[a1:a2])+'-]' if a2>a1 else '')
            out.append('{+'+''.join(tb[b1:b2])+'+}' if b2>b1 else '')
    print(''.join(out))
