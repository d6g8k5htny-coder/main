import re, sys
def check(path):
    lines=open(path,encoding='utf-8').read().splitlines()
    i=0; bad=0; tables=0; rows=0
    while i < len(lines):
        if lines[i].startswith('|') and i+1 < len(lines) and re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', lines[i+1]):
            tables+=1
            n=len(re.findall(r'(?<!\\)\|', lines[i]))
            nsep=len(re.findall(r'\|', lines[i+1]))
            if n!=nsep: print(path,'sep mismatch',i+1); bad+=1
            j=i+2
            while j < len(lines) and lines[j].startswith('|'):
                m=len(re.findall(r'(?<!\\)\|', lines[j])); rows+=1
                if m!=n: print(path,'row',j+1,'pipes',m,'header',n); bad+=1
                if not lines[j].rstrip().endswith('|'): print(path,'row not closed',j+1); bad+=1
                j+=1
            i=j
        else: i+=1
    print(path,'tables',tables,'rows',rows,'bad',bad)
for p in sys.argv[1:]: check(p)
