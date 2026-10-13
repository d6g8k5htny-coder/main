import json,re,sys
src=open(sys.argv[1],encoding='utf-8').read()
reg=json.load(open(sys.argv[2]))
ents={s['declaration']:s for c in reg['claims'] for s in c.get('statements',[]) if 'Side24Images' in s.get('declaration','')}
for name in ['imageBound','entrywiseCovarianceBound','covarianceOrdering']:
    m=re.search(r'/--((?:(?!-/).)*?)-/\s*theorem '+name+r'\b',src,re.S)
    doc=m.group(1)
    dne=doc[doc.index('Does not establish:')+len('Does not establish:'):]
    norm=' '.join(dne.split())
    e=ents['UniversalLaw.Proofs.Side24Images.'+name]
    eq=norm==' '.join(e['does_not_claim'].split())
    print(name,'equal(ws-norm):',eq, 'raw-equal:', norm==e['does_not_claim'])
    if not eq:
        a=norm;b=' '.join(e['does_not_claim'].split())
        i=next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]),min(len(a),len(b)))
        print('  diverge at',i,'\n  doc:',a[i-80:i+120],'\n  reg:',b[i-80:i+120])
    # anchors
    print('  anchor:',e['anchor'],'| fp:',e['formal_progress'])
