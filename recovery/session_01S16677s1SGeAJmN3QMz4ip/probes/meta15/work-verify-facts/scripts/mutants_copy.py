"""Mutation probe: which faulty marker predicates survive the proposal's matrix
read weakly (ok False / category present somewhere) vs strictly (exact per-file category set)."""
import json
K='scientific_status_authority'
T,M='TRUE','MALFORMED'

def walk(v):
    # iterative pre-order over dicts/lists, yields each dict
    stack=[v]
    while stack:
        x=stack.pop()
        if isinstance(x,dict):
            yield x; stack.extend(reversed(list(x.values())))
        elif isinstance(x,list):
            stack.extend(reversed(x))

def reference(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if v is True: out.add(T)
            elif v is False: pass
            else: out.add(M)
    return out

def predecessor(data):
    return {T} if any(d.get(K) is True for d in walk(data)) else set()

def m_eq(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if v == False: pass
            elif v is True: out.add(T)
            else: out.add(M)
    return out

def m_truthy(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if not v: pass
            elif v is True: out.add(T)
            else: out.add(M)
    return out

def m_truthy_true(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if v is False: pass
            elif v: out.add(T)
            else: out.add(M)
    return out

def m_membership(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if v in (False,): pass
            elif v in (True,): out.add(T)
            else: out.add(M)
    return out

def m_root_only_type(data):
    out=predecessor(data)
    if isinstance(data,dict) and K in data and type(data[K]) is not bool: out.add(M)
    return out

def m_no_descend_marker(data):
    out=set()
    def rec(x):
        if isinstance(x,dict):
            if K in x:
                v=x[K]
                if v is True: out.add(T)
                elif v is not False: out.add(M)
            for k,v in x.items():
                if k!=K: rec(v)
        elif isinstance(x,list):
            for v in x: rec(v)
    rec(data); return out

def m_first_finding(data):
    for d in walk(data):
        if K in d:
            v=d[K]
            if v is True: return {T}
            if v is not False: return {M}
    return set()

def m_string_false(data):
    out=set()
    for d in walk(data):
        if K in d:
            v=d[K]
            if v is False or v=='false': pass
            elif v is True: out.add(T)
            else: out.add(M)
    return out

MUTANTS={'predecessor(adf2945b)':predecessor,'eq_False':m_eq,'truthiness_accept':m_truthy,
 'truthiness_true_category':m_truthy_true,'tuple/set_membership':m_membership,
 'root_only_type_check':m_root_only_type,'no_descent_into_marker_value':m_no_descend_marker,
 'first_finding_short_circuit':m_first_finding,'string_false_accepted':m_string_false}

def nest(v): return {'nested':[{K:v}]}
# Proposal matrix, literally: values; placement unspecified -> try both root and nested
NEG=[0,1,0.0,1.0,'','x',None,[],[1],{},{'a':1}]
PROPOSAL_CASES=[]
for v in NEG:
    PROPOSAL_CASES.append(('root',{K:v}))
PROP_TRUE=[('true-root',{K:True}),('true-nested-dict',{'a':{K:True}}),('true-nested-list',{'a':[{K:True}]})]
PROP_CONTAIN=[('malformed-containing-true',{K:{K:True}})]
POS=[('absent',{'a':1}),('false-root',{K:False}),('false-nested',nest(False)),('ordinary',{'a':[1,{'b':None}]})]

AMEND_EXTRA=[('nested',nest(v)) for v in NEG+['false','true','0',-0.0,float('nan')]]
AMEND_EXTRA+=[('root',{K:v}) for v in ['false','true','0',-0.0,float('nan')]]
AMEND_SIB=[('sibling malformed-then-true',{'a':{K:1},'b':{K:True}}),('sibling true-then-malformed',{'a':{K:True},'b':{K:1}}),
           ('malformed-list-containing-true',{K:[{K:True}]})]

def run(matrix_name, cases_neg, cases_true, cases_contain, cases_pos, strict):
    print('==',matrix_name,'strict' if strict else 'weak')
    for name,f in MUTANTS.items():
        killed=[]
        for label,doc in cases_pos:
            if f(doc)!=reference(doc): killed.append('pos:'+label)
        for label,doc in cases_neg:
            got,exp=f(doc),reference(doc)
            if strict:
                if got!=exp: killed.append(f'neg:{label}:{json.dumps(doc)[:40]}')
            else:
                if not got: killed.append(f'neg:{label}:{json.dumps(doc)[:40]}')   # weak: only 'refused'
        for label,doc in cases_true:
            got=f(doc)
            if (got!={T}) if strict else (T not in got): killed.append('true:'+label)
        for label,doc in cases_contain:
            got,exp=f(doc),reference(doc)
            if strict:
                if got!=exp: killed.append('contain:'+label)
            else:
                if not got: killed.append('contain:'+label)   # weak: 'both may be reported' -> only refusal required
        print(f'  {name:32s}', 'KILLED' if killed else 'SURVIVES', ('('+str(len(killed))+': '+'; '.join(killed[:3])+')') if killed else '')

run('proposal matrix (negatives at root dict only)',PROPOSAL_CASES,PROP_TRUE,PROP_CONTAIN,POS,False)
run('proposal matrix (negatives at root dict only)',PROPOSAL_CASES,PROP_TRUE,PROP_CONTAIN,POS,True)
run('amended matrix',PROPOSAL_CASES+AMEND_EXTRA,PROP_TRUE,PROP_CONTAIN+AMEND_SIB,POS,True)
