import hashlib, json, os
root='/home/user/wt_capg/frontiers/c8_cap_failure_coefficient_planar_20260930'
files=[]
for name in sorted(os.listdir(root)):
    if name=='SOURCE_FILES.json' or name.startswith('.') or name=='__pycache__': continue
    data=open(os.path.join(root,name),'rb').read()
    files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob':hashlib.sha1(b'blob %d\0'%len(data)+data).hexdigest()})
doc={'schema':1,'scientific_effect':'NONE','certified':True,'review_status':'AUTHOR_SIDE_LIMIT_THEOREM_WITH_CERTIFIED_COEFFICIENT','object':'CL-C8-CAP-FAILURE-COEFFICIENT-PLANAR-20260930-v1','files':files,'nonauthor_reads':[]}
json.dump(doc, open(os.path.join(root,'SOURCE_FILES.json'),'w'), indent=1); open(os.path.join(root,'SOURCE_FILES.json'),'a').write('\n')
print(json.dumps(files, indent=0))
