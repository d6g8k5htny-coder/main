import sys, json, csv
o = json.load(open(sys.argv[1]))[0]['_rowsAP']
for k in sorted(o, key=int):
    if o[k][0] in ('EV-LS-REQ030',) or o[k][5] in ('SOURCE_ADMISSION_REPAIR','DUPLICATE_REGISTRATION_RECEIPT','EXACT_REVIEW_REQUEST_FOR_LS_DER036','ORGANIZATIONALLY_DISTINCT_REVIEW_REQUEST'):
        print('sheet row', k, o[k][0], '|', o[k][5], '| G', o[k][6])
rows = list(csv.reader(open(sys.argv[2], encoding='utf-8', newline='')))
for i, r in enumerate(rows, 1):
    if r[0] == 'EV-LS-REQ030': print('csv line', i, r[0], '|', r[5])
