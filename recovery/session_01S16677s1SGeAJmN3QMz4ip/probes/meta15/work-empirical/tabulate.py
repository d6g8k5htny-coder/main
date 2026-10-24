import json
from pathlib import Path
W = Path(__file__).resolve().parent
R = json.load(open(W / 'results.json'))
def cell(r):
    cats = ','.join(r['api_categories']) or 'none'
    return f"{cats} {r['cli_normal']['rc']}/{r['cli_O']['rc']}"
order_p = ['root', 'nested2', 'list_root', 'list_in_dict', 'deep_mixed']
vals = []
for r in R:
    if r['kind'] == 'matrix' and r['value'] not in vals: vals.append(r['value'])
idx = {(r['value'], r['placement']): r for r in R if r['kind'] == 'matrix'}
lines = ['# Matrix: value x placement -> API violation categories | CLI exit normal/-O', '',
         'AUTH = "generated scientific-status authority: <path>"; PARSE = "generated JSON parse error: ..."; none = ok:true, violations [].', '',
         'Placements: ' + '; '.join(f"{p} = {next(r['placement_desc'] for r in R if r.get('placement') == p)}" for p in order_p)
         + '. D = {"k":"v","scientific_status_authority":V} (absent: D = {"k":"v"}).', '',
         '| value | ' + ' | '.join(order_p) + ' |', '|---|' + '---|' * len(order_p)]
for v in vals:
    lines.append(f'| `{v}` | ' + ' | '.join(cell(idx[(v, p)]) for p in order_p) + ' |')
lines += ['', '# Extra cases', '', '| case | input | API categories | CLI normal/-O | violations (API) |', '|---|---|---|---|---|']
for r in R:
    if r['kind'] != 'extra': continue
    inp = '; '.join(f"{k}: {v}" for k, v in r['files'].items()) or '(no generated dir)'
    inp = inp.replace('|', '\\|')
    viol = r['api'].get('violations') if 'exception' not in r['api'] else r['api']['exception']
    lines.append(f"| {r['case']} | `{inp[:110]}` | {','.join(r['api_categories']) or 'none'} | {r['cli_normal']['rc']}/{r['cli_O']['rc']} | {str(viol)[:160].replace('|', '/')} |")
cons = {
 'records': len(R),
 'api_O_equal_all': all(r['api_O_equal'] for r in R if 'exception' not in r['api']),
 'cli_stdout_byte_equal_expected_all_noncrash': all(r[t]['stdout_byte_equal_expected'] for r in R if 'exception' not in r['api'] for t in ('cli_normal', 'cli_O')),
 'cli_stderr_empty_all_noncrash': all(r[t]['stderr_tail'] == '' for r in R if 'exception' not in r['api'] for t in ('cli_normal', 'cli_O')),
 'input_unchanged_all': all(r['input_unchanged'] for r in R),
 'crash_cases': [r['case'] for r in R if 'exception' in r['api']],
}
lines += ['', '# Consistency', '', '```', json.dumps(cons, indent=1), '```']
pass
with open(W / 'results.tsv', 'w') as f:
    f.write('case\tkind\tvalue\tplacement\tapi_categories\tcli_normal_rc\tcli_O_rc\tapi_violations_or_exception\n')
    for r in R:
        viol = r['api'].get('violations') if 'exception' not in r['api'] else r['api']['exception']
        f.write('\t'.join(map(str, [r['case'], r['kind'], r.get('value', ''), r.get('placement', ''), ','.join(r['api_categories']) or 'none', r['cli_normal']['rc'], r['cli_O']['rc'], json.dumps(viol)])) + '\n')
print('\n'.join(lines))
