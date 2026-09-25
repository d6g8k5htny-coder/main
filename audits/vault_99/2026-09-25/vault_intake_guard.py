"""Fail-closed register-before-move guard for Vault 99. Workflow control only."""
REQUIRED={'drive_id','title','category','source_parent_id','reason','authority_replacement_id','authority_replacement_title','replacement_outside_vault','sole_copy','live_hold_research','active_pkg_content','routing_or_constitution','manifest_row_prepared','operator_confirmed'}
CATEGORIES={'A1','A2','A3','A4'}
def validate_entry(x):
    if not isinstance(x,dict): raise ValueError('entry must be object')
    missing=sorted(REQUIRED-set(x))
    if missing:return {'allowed':False,'reasons':['missing fields: '+','.join(missing)]}
    reasons=[]
    for k in ('replacement_outside_vault','sole_copy','live_hold_research','active_pkg_content','routing_or_constitution','manifest_row_prepared','operator_confirmed'):
        if type(x[k]) is not bool: reasons.append(k+' must be exact boolean')
    if x.get('category') not in CATEGORIES: reasons.append('category must be A1-A4')
    if not str(x.get('drive_id','')).strip(): reasons.append('drive_id required')
    if not str(x.get('authority_replacement_id','')).strip(): reasons.append('authority replacement ID required')
    if not str(x.get('authority_replacement_title','')).strip(): reasons.append('authority replacement title required')
    if x.get('replacement_outside_vault') is not True: reasons.append('replacement must remain outside vault')
    if x.get('sole_copy') is not False: reasons.append('sole copy forbidden')
    if x.get('live_hold_research') is not False: reasons.append('live HOLD research forbidden')
    if x.get('active_pkg_content') is not False: reasons.append('active PKG content forbidden')
    if x.get('routing_or_constitution') is not False: reasons.append('routing/constitution forbidden')
    if x.get('manifest_row_prepared') is not True: reasons.append('same-turn manifest row required before move')
    if x.get('operator_confirmed') is not True: reasons.append('operator confirmation required')
    return {'allowed':not reasons,'reasons':reasons,'scientific_effect':'NONE'}
