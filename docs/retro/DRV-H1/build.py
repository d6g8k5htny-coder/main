import json,re,collections,csv
rows=[]
for l in open('/workspace/r13/walk.jsonl'):
  d=json.loads(l)
  if not d['path'].startswith('01'): continue
  for c in d['children']:
    if c['mime_type']=='application/vnd.google-apps.folder': continue
    rows.append((d['path'],d['id'],c))
NAV=re.compile(r'START_HERE|START HERE|INDEX|README|ROUTER|HANDOFF|MANIFEST|\.sha256|NAV|LANE_MAP|LIVE-COPY',re.I)
BIN=re.compile(r'\.(zip|gz|tgz|png|jpg|sh|gs)$',re.I)
BINMIME=('application/gzip','image/png','application/zip','application/x-zip-compressed')
# rules: (regex on path+name, target, kind)
R=[
 (r'H5|JETMOD|OBL-H5','hist.OBL-H5-JETMOD','graph-node(hist; never bound)'),
 (r'CH-LIFT|CH_LIFT','hist.CH-LIFT','graph-node(hist; never bound)'),
 (r'ENV-RESCOV|RESCOV|rnu_env','hist.ENV-RESCOV','graph-node(hist; never bound)'),
 (r'ALLCELL|FDZ','hist.ALLCELL-FDZ-Q4','graph-node(hist; never bound)'),
 (r'LOGQ','hist.LOGQ-TAIL','graph-node(hist; never bound)'),
 (r'SYM-Fw|Fw[-_]jet','hist.SYM-Fw-jet','graph-node(hist; never bound)'),
 (r'Piece-2|PIECE_2','hist.Piece-2-annulus','graph-node(hist; never bound)'),
 (r'SIDE24|side24','math.side24-coefficient','graph-node'),
 (r'full[_ ]price|P15','math.p15-full-price','graph-node'),
 (r'annulus','math.rn-fixed-annulus-window','graph-node'),
 (r'remote','math.rn-fixed-remote-window','graph-node'),
 (r'collar','math.d5-component.collar-proof','graph-node'),
 (r'punctured|pin[-_ ]neigh|six[-_]pin','math.d5-pin-neighborhood-first-moment','graph-node'),
 (r'\bD1\b|D1_|d1_|matrix[-_ ]cap|marked[-_ ]cyl','math.uniform-matrix-cap-lifetime','graph-node'),
 (r'lifetime|remainder','math.lifetime-remainder','graph-node'),
 (r'RN[-_ ]?UNIF|RN5|\bRN[-_]','math.rn-count-interface','graph-node'),
 (r'mesoscopic','math.rn-mesoscopic-reduction','graph-node'),
 (r'LM00[6-9]','unit:coefficients/side24_d5_20260930','Math- unit (LM coefficients; see GB-8h..j)'),
 (r'\bD5\b|D5_|d5_','unit:frontiers/d5_dimension_lift_20260929','Math- unit'),
 (r'\bC6\b|C6_|c6_','unit:frontiers/c6_remote_coefficient_numerics_20260930','Math- unit'),
 (r'Sard|sard','unit:frontiers/sard_g_pinned_transfer_20260929','Math- unit'),
 (r'elder','unit:frontiers/em_intermediate_elder_20261006','Math- unit'),
 (r'\.lean$|\bLean\b|formal','unit:formal (GB-8p/q region)','Math- unit region'),
]
R=[(re.compile(a,re.I),b,c) for a,b,c in R]
def slug(s): return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')[:48]
out=[];cnt=collections.Counter();props=collections.Counter()
for path,pid,c in rows:
  n=c['name']; hay=path+'/'+n
  if NAV.search(n): continue
  if BIN.search(n) or c['mime_type'] in BINMIME: continue
  tgt=kind=None
  for rx,t,k in R:
    if rx.search(hay): tgt,kind=t,k;break
  if tgt: status='MAPPED_EXISTING'
  else:
    parts=path.split('/')
    if len(parts)>=2 and parts[1] not in ('NONAUTH','DROPBOX_STAGING','ROOT_DELIVERY','01_PLATFORM') and 'GOVERNANCE' not in path.upper():
      grp='/'.join(parts[1:3]); tgt='proposed.drv-h1.'+slug(grp); kind='PROPOSED_NEW_NODE (AUTHOR_SIDE_CANDIDATE max)'; status='PROPOSED_NEW_NODE'; props[tgt]+=1
    else:
      tgt='';kind='intake/staging/platform/governance: no math unit fits';status='UNMAPPED'
  cnt[status]+=1
  out.append(dict(drive_file_id=c['id'],drive_parent_id=pid,drive_path='01_ACTIVE_RESEARCH_PACKAGES/'+path[3:]+'/'+n if path!='01' else n,
    mime_type=c['mime_type'],size_bytes=c['size_bytes'],modified_time_utc=c['modified_time'],status=status,target=tgt,target_kind=kind,
    match_basis='filename/path keyword (heuristic, unreviewed)' if status!='UNMAPPED' else '',bound='NO',result='NOT A RESULT',credit=0,obl='OPEN'))
with open('DRV-H1.mapping.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
json.dump(dict(lane='DRV-H1',counts=cnt,proposed_nodes=props,total=len(out)),open('summary.json','w'),indent=1)
print(cnt,len(props),len(out),len(rows))
tc=collections.Counter(o['target'] for o in out if o['status']=='MAPPED_EXISTING');print(tc.most_common())
print(props.most_common())
