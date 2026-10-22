import {OPEN_CLASSIFICATIONS, NON_RESEARCH_KINDS} from './dependency-model.mjs?site-release=14af1552ef5b72df346782ea4c3d2a9e1fbeeeb89c9ec7421de6f63e4f8e7aff';

const REPOSITORIES = new Set(['d6g8k5htny-coder/main','d6g8k5htny-coder/Math-']);
const SHA = /^[0-9a-f]{40}$/;
const DIGEST = /^[0-9a-f]{64}$/;
export const hasNode=(graph,id)=>typeof id==='string' && !!graph?.nodes && Object.prototype.hasOwnProperty.call(graph.nodes,id);
export function safePath(path) {
  return typeof path==='string' && path.length>0 && !/[\\:\u0000-\u001f]/.test(path) && path.split('/').every(p=>p && p!=='.' && p!=='..');
}
export function sourceURL(source) {
  if(!REPOSITORIES.has(source?.repository)||!SHA.test(source?.commit)||!safePath(source?.path)) return null;
  return `https://github.com/${source.repository}/blob/${source.commit}/${source.path.split('/').map(encodeURIComponent).join('/')}`;
}
export function sourceForNode(snapshot,node) {
  if(!node || !Object.prototype.hasOwnProperty.call(node,'source') || typeof node.source!=='string')return null;
  return snapshot.sources.find(s=>s.repository===snapshot.graph.identity.repository && s.path===node?.source) || null;
}
export function nodePage(id) { return typeof id==='string' && id ? `nodes/${encodeURIComponent(id)}.html` : null; }
export function validateSnapshot(data) {
  if(data?.schema_version!==1||data.scientific_effect!=='NONE'||!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/.test(data.observed_at)||!Number.isFinite(Date.parse(data.observed_at))) throw Error('Invalid current source snapshot.');
  if(!Array.isArray(data.repositories)||data.repositories.length!==2) throw Error('Invalid repository inventory.');
  const commits=new Map();
  for(const r of data.repositories) {if(!REPOSITORIES.has(r.repository)||!SHA.test(r.commit)||!SHA.test(r.tree)||commits.has(r.repository))throw Error('Invalid pinned repository.');commits.set(r.repository,r.commit);}
  const identity=s=>sourceURL(s) && s.commit===commits.get(s.repository) && SHA.test(s.blob) && Number.isInteger(s.bytes) && s.bytes>=0;
  if(!Array.isArray(data.sources)||data.sources.some(s=>!identity(s))) throw Error('Invalid source identity.');
  if(!identity(data.graph?.identity)||!DIGEST.test(data.graph.identity.sha256)||!data.graph.data?.nodes||!Array.isArray(data.graph.data.edges)) throw Error('Invalid graph identity.');
  const nodes=data.graph.data.nodes;
  if(Array.isArray(nodes)||Object.values(nodes).some(n=>!n||typeof n.kind!=='string'||typeof n.classification!=='string'))throw Error('Invalid graph nodes.');
  for(const e of data.graph.data.edges)if(!Object.prototype.hasOwnProperty.call(nodes,e.from)||!Object.prototype.hasOwnProperty.call(nodes,e.to)||typeof e.required!=='boolean'||typeof e.relation!=='string')throw Error('Invalid graph reference.');
  if(!Array.isArray(data.equations)||data.equations.some(e=>!sourceURL(e)||e.commit!==commits.get(e.repository)||typeof e.id!=='string'||typeof e.text!=='string'||!DIGEST.test(e.sha256)||!Number.isInteger(e.line_start)||!Number.isInteger(e.line_end)||e.line_start<1||e.line_end<e.line_start))throw Error('Invalid equation identity.');
  if(!Array.isArray(data.references)||data.references.some(s=>!identity(s)||!DIGEST.test(s.sha256)))throw Error('Invalid source references.');
  return data;
}
class SnapshotIntegrityError extends Error {}
export async function parseVerifiedSnapshot(raw,identity,cryptoAPI=globalThis.crypto) {
  const bytes=raw instanceof Uint8Array?raw:new Uint8Array(raw);
  if(bytes.byteLength!==Number(identity.bytes))throw new SnapshotIntegrityError('Current payload byte count mismatch.');
  if(!DIGEST.test(identity.sha256)||!cryptoAPI?.subtle)throw Error('Current payload digest verification unavailable.');
  const hash=Array.from(new Uint8Array(await cryptoAPI.subtle.digest('SHA-256',bytes))).map(b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==identity.sha256)throw new SnapshotIntegrityError('Current payload digest mismatch.');
  return validateSnapshot(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
}
export async function fetchVerifiedSnapshot(fetcher,identity,{signal,cryptoAPI=globalThis.crypto}={}) {
  if(!DIGEST.test(identity?.sha256)||!Number.isSafeInteger(Number(identity?.bytes))||Number(identity.bytes)<0)throw Error('Invalid current source identity.');
  const read=async url=>{
    const response=await fetcher(url,{cache:'no-store',credentials:'omit',signal});
    if(!response.ok)throw Error('Current source snapshot unavailable.');
    return parseVerifiedSnapshot(await response.arrayBuffer(),identity,cryptoAPI);
  };
  try{return await read('current.json');}
  catch(error){
    if(!(error instanceof SnapshotIntegrityError))throw error;
    // A mutable alias can lag the page across publication. One digest-keyed
    // request may recover it; only the advertised bytes can be admitted.
    return read('current.json?observation='+identity.sha256);
  }
}
export function freshness(observed,now=Date.now()) {
  const age=now-Date.parse(observed);
  if(!Number.isFinite(age)||age < -60000)return 'clock disagreement';
  return age>7200000?'stale':'current';
}
export function filterSources(sources,query) {
  const words=String(query||'').toLowerCase().trim().split(/\s+/).filter(Boolean);
  return sources.filter(s=>words.every(w=>`${s.repository} ${s.path}`.toLowerCase().includes(w)));
}
export function evidenceAxes(node) {
  return ['source','review','kernel','computation','alignment'].map(axis=>({axis,state:(axis==='source' && !!node.source)||(axis==='review' && Object.keys(node).some(k=>k.startsWith('review_')&&node[k]))?'recorded':'not_recorded'}));
}
export const RESUME_NOT_RECORDED='not recorded in this snapshot';
export const RESUME_FRESHNESS_WARNING='Refresh from the live canonical record before any ownership- or head-dependent action. This block is a dated snapshot observation, not live task state.';
export const RESUME_REVIEW_LABEL='Recorded review metadata. This is a review record, not the current task, an owner claim or a completion record. A closed or accepted review does not establish that work on this node is finished.';
export const RESUME_OPERATIONAL=Object.freeze([['next_action','Next action'],['missing_input','Missing input'],['owner_evidence','Owner evidence'],['retained_stop','Retained stop']].map(row=>Object.freeze(row)));
export const RESUME_ISSUE_NOTE='Issue number recorded without a repository in this snapshot; no link is inferred.';
export const RESUME_NO_TASK='No explicitly identified task record for this node. A recorded review issue is not used as a task record.';
// Site-maintained task pointers, not graph metadata. Tools/current_site_history.py keeps the same table.
export const RESUME_TASK_RECORDS=Object.freeze({
  'math.d5-component.punctured-pin-proof':Object.freeze({url:'https://github.com/d6g8k5htny-coder/main/pull/327#issuecomment-6321244764',label:'Operational request: main#327 comment 6321244764',meaning:'A request for a source-only registration input for this node. It is an operational pointer, not scientific acceptance, source-binding admission, a registration or an owner claim.',identified_on:'2026-10-21',identified_in:'https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6321679618'}),
});
const REVIEW_URL=/^https:\/\/github\.com\/d6g8k5htny-coder\/(?:main|Math-)\/(?:issues|pull)\/[1-9][0-9]*(?:#[A-Za-z0-9_-]+)?$/;
const metadataText=value=>typeof value==='string'?value:JSON.stringify(value);
const NODE_STATUS_FORBIDDEN=new Set(['lemma_closed','prizes_solved','discharges_OBL_H5_JETMOD','certified_C_H','freeze','inventable_attempt_accepted']);
const NODE_STATUS_STATE=/^(?:fresh|stale|unknown)$/;
const NODE_STATUS_OBSERVED=/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
function nodeStatusForbidden(value){
  if(!value||typeof value!=='object')return false;
  return Object.entries(value).some(([key,item])=>NODE_STATUS_FORBIDDEN.has(key)||nodeStatusForbidden(item));
}
// A record without an evidence URL is not applied. Freshness is not scientific validity.
function nodeStatusRecord(statusRecords,id){
  if(!statusRecords||typeof statusRecords!=='object'||Array.isArray(statusRecords)||!Object.prototype.hasOwnProperty.call(statusRecords,id))return null;
  const record=statusRecords[id];
  if(!record||typeof record!=='object'||Array.isArray(record)||record.scientific_effect!=='NONE'||nodeStatusForbidden(record))return null;
  if(!Object.prototype.hasOwnProperty.call(record,'owner'))return null;
  const owner=record.owner;
  if(owner!==null&&(typeof owner!=='object'||Array.isArray(owner)||typeof owner.id!=='string'||!owner.id||typeof owner.evidence_url!=='string'||!owner.evidence_url))return null;
  return record;
}
function nodeStatusList(value){return Array.isArray(value)&&value.every(item=>typeof item==='string'&&item)&&value.length?value.join('; '):null;}
function nodeStatusOperational(record){
  const owner=record.owner;
  return RESUME_OPERATIONAL.map(([field,label])=>{
    const row={field,label,text:RESUME_NOT_RECORDED};
    if(field==='next_action'&&typeof record.next_action==='string'&&record.next_action){
      row.text=record.next_action;
      if(typeof record.status==='string'&&NODE_STATUS_STATE.test(record.status)&&typeof record.observed_at==='string'&&NODE_STATUS_OBSERVED.test(record.observed_at))row.note=`Node-status view ${record.status} as of ${record.observed_at}. Scientific effect NONE. Freshness is not scientific validity.`;
    }else if(field==='missing_input'){const text=nodeStatusList(record.missing_input);if(text)row.text=text;}
    else if(field==='owner_evidence'){row.text=owner===null?'No owner recorded in this node-status view.':owner.id;if(owner&&owner.evidence_url)row.url=owner.evidence_url;}
    else if(field==='retained_stop')row.text=nodeStatusList(record.stops)||'No retained stop recorded in this node-status view.';
    return row;
  });
}
// One projection feeds the atlas inspector and the static node pages; only `now` adds a freshness state.
// statusRecords is optional. The snapshot-only call stays unchanged for static pages.
export function resumeProjection(snapshot,id,now,statusRecords) {
  const graph=snapshot?.graph?.data;if(!hasNode(graph,id))return null;
  const node=graph.nodes[id],identity=snapshot.graph.identity,own=key=>Object.prototype.hasOwnProperty.call(node,key);
  const task=Object.prototype.hasOwnProperty.call(RESUME_TASK_RECORDS,id)?RESUME_TASK_RECORDS[id]:null;
  const record=nodeStatusRecord(statusRecords,id);
  return {node:id,
    snapshot:{observed_at:snapshot.observed_at,graph_repository:identity.repository,graph_commit:identity.commit,graph_path:identity.path,graph_sha256:identity.sha256,main_commit:snapshot.repositories.find(r=>r.repository==='d6g8k5htny-coder/main')?.commit??null},
    freshness:Number.isFinite(now)?freshness(snapshot.observed_at,now):null,warning:RESUME_FRESHNESS_WARNING,
    state:['classification','controlling'].map(field=>({field,text:own(field)&&node[field]!==null?metadataText(node[field]):RESUME_NOT_RECORDED})),
    review:{label:RESUME_REVIEW_LABEL,fields:Object.keys(node).filter(key=>key.startsWith('review_')).map(field=>{const value=node[field];return {field,text:metadataText(value),url:typeof value==='string'&&REVIEW_URL.test(value)?value:null,note:field==='review_issue'&&typeof value==='number'?RESUME_ISSUE_NOTE:null};})},
    operational:record?nodeStatusOperational(record):RESUME_OPERATIONAL.map(([field,label])=>({field,label,text:RESUME_NOT_RECORDED})),
    continuation:task?{...task}:null,continuation_text:task?null:RESUME_NO_TASK};
}
export const MAX_WINDOW_NODES=150;
export const REQUIRED_ROUTE_SEMANTICS='One shortest route per unresolved research target follows recorded dependent-to-prerequisite edges, with every edge required. The selected root is excluded as a target; contextual edges supply no required route.';
const REACH_LABELS=Object.freeze({allEdge:'All-edge reach (includes context)',requiredOnly:'Required-dependency reach'});
function unresolved(node) {
  return OPEN_CLASSIFICATIONS.has(node.classification) && !NON_RESEARCH_KINDS.has(node.kind);
}
function boundedOffset(value) {
  return Number.isSafeInteger(value) && value>=0 ? Math.min(value,100_000) : 0;
}
function options(value) {
  const selected=typeof value==='number'?{limit:value}:value || {};
  return {...selected,limit:Number.isInteger(selected.limit)&&selected.limit>0?Math.min(selected.limit,MAX_WINDOW_NODES):MAX_WINDOW_NODES,offset:boundedOffset(selected.offset)};
}
function windowed(items,selected) {
  const offset=Math.min(selected.offset,items.length),limit=selected.limit;
  const values=items.slice(offset,offset+limit);
  return {values,matches:items.length,omitted:items.length-values.length,window:{offset,limit,nextOffset:offset+limit<items.length?offset+limit:null,previousOffset:offset>0?Math.max(0,offset-limit):null}};
}
function nodeMatches(graph,id,selected) {
  const node=graph.nodes[id];
  return ['classification','kind','layer'].every(field=>!selected[field] || node[field]===selected[field]);
}
function graphWindow(graph,candidates,selected) {
  const page=windowed(candidates.filter(id=>nodeMatches(graph,id,selected)),selected),known=new Set(page.values);
  const {values,...counts}=page;
  return {ids:values,edges:graph.edges.filter(e=>known.has(e.from)&&known.has(e.to)),total:Object.keys(graph.nodes).length,...counts};
}
// Reverse traversal counts distinct dependents in the full graph, independent
// of the rendered window. Context edges count only in the all-edge lane.
export function downstreamReach(graph,id) {
  const known=hasNode(graph,id);
  const walk=requiredOnly=>{
    if(!known)return [];
    const visited=new Set([id]),queue=[id];
    for(let cursor=0;cursor<queue.length;cursor++)for(const edge of graph.edges) {
      if(edge.to!==queue[cursor] || (requiredOnly && edge.required!==true) || !hasNode(graph,edge.from) || visited.has(edge.from))continue;
      visited.add(edge.from);queue.push(edge.from);
    }
    visited.delete(id);return [...visited].sort((a,b)=>a.localeCompare(b));
  };
  const allEdgeIds=walk(false),requiredOnlyIds=walk(true);
  return {known,allEdge:allEdgeIds.length,requiredOnly:requiredOnlyIds.length,allEdgeIds,requiredOnlyIds,labels:REACH_LABELS};
}
export function unresolvedTargets(graph) {
  return Object.entries(graph.nodes).filter(([,node])=>unresolved(node)).map(([id,node])=>({id,node,...downstreamReach(graph,id)})).sort((a,b)=>b.allEdge-a.allEdge || a.id.localeCompare(b.id));
}
// Breadth-first search follows source edges forward (dependency direction),
// keeping one shortest route per target with original relation metadata.
export function requiredDependencyPaths(graph,id) {
  if(!hasNode(graph,id))return [];
  const queue=[{ids:[id],edges:[]}],visited=new Set([id]),routes=[];
  for(let cursor=0;cursor<queue.length;cursor++) {
    const path=queue[cursor],current=path.ids.at(-1);
    if(current!==id && unresolved(graph.nodes[current]))routes.push(path);
    const dependencies=graph.edges.filter(edge=>edge.from===current && edge.required===true && hasNode(graph,edge.to)).sort((a,b)=>a.to.localeCompare(b.to) || a.relation.localeCompare(b.relation));
    for(const edge of dependencies) {
      if(visited.has(edge.to))continue;
      visited.add(edge.to);queue.push({ids:[...path.ids,edge.to],edges:[...path.edges,edge]});
    }
  }
  return routes.sort((a,b)=>a.ids.length-b.ids.length || a.ids.at(-1).localeCompare(b.ids.at(-1)));
}
export function neighborhood(graph,expanded,limit=MAX_WINDOW_NODES) {
  const ids=[],known=new Set();
  const add=id=>{if(hasNode(graph,id)&&!known.has(id)){known.add(id);ids.push(id);}};
  for(const id of expanded)add(id);
  const rootCount=ids.length;
  for(const id of expanded)for(const e of graph.edges){if(e.from===id)add(e.to);if(e.to===id)add(e.from);}
  return {...graphWindow(graph,ids,options(limit)),candidateCount:ids.length,rootCount,frontierCount:ids.length-rootCount};
}
export function readState(search,snapshot) {
  const q=new URLSearchParams(search);const node=q.get('atlas-node')||(hasNode(snapshot.graph.data,'math.uniform-matrix-cap-lifetime')?'math.uniform-matrix-cap-lifetime':Object.keys(snapshot.graph.data.nodes)[0]);const equation=q.get('equation')||null;
  let error=null;
  if(!hasNode(snapshot.graph.data,node))error=`Unknown node: ${node}. No selection inferred.`;
  if(equation&&!snapshot.equations.some(e=>e.id===equation))error=`Unknown equation: ${equation}. No selection inferred.`;
  if((q.get('snapshot')&&q.get('snapshot')!==snapshot.graph.identity.commit)||(q.get('main-snapshot')&&q.get('main-snapshot')!==snapshot.repositories.find(r=>r.repository.endsWith('/main')).commit))error='This bookmark names another snapshot. Use its pinned source; the current graph is not substituted.';
  const rawOffset=q.get('atlas-offset')||'0';
  return {node,error,equation,view:q.get('atlas-view')==='list'?'list':'graph',sourceQuery:q.get('source-q')||'',query:q.get('atlas-q')||'',offset:/^\d+$/.test(rawOffset)?boundedOffset(Number(rawOffset)):0,equationQuery:q.get('equation-q')||'',classification:q.get('atlas-classification')||'',kind:q.get('atlas-kind')||'',layer:q.get('atlas-layer')||''};
}

export function searchGraph(graph,query,limit=MAX_WINDOW_NODES) {
  const words=String(query||'').toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches=Object.keys(graph.nodes).filter(id=>words.every(word=>`${id} ${JSON.stringify(graph.nodes[id])}`.toLowerCase().includes(word)));
  return graphWindow(graph,matches,options(limit));
}

export function searchEquations(equations,query,limit=MAX_WINDOW_NODES) {
  const words=String(query||'').toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches=equations.filter(e=>words.every(word=>`${e.text} ${e.repository} ${e.path}`.toLowerCase().includes(word)));
  const {values,...counts}=windowed(matches,options(limit));
  return {equations:values,total:equations.length,...counts};
}

// Current inventory content associations do not classify or bind graph nodes.
export const ACQUISITION_LABELS=Object.freeze({metadata_only:'Metadata only',graph_data_included:'Parsed graph',source_identity_only:'Reference identity',excerpts_included:'Included excerpts',unavailable:'Unavailable'});
export const ACQUISITION_UNAVAILABLE='Acquisition associations could not be qualified for this observation; no content association inferred.';
export async function projectCurrentAcquisitions(data) {
  const reasons=new Set(),rows=[],index=new Map(),commits=new Map();
  const result=()=>({disposition:reasons.size?'unavailable':'qualified',reasons:[...reasons].sort(),rows});
  if(!data||typeof data!=='object'||Array.isArray(data)){reasons.add('invalid_acquisition_input');return result();}
  const unicode=s=>typeof s==='string'&&!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(s);
  const path=s=>unicode(s)&&[...s].length<=4096&&safePath(s)&&!/[\u007f%]/.test(s);
  if(!Array.isArray(data.repositories)||data.repositories.length!==2)reasons.add('invalid_acquisition_input');
  else for(const r of data.repositories){if(!r||!REPOSITORIES.has(r.repository)||typeof r.commit!=='string'||!SHA.test(r.commit)||commits.has(r.repository))reasons.add('invalid_acquisition_input');else commits.set(r.repository,r.commit);}
  const identity=(s,whole=true)=>s&&typeof s==='object'&&REPOSITORIES.has(s.repository)&&typeof s.commit==='string'&&SHA.test(s.commit)&&s.commit===commits.get(s.repository)&&path(s.path)&&(!whole||(typeof s.blob==='string'&&SHA.test(s.blob)&&Number.isSafeInteger(s.bytes)&&s.bytes>=0&&s.bytes<=16*1024*1024))?[s.repository,s.commit,s.path]:null;
  const inventoryShape=Array.isArray(data.sources);
  if(!inventoryShape)reasons.add('invalid_acquisition_input');
  for(const s of inventoryShape?data.sources:[]){const key=identity(s),encoded=JSON.stringify(key),row={key,identity:key?[...key,s.blob,s.bytes]:null,categories:[],whole_files:[],excerpts:[]};rows.push(row);if(!key)reasons.add('invalid_acquisition_identity');else if(index.has(encoded))reasons.add('duplicate_inventory_identity');else index.set(encoded,{source:s,row});}
  const wholeSeen=new Set();
  const whole=(s,category,graph=false)=>{const key=identity(s);if(!key||(graph&&s.repository!=='d6g8k5htny-coder/Math-')||typeof s.sha256!=='string'||!DIGEST.test(s.sha256)){reasons.add('invalid_acquisition_identity');return;}const encoded=JSON.stringify(key);if(wholeSeen.has(encoded))reasons.add('duplicate_acquisition_identity');wholeSeen.add(encoded);if(!inventoryShape)return;const match=index.get(encoded);if(!match){reasons.add('missing_acquisition_inventory');return;}if(match.source.blob!==s.blob)reasons.add('acquisition_blob_conflict');if(match.source.bytes!==s.bytes)reasons.add('acquisition_bytes_conflict');match.row.categories.push(category);match.row.whole_files.push({category,sha256:s.sha256});};
  if(!data.graph||typeof data.graph.identity!=='object'||!data.graph.identity||Array.isArray(data.graph.identity))reasons.add('invalid_acquisition_input');else whole(data.graph.identity,'graph_data_included',true);
  if(!Array.isArray(data.references))reasons.add('invalid_acquisition_input');else for(const s of data.references)whole(s,'source_identity_only');
  const ids=new Set(),spans=new Set();
  if(!Array.isArray(data.equations))reasons.add('invalid_acquisition_input');else for(const e of data.equations){const key=identity(e,false);if(!key||!unicode(e.id)||!e.id||!unicode(e.text)||!Number.isSafeInteger(e.line_start)||!Number.isSafeInteger(e.line_end)||e.line_start<1||e.line_end<e.line_start||typeof e.sha256!=='string'||!DIGEST.test(e.sha256)){reasons.add('invalid_excerpt_record');continue;}
    let digest;try{digest=Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(e.text))),b=>b.toString(16).padStart(2,'0')).join('');}catch{reasons.add('invalid_excerpt_record');continue;}
    if(digest!==e.sha256){reasons.add('invalid_excerpt_record');continue;}const span=JSON.stringify([...key,e.line_start,e.line_end]);if(ids.has(e.id)||spans.has(span))reasons.add('duplicate_excerpt_identity');ids.add(e.id);spans.add(span);if(!inventoryShape)continue;const match=index.get(JSON.stringify(key));if(!match){reasons.add('missing_excerpt_inventory');continue;}if(!match.row.categories.includes('excerpts_included'))match.row.categories.push('excerpts_included');match.row.excerpts.push({id:e.id,line_start:e.line_start,line_end:e.line_end,sha256:e.sha256});
  }
  for(const row of rows)if(reasons.size){row.categories=['unavailable'];row.whole_files=[];row.excerpts=[];}else if(!row.categories.length)row.categories=['metadata_only'];
  return result();
}
