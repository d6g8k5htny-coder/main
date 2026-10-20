import {OPEN_CLASSIFICATIONS, NON_RESEARCH_KINDS} from './dependency-model.mjs?site-release=221190dfe9a7e6a13a5becc8078185060ea4edb58f041a1ddf27ae7dcb88b456';

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
export async function parseVerifiedSnapshot(raw,identity,cryptoAPI=globalThis.crypto) {
  const bytes=raw instanceof Uint8Array?raw:new Uint8Array(raw);
  if(bytes.byteLength!==Number(identity.bytes))throw Error('Current payload byte count mismatch.');
  if(!DIGEST.test(identity.sha256)||!cryptoAPI?.subtle)throw Error('Current payload digest verification unavailable.');
  const hash=Array.from(new Uint8Array(await cryptoAPI.subtle.digest('SHA-256',bytes))).map(b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==identity.sha256)throw Error('Current payload digest mismatch.');
  return validateSnapshot(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
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
