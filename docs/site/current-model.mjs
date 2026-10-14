import {OPEN_CLASSIFICATIONS, NON_RESEARCH_KINDS} from './dependency-model.mjs?site-release=81664321ae3536ab28eeb2dd5e8b18e82157243b8a9d89b25c23588d21008c0c';

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
// An admitted snapshot is immutable. Build its adjacency once, retaining source
// edge identity/order for relation inspection and a separately sorted BFS lane.
const snapshotIndexes=new WeakMap();
function graphIndex(graph) {
  let index=snapshotIndexes.get(graph);if(index)return index;
  const outgoing=new Map(),incoming=new Map(),required=new Map();
  for(const id of Object.keys(graph.nodes)){outgoing.set(id,[]);incoming.set(id,[]);required.set(id,[]);}
  for(const edge of graph.edges){
    if(!hasNode(graph,edge.from)||!hasNode(graph,edge.to))continue;
    outgoing.get(edge.from).push(edge);incoming.get(edge.to).push(edge);
    if(edge.required===true)required.get(edge.from).push(edge);
  }
  for(const edges of required.values())edges.sort((a,b)=>a.to.localeCompare(b.to)||a.relation.localeCompare(b.relation));
  index={outgoing,incoming,required,analysis:null};snapshotIndexes.set(graph,index);return index;
}
// Complete indexed reverse traversal. Count-only callers retain no reachable-ID
// arrays; the public full API keeps its exact sorted-ID contract.
function reverseReach(graph,id,includeIds) {
  const known=hasNode(graph,id),index=graphIndex(graph);
  const walk=requiredOnly=>{
    if(!known)return new Set();
    const visited=new Set([id]),queue=[id];
    for(let cursor=0;cursor<queue.length;cursor++)for(const edge of index.incoming.get(queue[cursor])){
      if((requiredOnly&&edge.required!==true)||visited.has(edge.from))continue;
      visited.add(edge.from);queue.push(edge.from);
    }
    visited.delete(id);return visited;
  };
  const all=walk(false),required=walk(true),result={known,allEdge:all.size,requiredOnly:required.size,labels:REACH_LABELS};
  if(includeIds){result.allEdgeIds=[...all].sort((a,b)=>a.localeCompare(b));result.requiredOnlyIds=[...required].sort((a,b)=>a.localeCompare(b));}
  return result;
}
export function downstreamReach(graph,id){return reverseReach(graph,id,true);}
export function downstreamReachCounts(graph,id){return reverseReach(graph,id,false);}
export function unresolvedTargets(graph) {
  return Object.entries(graph.nodes).filter(([,node])=>unresolved(node)).map(([id,node])=>({id,node,...downstreamReach(graph,id)})).sort((a,b)=>b.allEdge-a.allEdge || a.id.localeCompare(b.id));
}
// Complete deterministic BFS stores one predecessor/depth per reached node.
// Only the selected root's compact analysis is cached, never all roots' paths.
export function requiredDependencyAnalysis(graph,id) {
  const index=graphIndex(graph);if(index.analysis?.root===id)return index.analysis;
  const known=hasNode(graph,id),predecessors=new Map(),depths=new Map(),targets=[],queue=known?[id]:[];let traversedEdges=0;
  if(known)depths.set(id,0);
  for(let cursor=0;cursor<queue.length;cursor++){
    const current=queue[cursor],depth=depths.get(current);
    if(current!==id&&unresolved(graph.nodes[current]))targets.push({id:current,length:depth+1});
    for(const edge of index.required.get(current)){
      traversedEdges++;if(depths.has(edge.to))continue;
      predecessors.set(edge.to,edge);depths.set(edge.to,depth+1);queue.push(edge.to);
    }
  }
  targets.sort((a,b)=>a.length-b.length||a.id.localeCompare(b.id));
  const prefixSums=[0];for(const target of targets)prefixSums.push(prefixSums.at(-1)+target.length);
  const analysis={root:id,known,predecessors,depths,targets,prefixSums,targetCount:targets.length,totalOccurrences:prefixSums.at(-1),traversedEdges};
  index.analysis=analysis;return analysis;
}
function reconstructRoute(analysis,target){
  const ids=new Array(target.length),edges=new Array(target.length-1);let current=target.id;
  for(let i=target.length-1;i>=0;i--){ids[i]=current;if(i){const edge=analysis.predecessors.get(current);edges[i-1]=edge;current=edge.from;}}
  return {ids,edges};
}
function pageWindow(total,selected={}){
  const limit=Number.isInteger(selected.limit)&&selected.limit>0?Math.min(selected.limit,MAX_WINDOW_NODES):MAX_WINDOW_NODES;
  const pages=Math.max(1,Math.ceil(total/limit)),requested=Number.isSafeInteger(selected.page)&&selected.page>0?selected.page:1,page=Math.min(requested,pages);
  return {page,pages,limit,offset:(page-1)*limit};
}
export function requiredDependencyWindow(analysis,selected={}){
  const paging=pageWindow(analysis.totalOccurrences,selected),{offset,limit}=paging,end=Math.min(offset+limit,analysis.totalOccurrences),segments=[];let reconstructedNodes=0;
  // Binary search the first intersecting route; direct last-page access never
  // materializes preceding routes. At most two boundary routes can be partial.
  let lo=0,hi=analysis.targetCount;
  while(lo<hi){const mid=Math.floor((lo+hi)/2);if(analysis.prefixSums[mid+1]<=offset)lo=mid+1;else hi=mid;}
  for(let i=lo;i<analysis.targetCount&&analysis.prefixSums[i]<end;i++){
    const target=analysis.targets[i],start=Math.max(0,offset-analysis.prefixSums[i]),stop=Math.min(target.length,end-analysis.prefixSums[i]);
    const route=reconstructRoute(analysis,target);reconstructedNodes+=route.ids.length;
    const ids=route.ids.slice(start,stop),incomingEdges=ids.map((_,j)=>start+j?route.edges[start+j-1]:null);
    segments.push({target:target.id,routeIndex:i,start,end:stop,length:target.length,complete:start===0&&stop===target.length,ids,incomingEdges});
  }
  return {...paging,segments,occurrences:end-offset,targetCount:analysis.targetCount,totalOccurrences:analysis.totalOccurrences,omitted:analysis.totalOccurrences-(end-offset),reconstructedNodes};
}
export function directRelationWindow(graph,id,direction,selected={}){
  const index=graphIndex(graph),rows=(direction==='outgoing'?index.outgoing:index.incoming).get(id)||[],paging=pageWindow(rows.length,selected);
  return {...paging,rows:rows.slice(paging.offset,paging.offset+paging.limit),total:rows.length,omitted:rows.length-Math.min(paging.limit,rows.length-paging.offset)};
}
// Compatibility API for callers explicitly requesting all output. The atlas
// uses requiredDependencyWindow instead; this full output is inherently large.
export function requiredDependencyPaths(graph,id){const analysis=requiredDependencyAnalysis(graph,id);return analysis.targets.map(target=>reconstructRoute(analysis,target));}
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
  const rawRoutePage=q.get('atlas-route-page')||'1',routePage=/^\d+$/.test(rawRoutePage)&&Number.isSafeInteger(Number(rawRoutePage))&&Number(rawRoutePage)>0?Number(rawRoutePage):1;
  return {node,error,equation,view:q.get('atlas-view')==='list'?'list':'graph',sourceQuery:q.get('source-q')||'',query:q.get('atlas-q')||'',offset:/^\d+$/.test(rawOffset)?boundedOffset(Number(rawOffset)):0,routePage,equationQuery:q.get('equation-q')||'',classification:q.get('atlas-classification')||'',kind:q.get('atlas-kind')||'',layer:q.get('atlas-layer')||''};
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
