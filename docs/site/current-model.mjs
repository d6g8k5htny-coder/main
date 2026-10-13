const REPOSITORIES = new Set(['d6g8k5htny-coder/main','d6g8k5htny-coder/Math-']);
const SHA = /^[0-9a-f]{40}$/;
const DIGEST = /^[0-9a-f]{64}$/;
export const hasNode=(graph,id)=>Object.prototype.hasOwnProperty.call(graph.nodes,id);
export function safePath(path) {
  return typeof path==='string' && path.length>0 && !/[\\:\u0000-\u001f]/.test(path) && path.split('/').every(p=>p && p!=='.' && p!=='..');
}
export function sourceURL(source) {
  if(!REPOSITORIES.has(source?.repository)||!SHA.test(source?.commit)||!safePath(source?.path)) return null;
  return `https://github.com/${source.repository}/blob/${source.commit}/${source.path.split('/').map(encodeURIComponent).join('/')}`;
}
export function sourceForNode(snapshot,node) {
  return snapshot.sources.find(s=>s.repository===snapshot.graph.identity.repository && s.path===node?.source) || null;
}
export function nodePage(id) { return `nodes/${encodeURIComponent(id)}.html`; }
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
export function neighborhood(graph,expanded,limit=150) {
  const ids=[],known=new Set();
  const add=id=>{if(hasNode(graph,id)&&!known.has(id)&&ids.length<limit){known.add(id);ids.push(id);}};
  for(const id of expanded)add(id);
  for(const id of expanded)for(const e of graph.edges){if(e.from===id)add(e.to);if(e.to===id)add(e.from);}
  return {ids,edges:graph.edges.filter(e=>known.has(e.from)&&known.has(e.to)),total:Object.keys(graph.nodes).length};
}
export function readState(search,snapshot) {
  const q=new URLSearchParams(search);const node=q.get('atlas-node')||Object.keys(snapshot.graph.data.nodes)[0];const equation=q.get('equation')||null;
  let error=null;
  if(!hasNode(snapshot.graph.data,node))error=`Unknown node: ${node}. No selection inferred.`;
  if(equation&&!snapshot.equations.some(e=>e.id===equation))error=`Unknown equation: ${equation}. No selection inferred.`;
  if((q.get('snapshot')&&q.get('snapshot')!==snapshot.graph.identity.commit)||(q.get('main-snapshot')&&q.get('main-snapshot')!==snapshot.repositories.find(r=>r.repository.endsWith('/main')).commit))error='This bookmark names another snapshot. Use its pinned source; the current graph is not substituted.';
  return {node,error,equation,view:q.get('atlas-view')==='list'?'list':'graph',sourceQuery:q.get('source-q')||'',query:q.get('atlas-q')||''};
}

export function searchGraph(graph,query,limit=150) {
  const words=String(query||'').toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches=Object.keys(graph.nodes).filter(id=>words.every(word=>`${id} ${JSON.stringify(graph.nodes[id])}`.toLowerCase().includes(word)));
  const ids=matches.slice(0,limit),known=new Set(ids);
  return {ids,edges:graph.edges.filter(e=>known.has(e.from)&&known.has(e.to)),matches:matches.length,total:Object.keys(graph.nodes).length};
}
