import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {
  ROUTE_WINDOW_LIMIT,
  downstreamReachCounts,
  requiredDependencyAnalysis,
  requiredDependencyWindow,
  routePageFromSearch,
} from '../docs/site/atlas-route-window.mjs';

const topology=(edges,metadata={})=>({
  nodes:Object.fromEntries([...new Set(edges.flatMap(edge=>[edge.from,edge.to]))].map(id=>[
    id,{kind:'candidate',classification:'OPEN_ACTIVE',layer:'D1',...metadata[id]},
  ])),
  edges:edges.map(edge=>({relation:'uses',...edge})),
});
const diamond=()=>topology([
  {from:'A',to:'B',required:true},
  {from:'A',to:'C',required:true},
  {from:'B',to:'D',required:true},
  {from:'C',to:'D',required:true},
],Object.fromEntries(['A','B','C'].map(id=>[id,{classification:'PROVED_REVIEWED'}])));

test('route page bookmark accepts only positive safe integers',()=>{
  for(const [raw,expected] of [['1',1],['333367',333367],['0',1],['-1',1],['1.5',1],['Infinity',1],['9007199254740992',1],['',1]])
    assert.equal(routePageFromSearch('?atlas-route-page='+raw),expected);
});

test('bounded route windows concatenate to exact shortest required routes',()=>{
  const graph=diamond();
  graph.nodes.E={kind:'candidate',classification:'OPEN_ACTIVE',layer:'D1'};
  graph.edges.push({from:'D',to:'E',required:true,relation:'continuation',extra:'literal'});
  const analysis=requiredDependencyAnalysis(graph,'A');
  assert.equal(analysis.targetCount,2);
  assert.equal(analysis.totalOccurrences,7);
  const actual=[];
  for(let page=1;page<=Math.ceil(analysis.totalOccurrences/2);page++){
    const result=requiredDependencyWindow(analysis,{page,limit:2});
    assert.ok(result.occurrences<=2);
    for(const segment of result.segments)for(let i=0;i<segment.ids.length;i++)
      actual.push({target:segment.target,position:segment.start+i,id:segment.ids[i],edge:segment.incomingEdges[i]});
  }
  assert.deepEqual(actual.map(row=>[row.target,row.position,row.id]),[
    ['D',0,'A'],['D',1,'B'],['D',2,'D'],
    ['E',0,'A'],['E',1,'B'],['E',2,'D'],['E',3,'E'],
    // D is the only unresolved target before E; C is an alternate equal-length path.
  ].slice(0,actual.length));
  const all=requiredDependencyWindow(analysis,{page:1,limit:ROUTE_WINDOW_LIMIT});
  assert.deepEqual(all.segments.map(segment=>segment.ids),[['A','B','D'],['A','B','D','E']]);
  assert.equal(all.segments[1].incomingEdges[3],graph.edges[4]);
  assert.equal(requiredDependencyWindow(analysis,{page:999,limit:2}).page,Math.ceil(analysis.totalOccurrences/2));
  for(const id of ['missing','constructor','__proto__']){
    const empty=requiredDependencyAnalysis(graph,id);
    assert.equal(empty.targetCount,0);
    assert.equal(requiredDependencyWindow(empty).occurrences,0);
  }
});

function chainFixture(size){
  const nodes={},edges=[];
  for(let i=0;i<size;i++){
    const id='chain.'+String(i).padStart(5,'0');
    nodes[id]={kind:'candidate',classification:i?'OPEN_ACTIVE':'PROVED_REVIEWED',layer:'D1'};
    if(i)edges.push({from:'chain.'+String(i-1).padStart(5,'0'),to:id,required:true,relation:'chain',ordinal:i});
  }
  return {nodes,edges};
}

test('10000-node chain analysis stays linear and first middle last windows stay bounded',()=>{
  const graph=chainFixture(10000),analysis=requiredDependencyAnalysis(graph,'chain.00000');
  assert.equal(analysis.targetCount,9999);
  assert.equal(analysis.totalOccurrences,50004999);
  assert.equal(analysis.predecessors.size,9999);
  assert.equal(analysis.depths.size,10000);
  assert.equal(analysis.traversedEdges,9999);
  const pages=Math.ceil(analysis.totalOccurrences/ROUTE_WINDOW_LIMIT);
  for(const page of [1,Math.floor(pages/2),pages]){
    const result=requiredDependencyWindow(analysis,{page});
    assert.ok(result.occurrences<=ROUTE_WINDOW_LIMIT);
    assert.equal(result.offset,(page-1)*ROUTE_WINDOW_LIMIT);
    assert.equal(result.totalOccurrences,50004999);
    assert.equal(result.omitted,50004999-result.occurrences);
    assert.ok(result.reconstructedNodes<=2*10000+ROUTE_WINDOW_LIMIT);
    assert.equal(result.segments.reduce((count,segment)=>count+segment.ids.length,0),result.occurrences);
    for(const segment of result.segments)for(let i=0;i<segment.ids.length;i++){
      const position=segment.start+i;
      assert.equal(segment.ids[i],'chain.'+String(position).padStart(5,'0'));
      assert.equal(segment.incomingEdges[i],position?graph.edges[position-1]:null);
    }
  }
  assert.equal(requiredDependencyAnalysis(graph,'chain.00000'),analysis);
});

test('indexed downstream counts remain exact without retaining reachable ID arrays',()=>{
  const graph=chainFixture(10000);
  assert.deepEqual(downstreamReachCounts(graph,'chain.09999'),{known:true,allEdge:9999,requiredOnly:9999});
  assert.deepEqual(downstreamReachCounts(graph,'chain.00000'),{known:true,allEdge:0,requiredOnly:0});
  assert.deepEqual(downstreamReachCounts(graph,'missing'),{known:false,allEdge:0,requiredOnly:0});
});

test('snapshot index reads source edges once across route pages and reach counts',()=>{
  const source=chainFixture(400);let reads=0,iterations=0;
  const edges=new Proxy(source.edges,{get(target,key,receiver){
    if(key===Symbol.iterator)return function*(){for(const edge of target){iterations++;yield edge;}};
    return Reflect.get(target,key,receiver);
  }});
  const graph={nodes:source.nodes,get edges(){reads++;return edges;}};
  const analysis=requiredDependencyAnalysis(graph,'chain.00000');
  assert.equal(reads,1);
  assert.equal(iterations,399);
  for(const page of [1,25,500,1])requiredDependencyWindow(analysis,{page});
  for(const id of ['chain.00010','chain.00399'])downstreamReachCounts(graph,id);
  assert.equal(requiredDependencyAnalysis(graph,'chain.00000'),analysis);
  assert.equal(reads,1);
  assert.equal(iterations,399);
  assert.equal(analysis.targets.some(target=>'ids' in target||'edges' in target),false);
});


test('current atlas consumes the bounded route window instead of materializing all routes',async()=>{
 const source=await readFile(new URL('../docs/site/atlas.mjs',import.meta.url),'utf8');
 assert.match(source,/atlas-route-window\.mjs\?site-release=[0-9a-f]{64}/);
 assert.match(source,/requiredDependencyAnalysis/);assert.match(source,/requiredDependencyWindow/);
 assert.match(source,/data-route-node/);assert.match(source,/atlas-route-page/);
 assert.doesNotMatch(source,/requiredDependencyPaths/);
});
