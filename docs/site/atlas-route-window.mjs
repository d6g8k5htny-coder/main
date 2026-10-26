import {OPEN_CLASSIFICATIONS, NON_RESEARCH_KINDS} from './dependency-model.mjs?site-release=14af1552ef5b72df346782ea4c3d2a9e1fbeeeb89c9ec7421de6f63e4f8e7aff';

export const ROUTE_WINDOW_LIMIT=150;
const snapshotIndexes=new WeakMap();
const hasNode=(graph,id)=>typeof id==='string'&&!!graph?.nodes&&Object.prototype.hasOwnProperty.call(graph.nodes,id);
const unresolved=node=>OPEN_CLASSIFICATIONS.has(node.classification)&&!NON_RESEARCH_KINDS.has(node.kind);

function graphIndex(graph){
  let index=snapshotIndexes.get(graph);if(index)return index;
  const incoming=new Map(),required=new Map();
  for(const id of Object.keys(graph.nodes)){incoming.set(id,[]);required.set(id,[]);}
  for(const edge of graph.edges){
    if(!hasNode(graph,edge.from)||!hasNode(graph,edge.to))continue;
    incoming.get(edge.to).push(edge);
    if(edge.required===true)required.get(edge.from).push(edge);
  }
  for(const edges of required.values())edges.sort((a,b)=>a.to.localeCompare(b.to)||a.relation.localeCompare(b.relation));
  index={incoming,required,analysis:null,counts:new Map()};snapshotIndexes.set(graph,index);return index;
}

export function downstreamReachCounts(graph,id){
  const index=graphIndex(graph);if(index.counts.has(id))return index.counts.get(id);
  const known=hasNode(graph,id);
  const walk=requiredOnly=>{
    if(!known)return 0;
    const visited=new Set([id]),queue=[id];
    for(let cursor=0;cursor<queue.length;cursor++)for(const edge of index.incoming.get(queue[cursor])){
      if((requiredOnly&&edge.required!==true)||visited.has(edge.from))continue;
      visited.add(edge.from);queue.push(edge.from);
    }
    return visited.size-1;
  };
  const result={known,allEdge:walk(false),requiredOnly:walk(true)};
  index.counts.set(id,result);return result;
}

export function requiredDependencyAnalysis(graph,id){
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
  const limit=Number.isInteger(selected.limit)&&selected.limit>0?Math.min(selected.limit,ROUTE_WINDOW_LIMIT):ROUTE_WINDOW_LIMIT;
  const pages=Math.max(1,Math.ceil(total/limit)),requested=Number.isSafeInteger(selected.page)&&selected.page>0?selected.page:1,page=Math.min(requested,pages);
  return {page,pages,limit,offset:(page-1)*limit};
}

export function requiredDependencyWindow(analysis,selected={}){
  const paging=pageWindow(analysis.totalOccurrences,selected),{offset,limit}=paging,end=Math.min(offset+limit,analysis.totalOccurrences),segments=[];let reconstructedNodes=0;
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

export function routePageFromSearch(search){
  const raw=new URLSearchParams(search).get('atlas-route-page')||'1',value=Number(raw);
  return /^\d+$/.test(raw)&&Number.isSafeInteger(value)&&value>0?value:1;
}
