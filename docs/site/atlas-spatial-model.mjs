// Geometry is presentation only. Never classify nodes or infer missing edges.
export function spatialProjection(graph, displayed, selected, focus=false) {
  let ids=[...new Set(displayed)].filter(id=>Object.hasOwn(graph.nodes,id)).slice(0,150);
  if(focus){const dependencies=new Set(graph.edges.filter(e=>e.from===selected).map(e=>e.to));ids=ids.filter(id=>id===selected||dependencies.has(id));}
  const visible=new Set(ids),edges=graph.edges.filter(e=>visible.has(e.from)&&visible.has(e.to));
  const positions={},rings=new Map();
  // Stable lexical rings put the selected claim at the centre. Distance is not
  // mathematical strength; direction and relation remain in edges/list text.
  const distance=new Map();if(visible.has(selected))distance.set(selected,0);
  const queue=distance.size?[selected]:[];
  for(let i=0;i<queue.length;i++)for(const edge of edges){const other=edge.from===queue[i]?edge.to:edge.to===queue[i]?edge.from:null;if(other&&!distance.has(other)){distance.set(other,distance.get(queue[i])+1);queue.push(other);}}
  for(const id of [...ids].sort()){const ring=distance.get(id)??3;if(!rings.has(ring))rings.set(ring,[]);rings.get(ring).push(id);}
  for(const [ring,nodes]of rings)nodes.forEach((id,index)=>{const angle=2*Math.PI*index/nodes.length-.35;const radius=ring===0?0:Math.max(5,ring*6,nodes.length*.85);positions[id]=[Math.cos(angle)*radius,ring===0?0:Math.sin(angle)*radius*.68,(ring-1)*2+Math.sin(angle*2)*1.6];});
  return {ids,edges,positions};
}
