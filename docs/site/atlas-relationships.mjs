// Presentation over an admitted graph. Preserve every typed, directed edge.
export function relationshipGroups(graph, selectedId) {
  if (!Object.hasOwn(graph.nodes, selectedId)) return [];
  return [
    {id:'dependencies', label:'Depends on', direction:'from', other:'to'},
    {id:'dependents', label:'Used by', direction:'to', other:'from'},
  ].map(({id,label,direction,other}) => ({id,label,items:graph.edges
    .filter(edge => edge[direction] === selectedId)
    .map(edge => ({nodeId:edge[other],from:edge.from,to:edge.to,required:edge.required,relation:edge.relation}))}));
}

export function createRelationshipSections(graph, selectedId, {element,nodeButton,reach}) {
  return relationshipGroups(graph,selectedId).map(group => {
    const section=element('section',undefined,'atlas-relations');section.id='atlas-'+group.id;
    const heading=element('h4',`${group.label} (${group.items.length})`);heading.id=section.id+'-heading';
    section.setAttribute('aria-labelledby',heading.id);
    const list=element('ul');
    if(!group.items.length)list.append(element('li','No direct relation recorded.'));
    group.items.forEach((edge,index) => {
      const item=element('li',undefined,'atlas-relation-card');
      const button=nodeButton(edge.nodeId,{compact:true});button.setAttribute('class',[button.getAttribute('class'),'atlas-relation-node'].filter(Boolean).join(' '));
      const kind=element('p',edge.required?'Required relationship':'Context relationship','atlas-relation-kind');
      kind.id=section.id+'-kind-'+index;button.setAttribute('aria-describedby',kind.id);
      const detail=element('details');detail.id=section.id+'-record-'+index;
      const summary=element('summary','Recorded relationship');
      summary.setAttribute('aria-label',`Recorded relationship: ${edge.from} → ${edge.to}`);
      const counts=reach(edge.nodeId);
      detail.append(summary,element('p',`Recorded type: ${edge.relation}`),
        element('p',`${edge.from} → ${edge.to}`),
        element('p',`Downstream nodes · all edges: ${counts.allEdge} · required edges: ${counts.requiredOnly}`));
      item.append(button,kind,detail);list.append(item);
    });
    section.append(heading,list);return section;
  });
}
