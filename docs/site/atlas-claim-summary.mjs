import {element,anchor} from './current.mjs?site-release=fc7e8a98c1e23f476962c4c83e95ed3ce817490007987112d9b2cf6d3d90b142';
import {sourceForNode,sourceURL} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';

// Presentation over the controller's admitted record. No acquisition, source
// adoption, proof coverage inference, or scientific-status authority lives here.
export function createClaimSummary(data,node) {
  const summary=element('section');summary.id='atlas-claim-summary';
  summary.append(element('h4',node.scope?'Recorded scope':node.notes?'Recorded note':'Scope'),
    element('p',node.scope||node.notes||'Statement and scope extraction not recorded. Inspect the source reference.'));
  const source=sourceForNode(data,node);
  if(source){
    summary.append(anchor('Immutable source reference (document bytes not acquired by this graph view)',sourceURL(source)),
      element('p',`${source.repository} · ${source.commit.slice(0,8)}`,'atlas-source-pin'));
    const identity=element('details');identity.id='atlas-source-details';
    identity.append(element('summary','Source revision and blob'),
      element('p',`Source revision ${source.commit} · ${source.path}`),
      element('p',`Git blob ${source.blob} · ${source.bytes} bytes in the pinned tree. Reference presence is not mathematical verification.`));
    summary.append(identity);
  }else summary.append(element('p',node.source?`Unresolved source reference: ${node.source}. No matched public file in this pinned source inventory.`:'Source reference not recorded.'));

  const formal=element('section');formal.id='atlas-formal-evidence';
  formal.append(element('h4','Formal evidence for this claim'),
    element('p','No verified graph-to-formal-artifact mapping is recorded in this source projection. Missing evidence remains unknown.'));
  const details=element('details');details.id='atlas-formal-details';
  details.append(element('summary','Inspect declaration, execution and alignment'));
  const fields=element('dl');
  for(const [name,value]of [
    ['Acquisition','Graph metadata and pinned file inventory; proof document bytes are not acquired by this graph view.'],
    ['Mathematical review','Recorded review fields retain their original scope; no formal alignment or scientific acceptance follows.'],
    ['Declaration and hypotheses','No mapped declaration or hypothesis record. Read the exact mathematical source and its scope.'],
    ['Graph-source revision (formal artifact unmapped)',`${data.graph.identity.repository} ${data.graph.identity.commit} · ${data.graph.identity.path}`],
    ['Checked execution','No mapped trusted kernel-execution receipt. A green CI run alone does not establish coverage.'],
    ['Alignment','No mapped exact-source alignment review.'],
    ['Scientific acceptance','Unchanged; neither visualization nor formal execution promotes a claim.']
  ])fields.append(element('dt',name),element('dd',value));
  details.append(fields);
  const discovery=element('details');discovery.id='atlas-formal-packages';
  discovery.append(element('summary','Explore formal packages separately (unmapped)'),
    element('p','These are source references in the pinned inventory, not formal evidence attached to the selected claim. Open the manifests for declarations, scope and dependency pins; inspect trusted execution and alignment separately.'));
  for(const item of data.sources.filter(s=>['formal/manifest.json','formal/SCOPE.md','formal/ALIGNMENT.md'].includes(s.path))){
    const paragraph=element('p');paragraph.append(anchor(`${item.repository} · ${item.path} · ${item.commit.slice(0,8)}`,sourceURL(item)));discovery.append(paragraph);
  }
  details.append(discovery);formal.append(details);summary.append(formal);
  return summary;
}
