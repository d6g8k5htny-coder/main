import {element,anchor} from './current.mjs?site-release=f739c9840cf6ee25ce9d3ca809e030f6652751fc6b9b21d3c5d3f398dbf82438';
import {sourceForNode,sourceURL} from './current-model.mjs?site-release=4852e23a5fe1f95b7c7ec91ff3f9472e2c614fe962f7c328cc57bcf36aba950e';

import {createFormalEvidence} from './atlas-formal-evidence.mjs?site-release=aef53faa1be1af618d80d17f75bbc320560f6a03bbc3060aacd4db99f898b2c5';

// Presentation over the controller's admitted record. No acquisition, source
// adoption, proof coverage inference, or scientific-status authority lives here.
export function createClaimSummary(data,node,selectedId) {
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

  summary.append(createFormalEvidence(data,node,selectedId));
  return summary;
}
