import {element,anchor} from './current.mjs?site-release=27f990a4758c48ff2f4aa58b47a842bf61eebc3e0f7449f0dc68f63bc3c502f5';
import {sourceForNode,sourceURL} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';

import {createFormalEvidence} from './atlas-formal-evidence.mjs?site-release=51c7393f9036135ef2c813820cf8157dcc5025a2877f001b79f6d82f68894a84';

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
