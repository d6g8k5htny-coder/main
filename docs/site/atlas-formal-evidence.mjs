import {element,anchor} from './current.mjs?site-release=27f990a4758c48ff2f4aa58b47a842bf61eebc3e0f7449f0dc68f63bc3c502f5';
import {sourceURL} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';
import {formalConnection} from './atlas-formal-evidence-model.mjs?site-release=36dbb1f470e2c49e660275ff8090f74e56489b4cda5d0ccfd7b25b67e276f348';

const receiptURL=new URL('./evidence/side24-main-3b39/receipt.json',import.meta.url).href;
const typesURL=new URL('./evidence/side24-main-3b39/elaborated-types.log',import.meta.url).href;
const archiveURL=new URL('./evidence/side24-main-3b39/verified-execution.zip',import.meta.url).href;

export function createFormalEvidence(data,node,selectedId) {
  const connection=formalConnection(data,selectedId);
  const formal=element('section');formal.id='atlas-formal-evidence';
  formal.append(element('h4','Formal evidence for this claim'));
  formal.append(element('p',connection
    ?'Source-verified arithmetic companion only. It does not prove the coefficient or parent theorem. Package alignment review: REJECTED (AMEND).'
    :'No verified graph-to-formal-artifact mapping is recorded in this source projection. Missing evidence remains unknown.'));
  const details=element('details');details.id='atlas-formal-details';
  details.append(element('summary','Inspect declaration, execution and alignment'));
  const fields=element('dl'),execution=connection?.execution;
  const rows=[
    ['Acquisition',connection
      ?'Offline comparison verified that this exact proof document is the source pinned by the arithmetic manifest. This graph view does not re-fetch the proof document.'
      :'Graph metadata and pinned file inventory; proof document bytes are not acquired by this graph view.'],
    ['Mathematical review','Recorded review fields retain their original scope; no formal alignment or scientific acceptance follows.'],
    ['Declaration and hypotheses',connection
      ?'One closed Nat arithmetic declaration, shown above. It has no explicit hypothesis binders; the graph claim’s mathematical hypotheses remain those of its source.'
      :'No mapped declaration or hypothesis record. Read the exact mathematical source and its scope.'],
    [connection?'Graph-source revision':'Graph-source revision (formal artifact unmapped)',
      data.graph.identity.repository+' '+data.graph.identity.commit+' · '+data.graph.identity.path],
    ['Checked execution',connection
      ?execution.formalization_status+' at '+execution.checked_commit+'; run '+execution.run_id+', attempt '+execution.run_attempt+'. Original receipt '+execution.receipt_sha256+'. This records the named arithmetic declaration, not proof of the coefficient. A green CI run alone does not establish coverage.'
      :'No mapped trusted kernel-execution receipt. A green CI run alone does not establish coverage.'],
    ['Alignment',connection
      ?connection.formal.alignment_status+' in the manifest. Submitted independent review: '+connection.alignment.record.disposition+' (AMEND). Selected declaration row: '+connection.alignment.selected_target_verdict+'. The complete 31-target package has no accepted alignment: two endpoint anchors are MISALIGNED. The graph’s review fields are separate.'
      :'No mapped exact-source alignment review.'],
    ['Scientific acceptance','Unchanged; neither visualization nor formal execution promotes a claim.']
  ];
  for(const [name,value]of rows)fields.append(element('dt',name),element('dd',value));
  if(connection) {
    const record=element('section');record.id='atlas-formal-connection';
    const declaration=connection.declaration,pin=connection.formal;
    record.append(element('h5','Exact arithmetic declaration'),element('p',declaration.name));
    const statement=element('pre');statement.append(element('code',declaration.source_statement));record.append(statement);
    record.append(element('p','Informal anchor recorded by the existing manifest:'),
      element('blockquote',declaration.informal_anchor),
      element('p','Does not claim: '+declaration.does_not_claim+'. No coefficient enclosure, Kac–Rice result or parent theorem is established by this connection.'));
    const leanSource=sourceURL({...pin,path:declaration.module});
    record.append(anchor('Exact Lean source and definitions',leanSource+'#L'+declaration.line),
      element('p','Formal artifact: '+pin.repository+' · '+pin.commit.slice(0,8)+'. Complete source identity is available below.'));
    const expanded=element('details');expanded.id='atlas-formal-type-details';
    expanded.append(element('summary','Original elaborated type'));
    const type=element('pre');type.append(element('code',declaration.elaborated_type));expanded.append(type);
    record.append(expanded);
    const links=element('p');links.append(anchor('Original execution receipt (JSON)',receiptURL),' · ',
      anchor('Original type capture',typesURL),' · ',anchor('Original execution artifact (ZIP)',archiveURL));
    record.append(links,element('p','Transitive axioms for this declaration: '+(execution.target_axioms.length?execution.target_axioms.join(', '):'none reported in the original audit')+'.'));
    const source=connection.document,provenance=element('details');provenance.id='atlas-formal-source-match';
    provenance.append(element('summary','Inspect the source-identity connection'),
      element('p','Claim '+connection.claim.id+'; complete recorded fields must match. Proof SHA-256 '+source.sha256+'; Git blob '+source.blob+'; '+source.bytes+' bytes.'),
      element('p','Identity was assembled using '+source.repository+' '+source.commit+' · '+source.path+'. The current graph inventory has the identical recorded source blob; its own source link and revision remain unchanged.'),
      element('p','Formal artifact source revision '+pin.repository+' '+pin.commit+'. Manifest SHA-256 '+pin.manifest_sha256+'. Scope SHA-256 '+pin.scope_sha256+'.'),
      element('p','Lean: '+execution.lean_version+'.'),
      anchor('Pinned manifest source document',sourceURL(source.manifest_source)),
      element('p','Matching bytes connect the source references. They do not establish semantic alignment or transfer the arithmetic proof to the whole research claim.'));
    record.append(provenance);
    const review=connection.alignment,reviewLinks=element('p');
    reviewLinks.append(anchor('Submitted independent review: REJECTED (AMEND)',sourceURL(review.record.evidence)),' · ',
      anchor('Exact review record (JSON)',sourceURL(review.record_source)));
    record.append(reviewLinks,element('p','Reviewer: '+review.record.reviewer.provider+' / '+review.record.reviewer.agent+'. Its ALIGNED row for this declaration does not override the REJECTED package disposition or confer scientific acceptance.'));
    details.append(record);
  }
  details.append(fields);
  const discovery=element('details');discovery.id='atlas-formal-packages';
  discovery.append(element('summary',connection?'Package source inventory (no additional coverage)':'Explore formal packages separately (unmapped)'),
    element('p','These are source references in the pinned inventory. Open the manifests for declarations, scope and dependency pins; inspect trusted execution and alignment separately.'));
  for(const item of data.sources.filter(s=>['formal/manifest.json','formal/SCOPE.md','formal/ALIGNMENT.md'].includes(s.path))){
    const paragraph=element('p');paragraph.append(anchor(item.repository+' · '+item.path+' · '+item.commit.slice(0,8),sourceURL(item)));discovery.append(paragraph);
  }
  details.append(discovery);formal.append(details);return formal;
}
