import {sourceForNode,sourceURL} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';
import {EVIDENCE} from './atlas-formal-evidence-data.mjs?site-release=1ef5a9f4aa04b860ad8debf6b1981d0485b7bc1d8f0d970095e29ac2fe895d73';

const canonical=value=>JSON.stringify(value&&typeof value==='object'&&!Array.isArray(value)
  ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,JSON.parse(canonical(value[key]))]))
  :value);

// This is an exact source-record connection, not a statement-equivalence test.
// No name, numerical value, neighbor, graph status or unrelated CI is evidence.
export function formalConnection(snapshot,selectedId,evidence=EVIDENCE) {
  try {
    if(evidence?.schema_version!==1||selectedId!==evidence.claim.id)return null;
    const node=snapshot.graph.data.nodes[selectedId],doc=evidence.document,formal=evidence.formal;
    if(canonical(node)!==canonical(evidence.claim.record)||node.fingerprint!==doc.sha256
      ||doc.manifest_source.sha256!==doc.sha256||snapshot.graph.identity.repository!==doc.repository)return null;
    const source=sourceForNode(snapshot,node);
    if(!source||source.repository!==doc.repository||source.path!==doc.path
      ||source.blob!==doc.blob||source.bytes!==doc.bytes)return null;
    const manifests=snapshot.sources.filter(s=>s.repository===formal.repository&&s.path===formal.manifest_path);
    if(manifests.length!==1||manifests[0].blob!==formal.manifest_blob
      ||manifests[0].bytes!==formal.manifest_bytes)return null;
    const execution=evidence.execution;
    if(execution.checked_commit!==formal.commit||execution.repository!==formal.repository
      ||execution.formalization_status!=='kernel-checked'||evidence.scientific_effect!=='NONE')return null;
    const review=evidence.alignment;
    if(!review||review.record.disposition!==EVIDENCE.alignment.record.disposition
      ||review.record.manifest_sha256!==formal.manifest_sha256
      ||review.record.scope_sha256!==formal.scope_sha256
      ||review.record_source.sha256!==EVIDENCE.alignment.record_source.sha256)return null;
    if(!sourceURL(source)||!sourceURL({...formal,path:evidence.declaration.module}))return null;
    return {...evidence,current_source:source,current_graph:snapshot.graph.identity};
  } catch {
    return null;
  }
}
