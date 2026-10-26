// A navigation handoff to Cite's existing syntax-only source reference tool.
import {validateEntry} from './research-desk-store.mjs?site-release=547418ff3547b671266d1e46bf8d33f1f3eedf83571f6a0bb69ca9d7786ffaf3';
import {buildReference,referenceStateURL} from './source-reference.mjs?site-release=bdbf68df8b2c8459b67140d770561c01dddf325336e0e57ab18740379651c48b';

export function readingListCitationHref(value,baseURL) {
  const entry=validateEntry(value);
  if(entry.kind!=='source')throw new TypeError('Only a saved source can become a source reference.');
  const state={repository:entry.repository.split('/')[1],commit:entry.commit,path:entry.path,sha256:entry.sha256||''};
  // Cite intentionally has a narrower literal-path contract than the list.
  // Preserve the list identity and let that one row explain the refusal.
  try {buildReference(state.repository,state.commit,state.path,state.sha256);}
  catch {return null;}
  const url=new URL('cite.html',baseURL);
  url.hash='reference-builder';
  return referenceStateURL(url.href,state);
}
