// URL state is a reader convenience, never a source or scientific-status input.
// URLSearchParams.get deliberately uses the first occurrence of each filter.
export function readCatalogQuery(search) {
  const params=new URLSearchParams(search);
  const repository=params.get('repository');
  return {query:(params.get('q')||'').trim(),repository:['main','Math-'].includes(repository)?repository:'',path:(params.get('path')||'').trim()};
}
export function catalogQueryURL(href,filters,share=false) {
  const url=new URL(href);
  if(share){url.search='';url.hash='inventory';}
  for(const [key,value] of [['q',filters.query],['repository',filters.repository],['path',filters.path]]) {
    if(value.trim())url.searchParams.set(key,value.trim());else url.searchParams.delete(key);
  }
  return url;
}
export function connectCatalogQuery(win,controls,onRestore) {
  const current=()=>{try{return new URL(win.location.href);}catch{return null;}};
  const values=()=>({query:controls.search.value,repository:controls.repository.value,path:controls.path.value});
  const note='Filters appear in the address, shared links, and browser history. Search links use the catalog’s pinned source records, not a live research index.';
  const link=()=>{
    const url=current();if(!url)return;
    controls.link.href=catalogQueryURL(url,values(),true).href;
    controls.link.setAttribute('aria-disabled','false');
    controls.note.textContent=note;
  };
  const restore=()=>{
    const url=current();if(!url)return;
    const filters=readCatalogQuery(url.search);
    controls.search.value=filters.query;controls.repository.value=filters.repository;controls.path.value=filters.path;
    link();
  };
  // Read at verified-catalog readiness so newer navigation wins over startup.
  restore();
  if(current())win.addEventListener('popstate',()=>{restore();onRestore();});
  return ()=>{
    link();const url=current();if(!url)return;
    const next=catalogQueryURL(url,values());if(next.href===url.href)return;
    try {win.history.replaceState(win.history.state,'',next.href);}
    catch {controls.note.textContent='The address could not be updated. Use Link to this search to preserve these filters. Shared links include the filters and use the catalog’s pinned source records.';}
  };
}
