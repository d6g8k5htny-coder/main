// Presentation only: reveal preserved fragment targets inside disclosures.
// Keep native link history and the reader's current keyboard focus intact.
function fragmentTarget(document,hash) {
  if(!hash||hash==='#')return null;
  try{return document.getElementById(decodeURIComponent(hash.slice(1)));}
  catch{return null;}
}
function ancestors(target) {
  const found=[];
  for(let node=target;node;node=node.parentElement)if(node.tagName==='DETAILS')found.push(node);
  return found;
}
export function prepareReaderArchives(document,hash) {
  const keep=new Set(ancestors(fragmentTarget(document,hash)));
  for(const archive of document.querySelectorAll('[data-reader-archive]'))archive.open=keep.has(archive);
}
export function revealReaderFragment(document,hash) {
  const target=fragmentTarget(document,hash);
  if(!target)return false;
  let opened=false;
  for(const disclosure of ancestors(target))if(!disclosure.open){disclosure.open=true;opened=true;}
  if(opened)target.scrollIntoView({block:'start',behavior:'instant'});
  return opened;
}
export function restoreReaderLink(document,href,event) {
  if(event.defaultPrevented||event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return false;
  const link=event.target?.closest?.('a[href]');
  if(!link||link.hasAttribute('download'))return false;
  const target=link.getAttribute('target');
  if(target&&target.toLowerCase()!=='_self')return false;
  try {
    const current=new URL(href),destination=new URL(link.getAttribute('href'),document.baseURI||href);
    if(destination.origin!==current.origin||destination.pathname!==current.pathname||destination.search!==current.search)return false;
    // A repeated fragment emits no hashchange. Open the destination before
    // native navigation; the browser still owns history, scrolling and focus.
    return revealReaderFragment(document,destination.hash);
  }catch{return false;}
}
if(typeof window!=='undefined'&&typeof document!=='undefined') {
  prepareReaderArchives(document,window.location.hash);
  revealReaderFragment(document,window.location.hash);
  window.addEventListener('hashchange',()=>revealReaderFragment(document,window.location.hash));
  document.addEventListener('click',event=>restoreReaderLink(document,window.location.href,event));
}
