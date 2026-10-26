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
if(typeof window!=='undefined'&&typeof document!=='undefined') {
  prepareReaderArchives(document,window.location.hash);
  revealReaderFragment(document,window.location.hash);
  window.addEventListener('hashchange',()=>revealReaderFragment(document,window.location.hash));
}
