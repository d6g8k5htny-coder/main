// Library presentation only. Filtering and admission remain in current.mjs.
export function librarySearchHref(href,query) {
  const url=new URL(href);
  if(query)url.searchParams.set('source-q',query);else url.searchParams.delete('source-q');
  url.hash='library-current';
  return url.href;
}

export function startLibrarySourceSearch(doc,win) {
  if(!doc.getElementById('research-desk')||doc.getElementById('library-source-tools'))return null;
  const input=doc.querySelector('#current-research .current-search');
  if(!input||input.disabled)return null;
  const list=doc.querySelector('#current-research .current-results'),label=input.closest('label');
  if(!list||!label)return null;
  const make=(tag,id,text)=>{
    const node=doc.createElement(tag);node.id=id;if(text)node.textContent=text;return node;
  };
  const tools=make('div','library-source-tools');tools.className='library-source-tools';
  const clear=make('button','library-source-clear','Clear search');clear.type='button';
  const link=make('a','library-source-link','Link to this search');link.className='button secondary';
  tools.append(clear,link);
  // Reuse the existing save action and its identity; it is still disabled
  // until the verified snapshot's caller finishes initialization.
  const save=doc.getElementById('desk-save-current');if(save)tools.append(save);
  const note=make('p','library-source-query-note','This query reopens against the admitted source observation available then; results are not frozen.');
  const empty=make('p','library-source-empty');empty.setAttribute('role','status');empty.hidden=true;
  label.after(tools,note,empty);
  const described=new Set((input.getAttribute('aria-describedby')||'').split(/\s+/).filter(Boolean));
  described.add(note.id);input.setAttribute('aria-describedby',[...described].join(' '));
  function update() {
    clear.disabled=input.value.length===0;
    link.href=librarySearchHref(win.location.href,input.value);
    const missing=list.children.length===0;
    empty.hidden=!missing;
    const text=missing?'No source paths match this search. Try a shorter path or use Clear search. The dated catalog is a separate collection.':'';
    if(empty.textContent!==text)empty.textContent=text;
  }
  clear.addEventListener('click',()=>{
    input.value='';
    input.dispatchEvent(new win.Event('input',{bubbles:true}));
    input.focus({preventScroll:true});
  });
  input.addEventListener('input',update);
  win.addEventListener('popstate',update);
  // Fragment navigation can preserve other query state; keep the native link
  // aligned without changing the URL, history, results or keyboard focus.
  win.addEventListener('hashchange',update);
  const observer=new win.MutationObserver(update);observer.observe(list,{childList:true});
  update();
  return {update};
}
