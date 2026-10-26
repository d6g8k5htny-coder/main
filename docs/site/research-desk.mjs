// Reader-owned navigation notes. Never a source, proof or status authority.
import {currentSnapshot} from './current.mjs?site-release=fc7e8a98c1e23f476962c4c83e95ed3ce817490007987112d9b2cf6d3d90b142';
import {SCHEMA,MAX_FILE_BYTES,validateEntry,entryKey,parseReadingList,mergeEntries,sourceHref,searchHref,readingListJSON} from './research-desk-store.mjs?site-release=547418ff3547b671266d1e46bf8d33f1f3eedf83571f6a0bb69ca9d7786ffaf3';
import {readingListCitationHref} from './reading-list-citation.mjs?site-release=02153e8e3acb70abca98685f5b274c0472eea6450ed2e9c3fc3125a7b0196968';

export function sourceLinkKey(href) {
  try{
    const url=new URL(href);
    if(url.origin!=='https://github.com'||url.search||url.hash||url.username||url.password)return null;
    // Preserve segment boundaries: an encoded slash is not a path separator.
    return JSON.stringify([url.origin,...url.pathname.split('/').map(decodeURIComponent)]);
  }catch{return null;}
}

export function startResearchDesk(doc=document,win=window,snapshot=currentSnapshot) {
  const root=doc.getElementById('research-desk');if(!root)return null;
  const byId=id=>doc.getElementById(id),make=(tag,text,cls)=>{
    const element=doc.createElement(tag);if(text!==undefined)element.textContent=text;if(cls)element.className=cls;return element;
  };
  const list=byId('desk-entries'),reading=byId('reading-list'),message=byId('desk-message');
  byId('desk-import').disabled=false;
  const currentInput=doc.querySelector('#current-research .current-search');
  const sourceResults=doc.querySelector('#current-research .current-results');
  let entries=[],storage=null,persistent=false,storedProblem='',currentData=null,queue=Promise.resolve();
  const mode=()=>persistent?'Saved in this browser.':'Kept in this tab only; export to keep a copy.';
  const say=text=>{message.textContent=text+' '+mode();};
  const stored=()=>{const raw=storage.getItem(SCHEMA);return raw===null?[]:parseReadingList(raw).entries;};
  try {
    storage=win.localStorage;
    entries=stored();
    persistent=typeof win.navigator.locks?.request==='function';
    if(!persistent)storedProblem='Browser storage coordination is unavailable. ';
  }catch {storedProblem='Stored reading data is unavailable or unreadable; existing data is preserved. ';}
  say(storedProblem+'Only items you choose to save enter this list.');

  const currentSources=new Map(),renderedRows=new Map();
  const now=()=>new Date().toISOString();
  const sourceButtons=()=>doc.querySelectorAll('button.desk-save-source');
  function render() {
    const focused=doc.activeElement,oldRows=[...list.children];
    const focusIndex=oldRows.findIndex(row=>row.contains(focused));
    const keys=new Set(entries.map(entryKey));
    // Keep surviving controls and disclosures in place. A background source
    // update must not discard the DOM context of an unrelated saved item.
    for(const [key,saved] of renderedRows)if(!keys.has(key)){
      saved.row.remove();renderedRows.delete(key);
    }
    let cursor=list.firstElementChild;
    for(const entry of entries) {
      const key=entryKey(entry),signature=JSON.stringify(entry);
      let saved=renderedRows.get(key);
      if(!saved){
        const row=make('li'),content=make('div'),a=make('a');
        let identity=null,citation=null;
        if(entry.kind==='source') {
          a.textContent=entry.repository.split('/')[1]+' · '+entry.path;a.href=sourceHref(entry);
          content.append(make('small','Pinned source link','desk-kind'),a);
          const details=make('details');details.append(make('summary','Saved identity'));
          identity=make('dl');
          details.append(identity);content.append(details);
          const citationHref=readingListCitationHref(entry,win.location.href);
          if(citationHref){
            citation=make('a','Make a source reference','button secondary desk-cite-source');
            citation.href=citationHref;content.append(citation);
          }else content.append(make('p','Reference builder unavailable for this literal path.','desk-citation-unavailable'));
        }else {
          a.textContent=(entry.collection==='current'?'Current source search':'Dated catalog search')+': '+(entry.query||'All files');
          a.href=searchHref(entry,win.location.href);
          content.append(make('small',entry.collection==='current'?'Reopens against the next admitted observation':'Reopens the dated catalog','desk-kind'),a);
          if(entry.repository||entry.path)content.append(make('p',[entry.repository&&'Repository: '+entry.repository,entry.path&&'Path: '+entry.path].filter(Boolean).join(' · ')));
        }
        const remove=make('button','Remove','desk-remove');remove.type='button';remove.setAttribute('aria-label','Remove '+a.textContent);
        remove.addEventListener('click',()=>mutate(items=>items.filter(item=>entryKey(item)!==key),'Item removed.'));
        row.append(content,remove);
        saved={row,identity,citation,signature:null};renderedRows.set(key,saved);
      }
      if(saved.signature!==signature){
        if(saved.identity){
          saved.identity.replaceChildren();
          for(const [label,value] of [['Commit',entry.commit],['Git blob',entry.blob],['SHA-256',entry.sha256],['Bytes',entry.bytes],['Source observation',entry.observedAt],['Saved',entry.savedAt]]){
            if(value!==undefined)saved.identity.append(make('dt',label),make('dd',String(value)));
          }
        }
        if(saved.citation)saved.citation.href=readingListCitationHref(entry,win.location.href);
        saved.signature=signature;
      }
      if(saved.row!==cursor)list.insertBefore(saved.row,cursor);
      cursor=saved.row.nextElementSibling;
    }
    byId('desk-count').textContent=entries.length+' saved item'+(entries.length===1?'':'s');
    byId('desk-empty').hidden=entries.length>0;
    byId('desk-export').disabled=!entries.length;byId('desk-clear').disabled=!entries.length;
    for(const button of sourceButtons()){
      const saved=keys.has(button.dataset.deskKey);button.disabled=saved;button.textContent=saved?'Saved':'Save source';
    }
    if(focusIndex>=0){
      if(list.contains(focused)){
        // Native node moves may blur a retained control on older browsers.
        if(doc.activeElement!==focused)focused.focus({preventScroll:true});
      }else{
        const next=focused.matches('.desk-remove')&&list.children[Math.min(focusIndex,list.children.length-1)]?.querySelector('.desk-remove');
        (next||reading.querySelector('summary')).focus({preventScroll:true});
      }
    }else if([byId('desk-clear'),byId('desk-export')].includes(focused)&&focused.disabled){
      reading.querySelector('summary').focus({preventScroll:true});
    }
  }
  function mutate(operation,success,refusal='Could not update reading list: ') {
    queue=queue.then(async()=>{
      const apply=()=>{
        let base=entries;
        if(persistent){try{base=stored();}catch{persistent=false;storedProblem='Stored list could not be read; existing data is preserved. ';}}
        const next=operation(base); // Validation/capacity failures never partially write.
        if(persistent){try{storage.setItem(SCHEMA,readingListJSON(next));}catch{persistent=false;storedProblem='Browser storage could not be updated. ';}}
        entries=next;render();say(storedProblem+success);
      };
      if(persistent){
        let ran=false;
        try{await win.navigator.locks.request(SCHEMA,()=>{ran=true;apply();});}
        catch(error){if(ran)throw error;persistent=false;storedProblem='Browser storage coordination was refused. ';apply();}
      }else apply();
    }).catch(error=>say(refusal+error.message));
    return queue;
  }
  function remember(entry){return mutate(items=>mergeEntries(items,[entry]),'Item added to your reading list.');}
  function addSourceButton(parent,entry) {
    if(parent.querySelector('.desk-save-source'))return;
    const button=make('button','Save source','desk-save-source');button.type='button';button.dataset.deskKey=entryKey(entry);
    button.setAttribute('aria-label','Save source '+entry.path);
    button.addEventListener('click',()=>remember(entry));parent.append(button);
  }
  function attachCurrent() {
    if(!currentData||!sourceResults)return;
    for(const row of sourceResults.children){const a=row.querySelector(':scope > a'),source=a&&currentSources.get(sourceLinkKey(a.href));if(source)addSourceButton(row,source);}
    render();
  }
  function attachCatalog() {
    const catalog=byId('catalog');if(!catalog)return;
    for(const row of catalog.children){
      const cells=row.querySelectorAll('td'),a=cells[0]?.querySelector('a');if(!a||cells.length!==3)continue;
      try{
        const url=new URL(a.href);if(url.origin!=='https://github.com'||url.search||url.hash||url.username||url.password)continue;
        const parts=url.pathname.split('/');if(parts[1]!=='d6g8k5htny-coder'||parts[3]!=='blob')continue;
        const entry=validateEntry({kind:'source',collection:'catalog',repository:parts[1]+'/'+parts[2],commit:parts[4],path:parts.slice(5).map(decodeURIComponent).join('/'),sha256:cells[2].textContent.trim(),savedAt:now()});
        if(sourceLinkKey(sourceHref(entry))!==sourceLinkKey(url.href))continue;
        addSourceButton(cells[0],entry);
      }catch{} // Unrecognized rows never become saved identities.
    }
    byId('desk-save-catalog').disabled=byId('search').disabled;
    render();
  }

  const summaryFor=input=>input?.closest('details')?.querySelector(':scope > summary');
  function revealCurrent(focus=false) {
    if(!currentInput)return;
    for(let element=currentInput.parentElement;element;element=element.parentElement)if(element.tagName==='DETAILS')element.open=true;
    if(focus)(currentInput.disabled?summaryFor(currentInput):currentInput)?.focus({preventScroll:true});
  }
  const shouldReveal=()=>win.location.hash==='#library-current'||new URL(win.location.href).searchParams.has('source-q');
  if(shouldReveal())revealCurrent(false);
  byId('library-current-link').addEventListener('click',()=>{
    revealCurrent(false);
    // Native fragment navigation runs after the click listener, including a
    // repeated same-hash visit. Finish focus after that default action, only
    // while the reader has not moved to another control in the meantime.
    win.requestAnimationFrame(()=>{
      if(win.location.hash==='#library-current'&&[byId('library-current'),byId('library-current-link')].includes(doc.activeElement))revealCurrent(true);
    });
  });
  win.addEventListener('hashchange',()=>{if(win.location.hash==='#library-current')revealCurrent(true);});
  win.addEventListener('popstate',()=>{if(shouldReveal())revealCurrent(false);});
  const readyObserver=new MutationObserver(()=>{
    if(currentInput.disabled)return;
    // Move focus only when it still rests on the summary selected by the reader.
    if(doc.activeElement===summaryFor(currentInput))currentInput.focus({preventScroll:true});
    readyObserver.disconnect();
  });
  if(currentInput)readyObserver.observe(currentInput,{attributes:true,attributeFilter:['disabled']});
  const currentButton=byId('desk-save-current');
  currentInput?.closest('label')?.after(currentButton);
  currentButton.addEventListener('click',()=>{
    if(!currentData)return;
    try{remember(validateEntry({kind:'search',collection:'current',query:currentInput.value,repository:'',path:'',observedAt:currentData.observed_at,savedAt:now()}));}
    catch(error){say('Search could not be saved: '+error.message);}
  });
  const catalogButton=byId('desk-save-catalog');
  byId('catalog-link')?.parentElement?.append(catalogButton);
  catalogButton.addEventListener('click',()=>{
    if(byId('search').disabled)return;
    try{remember(validateEntry({kind:'search',collection:'catalog',query:byId('search').value,repository:byId('repository-filter').value,path:byId('path-filter').value,savedAt:now()}));}
    catch(error){say('Search could not be saved: '+error.message);}
  });
  // Observe only added result rows. Our own button/status updates do not loop.
  const currentObserver=new MutationObserver(records=>{if(records.some(record=>record.target===sourceResults))attachCurrent();});
  if(sourceResults)currentObserver.observe(sourceResults,{childList:true});
  const catalogObserver=new MutationObserver(()=>attachCatalog());
  if(byId('catalog'))catalogObserver.observe(byId('catalog'),{childList:true});
  const catalogReadyObserver=new MutationObserver(()=>{byId('desk-save-catalog').disabled=byId('search').disabled;});
  if(byId('search'))catalogReadyObserver.observe(byId('search'),{attributes:true,attributeFilter:['disabled']});
  attachCatalog();
  snapshot.then(data=>{
    if(!data)return;
    currentData=data;
    for(const source of data.sources){
      try{const entry=validateEntry({kind:'source',collection:'current',repository:source.repository,commit:source.commit,path:source.path,blob:source.blob,bytes:source.bytes,observedAt:data.observed_at,savedAt:now()});currentSources.set(sourceLinkKey(sourceHref(entry)),entry);}catch{}
    }
    currentButton.disabled=false;attachCurrent();
  }).catch(()=>{});

  byId('desk-clear').addEventListener('click',()=>mutate(()=>[],'Reading list cleared.'));
  byId('desk-export').addEventListener('click',()=>{
    try{
      const blob=new Blob([readingListJSON(entries)],{type:'application/json'}),url=URL.createObjectURL(blob),a=make('a');
      a.href=url;a.download='Universal_Law_Reading_List.json';doc.body.append(a);a.click();a.remove();win.setTimeout(()=>URL.revokeObjectURL(url),1000);
      say('Reading list exported. Queries and saved source identities are included.');
    }catch(error){say('Export unavailable: '+error.message);}
  });
  byId('desk-import').addEventListener('change',async event=>{
    const input=event.currentTarget,file=input.files?.[0];if(!file)return;
    try{
      if(file.size>MAX_FILE_BYTES)throw new Error('File exceeds128KiB.');
      const incoming=parseReadingList(await file.text()).entries;
      await mutate(items=>mergeEntries(items,incoming),'Reading list imported; duplicates retained once.','Import refused: ');
    }catch(error){say('Import refused: '+error.message);}
    finally{input.value='';}
  });
  win.addEventListener('storage',event=>{
    if(!persistent||(event.key!==SCHEMA&&event.key!==null))return;
    try{entries=stored();render();say('Reading list updated in another tab.');}
    catch{persistent=false;say('Stored list changed to unreadable data; existing stored data is preserved.');}
  });
  render();return {settled:()=>queue};
}
if(typeof document!=='undefined'&&typeof window!=='undefined')startResearchDesk();
