// Presentation over the one already-admitted observation, with no new fetch.
import {searchEquations,sourceURL} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';

export function libraryCollection(search) {
  return new URLSearchParams(search).get('library-collection')==='text'?'text':'files';
}
export function libraryDiscoveryHref(href,collection) {
  const url=new URL(href);
  if(collection==='text')url.searchParams.set('library-collection','text');
  else url.searchParams.delete('library-collection');
  return url.href;
}
export function excerptSourceHref(excerpt) {
  const url=sourceURL(excerpt),first=excerpt?.line_start,last=excerpt?.line_end;
  if(!url||!Number.isSafeInteger(first)||!Number.isSafeInteger(last)||first<1||last<first)return null;
  return url+(first===last?`#L${first}`:`#L${first}-L${last}`);
}

export function startLibraryDiscovery(doc,win,data,refreshLink=()=>{},options={}) {
  if(!doc.getElementById('research-desk')||doc.getElementById('library-collection')||!Array.isArray(data?.equations))return null;
  const section=doc.getElementById('current-research'),input=section?.querySelector('.current-search');
  const search=input?.closest('details'),label=input?.closest('label');
  const provenance=section?.querySelector(':scope > .current-source-details');
  if(!input||input.disabled||!search||!label||!provenance)return null;
  const make=(tag,text)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const ownedFocus=doc.activeElement===input;
  // Move the exact admitted controls, preserving their listeners and identities.
  section.insertBefore(search,provenance);search.classList.add('library-current-search');search.open=true;
  search.querySelector(':scope > summary').textContent='Search current material';
  const controls=make('div');controls.className='library-search-controls';
  const collectionLabel=make('label','Collection'),select=make('select');select.id='library-collection';
  for(const [value,text]of [['files','File paths'],['text','Acquired text']]){
    const option=make('option',text);option.value=value;select.append(option);
  }
  collectionLabel.append(select);label.before(controls);controls.append(collectionLabel,label);
  const queryLabel=make('span');label.replaceChildren(queryLabel,input);
  const note=make('p');note.id='library-collection-note';controls.after(note);
  const described=new Set((input.getAttribute('aria-describedby')||'').split(/\s+/).filter(Boolean));
  described.add(note.id);input.setAttribute('aria-describedby',[...described].join(' '));
  const files=make('div');files.id='library-file-results';
  const fileCount=section.querySelector('.current-count');fileCount.before(files);
  for(const node of [doc.getElementById('library-source-empty'),fileCount,section.querySelector('.current-results'),section.querySelector('.current-more')])if(node)files.append(node);
  const textPanel=make('div');textPanel.id='library-text-panel';
  const count=make('p');count.id='library-text-count';count.setAttribute('role','status');count.setAttribute('aria-atomic','true');
  const list=make('ul');list.id='library-text-results';list.setAttribute('aria-describedby',note.id);
  const more=make('button','Show more excerpts');more.id='library-text-more';more.type='button';
  textPanel.append(count,list,more);files.after(textPanel);
  const save=doc.getElementById('desk-save-current');
  let renderedQuery=null,renderedCount=0;
  function renderText(append=false) {
    if(append&&input.value!==renderedQuery)append=false;
    const offset=append?renderedCount:0;
    const found=searchEquations(data.equations,input.value,{limit:10,offset});
    const rows=found.equations.map(excerpt=>{
      const row=make('li'),href=excerptSourceHref(excerpt);
      const name=`${excerpt.repository.split('/')[1]} · ${excerpt.path} · lines ${excerpt.line_start}–${excerpt.line_end}`;
      const link=make(href?'a':'span',name);if(href)link.href=href;row.append(link);
      row.append(make('small','Acquired excerpt preview · not the complete proof text'));
      const preview=make('pre',excerpt.text.length>280?excerpt.text.slice(0,280)+'…':excerpt.text);row.append(preview);
      const identity=make('details');identity.append(make('summary','Excerpt identity'));
      identity.append(make('p','Commit: '+excerpt.commit),make('p','Excerpt SHA-256: '+excerpt.sha256));row.append(identity);
      options.decorateSource?.(row,excerpt);
      return row;
    });
    if(append)list.append(...rows);else list.replaceChildren(...rows);
    options.onResults?.();
    renderedQuery=input.value;renderedCount=offset+found.equations.length;
    count.textContent=`${found.matches} matching excerpts of ${found.total} acquired excerpts · ${renderedCount} shown. ${found.matches?'':'No acquired excerpt matches this query.'}`;
    more.hidden=renderedCount>=found.matches;
    return rows[0];
  }
  function apply() {
    const text=select.value==='text',focused=doc.activeElement;
    const hiding=text?files:textPanel;
    if(hiding.contains(focused)||text&&focused===save)select.focus({preventScroll:true});
    files.hidden=text;textPanel.hidden=!text;if(save)save.hidden=text;
    queryLabel.textContent=text?'Words or source path':'Source path or repository';
    input.placeholder=text?'Search the included mathematical text':'Try a proof name or source path';
    note.textContent=text?
      'Search covers only the acquired excerpts and their source paths. It does not search complete documents or infer connections to graph records.':
      'Search matches repository and file paths. Choose Acquired text to find words inside the included mathematical excerpts.';
    if(text)renderText();
  }
  select.value=libraryCollection(win.location.search);apply();
  if(ownedFocus)input.focus({preventScroll:true});
  select.addEventListener('change',()=>{
    try{win.history.replaceState(null,'',libraryDiscoveryHref(win.location.href,select.value));}catch{}
    apply();refreshLink();
  });
  input.addEventListener('input',()=>{if(select.value==='text')renderText();});
  more.addEventListener('click',()=>{
    const owned=doc.activeElement===more;const first=renderText(true);
    if(owned&&first){const link=first.querySelector('a');if(link)link.focus();else{first.tabIndex=-1;first.focus();}}
  });
  // Window-target popstate listeners run in registration order. Retain actual
  // result focus before the source controller can remove that focused row.
  let historyFocus=null;
  const ownsResult=node=>files.contains(node)||textPanel.contains(node)||node===save;
  doc.addEventListener('focusin',event=>{historyFocus=ownsResult(event.target)?event.target:null;});
  doc.addEventListener('pointerdown',event=>{if(!ownsResult(event.target))historyFocus=null;},{capture:true});
  win.addEventListener('popstate',()=>{
    const previous=historyFocus;historyFocus=null;
    select.value=libraryCollection(win.location.search);apply();refreshLink();
    if(previous&&(doc.activeElement===doc.body||doc.activeElement===previous)&&
      (!previous.isConnected||!previous.getClientRects().length))select.focus({preventScroll:true});
  });
  return {select};
}
