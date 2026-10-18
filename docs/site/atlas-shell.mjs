// Presentation only: this module never fetches, admits or classifies source data.
export function viewForViewport(search,parsedView,narrow){
  const requested=new URLSearchParams(search).get('atlas-view');
  return requested==='graph'||requested==='list'?requested:narrow?'list':'graph';
}

export function createAtlasShell({root,inspector,diagnostics,media,location,history,onChange}){
  const doc=root.ownerDocument,layout=inspector.parentElement,browse=layout.children[0];
  const make=(tag,id,text)=>{const node=doc.createElement(tag);if(id)node.id=id;if(text)node.textContent=text;return node;};
  const reader=make('section','atlas-reader');reader.setAttribute('class','atlas-reader');reader.setAttribute('aria-label','Selected node reader');
  const navigation=make('nav');navigation.setAttribute('class','atlas-reader-navigation');navigation.setAttribute('aria-label','Reader navigation');
  const back=make('button','atlas-reader-back','Back'),forward=make('button','atlas-reader-forward','Forward'),browseLink=make('a','atlas-browse-link','Return to atlas browse');
  for(const button of [back,forward])button.type='button';
  back.addEventListener('click',()=>history.back());forward.addEventListener('click',()=>history.forward());navigation.append(back,forward,browseLink);
  const tabs=make('div','atlas-inspector-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Selected node information');
  const names=['evidence','diagnostics'],buttons=[],panels=[];
  for(const name of names){
    const tab=make('button','atlas-tab-'+name,name==='evidence'?'Evidence':'Diagnostics');tab.type='button';tab.setAttribute('role','tab');tab.setAttribute('aria-controls','atlas-panel-'+name);
    const panel=make('div','atlas-panel-'+name);panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);panel.tabIndex=0;
    buttons.push(tab);panels.push(panel);tabs.append(tab);
    tab.addEventListener('click',()=>activate(name));
    tab.addEventListener('keydown',event=>{
      const index=buttons.indexOf(tab);let next;
      if(event.key==='ArrowRight')next=(index+1)%names.length;
      else if(event.key==='ArrowLeft')next=(index+names.length-1)%names.length;
      else if(event.key==='Home')next=0;
      else if(event.key==='End')next=names.length-1;
      else return;
      event.preventDefault();buttons[next].focus();
    });
  }
  layout.insertBefore(reader,inspector);panels[0].append(inspector);panels[1].append(diagnostics);reader.append(navigation,tabs,...panels);
  browse.setAttribute('data-atlas-browse','');
  const open=make('button','atlas-open-reader','Read selected node');open.type='button';browse.append(open);
  let mode='browse';
  function apply(){
    root.setAttribute('data-atlas-panel',mode);
    const selected=mode==='diagnostics'?1:0;
    for(let i=0;i<buttons.length;i++){buttons[i].setAttribute('aria-selected',String(i===selected));buttons[i].tabIndex=i===selected?0:-1;panels[i].hidden=i!==selected;}
  }
  function focusReading(){(mode==='diagnostics'?panels[1]:inspector).focus({preventScroll:true});if(media.matches)reader.scrollIntoView({block:'start',behavior:'auto'});}
  function focusBrowse(){open.focus({preventScroll:true});open.scrollIntoView({block:'center',inline:'nearest',behavior:'auto'});}
  function activate(next){if(mode===next)return;mode=next;apply();onChange();}
  function selectNode(){mode='evidence';apply();}
  function restore(url,{focus=false}={}){
    const value=new URL(url,location.href).searchParams.get('atlas-panel');mode=names.includes(value)?value:'browse';apply();
    if(focus){if(media.matches&&mode==='browse')focusBrowse();else focusReading();}
  }
  function decorateURL(url){if(mode==='browse')url.searchParams.delete('atlas-panel');else url.searchParams.set('atlas-panel',mode);return url;}
  function updateLinks(url){const target=new URL(url,location.href);target.searchParams.delete('atlas-panel');browseLink.href=target.href;}
  open.addEventListener('click',()=>{activate('evidence');focusReading();});
  browseLink.addEventListener('click',event=>{
    if(event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    event.preventDefault();activate('browse');if(media.matches)focusBrowse();else buttons[0].focus();
  });
  // A layout read can hide a mobile control and blur it before media.change.
  // Remember only that hidden-target blur; ordinary or outside focus wins.
  let blurredByLayout=null;
  root.addEventListener('focusout',event=>{blurredByLayout=!event.relatedTarget&&!event.target.getClientRects().length?event.target:null;});
  root.addEventListener('focusin',()=>{blurredByLayout=null;});
  media.addEventListener('change',()=>{
    const focused=doc.activeElement===doc.body&&blurredByLayout?blurredByLayout:doc.activeElement,wasReader=reader.contains(focused),wasBrowse=browse.contains(focused);blurredByLayout=null;
    onChange({write:false});
    const unavailable=!root.contains(focused)||!focused?.getClientRects().length;
    // The atlas renderer replaces source and node children. Repair after it
    // finishes so focus cannot remain detached or in a newly hidden region.
    if(media.matches){
      if(mode==='browse'&&(wasReader||wasBrowse&&unavailable))focusBrowse();
      else if(mode!=='browse'&&(wasBrowse||wasReader&&unavailable))focusReading();
    }else if(navigation.contains(focused)||focused===open||wasBrowse&&unavailable)buttons[mode==='diagnostics'?1:0].focus({preventScroll:true});
    else if(wasReader&&unavailable)focusReading();
  });
  restore(location.href);updateLinks(location.href);
  return {selectNode,focusReading,restore,decorateURL,updateLinks,reset(){mode='browse';apply();}};
}
