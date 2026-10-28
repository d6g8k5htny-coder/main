import {spatialProjection} from './atlas-spatial-model.mjs?site-release=fd1120b32431cde3ebcfa63bbe5e191a0502b4f1a62a6869b90c196c3677372f';

// Adapter over the admitted controller projection. It never parses a graph from
// unverified JSON and never writes scientific or controller state itself.
export function installSpatialAtlas({root,onChange}) {
  const doc=root.ownerDocument;
  // The small pure-DOM controller harness has no rendering/observer platform.
  if(!doc.defaultView?.MutationObserver)return {restore(){},decorateURL:u=>u};
  const get=id=>doc.getElementById(id),list=get('atlas-list'),stage=get('atlas-stage'),inspector=get('atlas-inspector');
  const make=(tag,text)=>{const node=doc.createElement(tag);if(text)node.textContent=text;return node;};
  const bar=make('div');bar.className='atlas-space-toolbar';
  const toggle=make('button','Explore in 3D');toggle.type='button';toggle.id='atlas-space-toggle';toggle.setAttribute('aria-pressed','false');
  const reset=make('button','Reset camera'),focus=make('button','Focus dependencies'),plus=make('button','Zoom in'),minus=make('button','Zoom out');
  for(const button of [reset,focus,plus,minus])button.type='button';focus.setAttribute('aria-pressed','false');
  bar.append(toggle,reset,focus,plus,minus);stage.before(bar);
  const section=make('section');section.className='atlas-space';section.hidden=true;section.setAttribute('aria-label','Spatial Atlas');stage.before(section);
  const status=make('p','Optional 3D uses the same displayed nodes and source snapshot.');status.id='atlas-space-status';
  const statusRegion=make('div');statusRegion.id='atlas-space-status-region';statusRegion.setAttribute('role','status');statusRegion.append(status);bar.after(statusRegion);
  const host=make('div');host.id='atlas-space-host';host.className='atlas-space-host';
  const instructions=make('p','Drag to orbit · Shift-drag or right-drag to pan · scroll or buttons to zoom. Keyboard: arrows orbit, Shift+arrows pan, +/− zoom, Home reset. Layout distance is not proof strength.');
  const legend=make('p','Arrows point from dependent to dependency. Solid: required; dashed: context. Exact recorded relationship types appear below.');
  const disclosure=make('details'),summary=make('summary','Accessible nodes and recorded relationships'),nodes=make('ul'),relations=make('ul');disclosure.open=true;disclosure.append(summary,nodes,relations);section.append(host,instructions,legend,disclosure);
  nodes.setAttribute('aria-label','Displayed nodes');
  relations.tabIndex=0;relations.setAttribute('aria-label','Recorded directed relationships, scrollable');
  for(const element of [stage,list,host])element.setAttribute('aria-describedby',((element.getAttribute('aria-describedby')||'')+' atlas-space-status').trim());
  let failureMessage='',active=false,focused=false,data=null,renderer=null,generation=0,lastSignature='',frame=0,disposed=false,intersecting=true;
  const cleanup=[];const listen=(n,t,f)=>{n.addEventListener(t,f);cleanup.push(()=>n.removeEventListener(t,f));};
  const buttons=()=>[...list.querySelectorAll('button')];
  const nodeID=button=>button.firstChild?.textContent||'';
  const selected=()=>inspector.querySelector('h3')?.textContent||'';
  const refused=()=>!get('atlas-error').hidden||!inspector.querySelector('h3');
  const shown=()=>root.getAttribute('data-atlas-panel')!=='diagnostics'&&!(matchMedia('(max-width:720px)').matches&&root.getAttribute('data-atlas-panel')!=='browse');
  function pick(id){buttons().find(button=>nodeID(button)===id)?.click();}
  function fail(message){failureMessage=message;generation++;renderer?.dispose();renderer=null;delete host.dataset.ready;delete host.dataset.frames;active=false;toggle.setAttribute('aria-pressed','false');section.hidden=true;stage.classList.remove('atlas-space-replaced');list.classList.remove('atlas-space-replaced');display();status.textContent=message+' Use Graph or List to continue.';}
  function display(){toggle.setAttribute('aria-pressed',String(active));focus.setAttribute('aria-pressed',String(focused));for(const b of [reset,focus,plus,minus])b.hidden=!active;section.hidden=!active||!shown()||refused();stage.classList.toggle('atlas-space-replaced',active&&!section.hidden);list.classList.toggle('atlas-space-replaced',active&&!section.hidden);renderer?.pause(section.hidden||!intersecting);}
  function sync(){frame=0;if(disposed)return;
    if(refused()){nodes.replaceChildren();relations.replaceChildren();lastSignature='';status.textContent='Source-bound bookmark refused; no current result count or formal evidence substituted.';if(active)fail('3D unavailable: source-bound bookmark refused. No graph substituted.');else display();return;}
    display();if(!active){const text=[failureMessage,get('atlas-count').textContent.split('. ')[0],'Optional 3D uses this same source-bound page.'].filter(Boolean).join(' ');if(status.textContent!==text)status.textContent=text;return;}if(!renderer||!data)return;
    const ids=buttons().map(nodeID),id=selected(),signature=JSON.stringify([ids,id,focused]);
    if(signature!==lastSignature){lastSignature=signature;const view=spatialProjection(data.graph.data,ids,id,focused);renderer.update(view,id);nodes.replaceChildren();relations.replaceChildren();
      for(const key of view.ids){const item=make('li'),button=make('button',key);button.type='button';button.setAttribute('aria-pressed',String(key===id));button.append(make('small',`${data.graph.data.nodes[key].kind} · ${data.graph.data.nodes[key].classification}`));button.addEventListener('click',()=>pick(key));item.append(button);nodes.append(item);}
      for(const edge of view.edges)relations.append(make('li',`${edge.from} → ${edge.to} · ${edge.required?'Required':'Context'} · ${edge.relation}`));
      if(!view.ids.length)nodes.append(make('li','No nodes match this displayed page. Clear search or change the existing filters.'));
      status.textContent=`3D · ${view.ids.length} displayed nodes · ${view.edges.length} recorded relationships · Math ${data.graph.identity.commit.slice(0,8)}. ${focused?'Direct dependencies within the displayed page; expand the neighborhood for more.':'Same filtered page as Graph and List.'}`;
    }
  }
  function schedule(){if(!frame&&!disposed)frame=requestAnimationFrame(sync);}
  async function activate(){failureMessage='';active=true;const ticket=++generation;lastSignature='';status.textContent='Loading optional 3D renderer…';display();
    try{const [{currentSnapshot},{createSpatialRenderer}]=await Promise.all([import('./current.mjs?site-release=27f990a4758c48ff2f4aa58b47a842bf61eebc3e0f7449f0dc68f63bc3c502f5'),import('./atlas-spatial.mjs?site-release=51162684675faa743673966321e285a5b533743b5e7b84005f0b4055dbdd5761')]);const snapshot=await currentSnapshot;if(ticket!==generation||disposed)return;if(!snapshot||refused())throw Error('source snapshot or bookmark did not verify');data=snapshot;renderer=createSpatialRenderer({host,onPick:pick,onFailure:fail});sync();}
    catch(error){if(ticket===generation&&!disposed)fail(`3D unavailable: ${error.message}.`);}
  }
  function deactivate(){generation++;active=false;renderer?.dispose();renderer=null;delete host.dataset.ready;delete host.dataset.frames;lastSignature='';display();status.textContent='3D paused. Graph and List retain the selected claim and source snapshot.';}
  listen(toggle,'click',()=>{if(active)deactivate();else activate();onChange();});
  listen(focus,'click',()=>{focused=!focused;sync();onChange();});listen(reset,'click',()=>renderer?.reset());listen(plus,'click',()=>renderer?.zoom(1/1.2));listen(minus,'click',()=>renderer?.zoom(1.2));
  // Registered before the controller: stop 3D without rendering or writing
  // history, then let its existing change listener commit the chosen view.
  listen(get('atlas-view'),'change',deactivate);
  listen(get('atlas-reset'),'click',()=>{focused=false;renderer?.reset();schedule();onChange();});
  const observer=new MutationObserver(schedule);observer.observe(list,{childList:true});observer.observe(inspector,{childList:true});observer.observe(root,{attributes:true,attributeFilter:['data-atlas-panel']});observer.observe(stage,{attributes:true,attributeFilter:['hidden']});
  const visibility=new IntersectionObserver(entries=>{intersecting=entries[0].isIntersecting;renderer?.pause(!intersecting||section.hidden);});visibility.observe(host);
  listen(window,'pagehide',()=>{deactivate();});listen(window,'pageshow',()=>restore(location.href));
  function restore(url){const params=new URL(url,location.href).searchParams;focused=params.get('atlas-space-focus')==='dependencies';const wants=params.get('atlas-space')==='3d';if(wants&&!active)activate();else if(!wants&&active)deactivate();else schedule();}
  function decorateURL(url){if(active)url.searchParams.set('atlas-space','3d');else url.searchParams.delete('atlas-space');if(active&&focused)url.searchParams.set('atlas-space-focus','dependencies');else url.searchParams.delete('atlas-space-focus');return url;}
  import('./current.mjs?site-release=27f990a4758c48ff2f4aa58b47a842bf61eebc3e0f7449f0dc68f63bc3c502f5').then(m=>m.currentSnapshot).then(snapshot=>{if(!disposed){data=snapshot;schedule();}});
  display();restore(location.href);
  return {restore,decorateURL,dispose(){disposed=true;deactivate();cancelAnimationFrame(frame);observer.disconnect();visibility.disconnect();cleanup.forEach(fn=>fn());}};
}
