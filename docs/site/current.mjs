import {fetchVerifiedSnapshot,freshness,filterSources,sourceURL,projectCurrentAcquisitions,ACQUISITION_LABELS,ACQUISITION_UNAVAILABLE} from './current-model.mjs?site-release=a30d68465e3acda48eedce64854afebc7af5c84903a4e6b0c649e08007b08999';
export function element(tag,text,className) {
  const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;
}
export function anchor(text,href) {const a=element('a',text);a.href=href;return a;}
export function renderCurrentSource(source,row) {
  const li=element('li'),url=sourceURL(source),name=`${source.repository.split('/')[1]} · ${source.path}`;
  li.append(url?anchor(name,url):element('span',name));
  const categories=row?.categories||['unavailable'];
  li.append(element('small','Acquisition evidence: '+categories.map(c=>ACQUISITION_LABELS[c]+(c==='excerpts_included'?` (${row.excerpts.length})`:'')).join(' · '),'acquisition-label'));
  if(row&&(row.excerpts.length||row.whole_files.length)){
    const details=element('details'),list=element('ul');details.append(element('summary','Acquisition identities'));
    for(const file of row.whole_files)details.append(element('p',`Git blob ${source.blob} · ${source.bytes} bytes · producer-recorded complete-file SHA-256 ${file.sha256}. The viewer has not fetched or rehashed the complete source document.`));
    for(const excerpt of row.excerpts){
      const item=element('li'),{line_start:first,line_end:last}=excerpt,fragment=first===last?`#L${first}`:`#L${first}-L${last}`,label=`${excerpt.id} · lines ${first}–${last}`;
      item.append(url?anchor(label,url+fragment):element('span',label),' · excerpt SHA-256 ',element('code',excerpt.sha256));list.append(item);
    }
    if(row.excerpts.length)details.append(list);li.append(details);
  }
  return li;
}
async function load() {
  const section=document.getElementById('current-research');if(!section)return null;
  const status=section.querySelector('.current-state');
  try {
    const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),15000);
    let data;
    try {data=await fetchVerifiedSnapshot(fetch,{sha256:section.dataset.currentDigest,bytes:section.dataset.currentBytes},{signal:controller.signal});}finally{clearTimeout(timer);}
    const acquisition=await projectCurrentAcquisitions(data),annotations=new Map(data.sources.map((source,i)=>[source,acquisition.rows[i]]));
    const notice=section.querySelector('.current-acquisition-notice');
    if(notice){notice.replaceChildren();if(acquisition.disposition==='unavailable'){const details=element('details');details.append(element('summary','Acquisition association diagnostics'),element('p',acquisition.reasons.join(', ')));notice.append(element('p',ACQUISITION_UNAVAILABLE,'inventory-unavailable'),details);}}
    const updateFreshness=()=>{const state=freshness(data.observed_at);status.textContent=`Verified source observation: ${data.observed_at} · ${state}. ${state==='stale'?'This observation is more than two hours old; check upstream sources.':''}`;status.dataset.freshness=state;};
    updateFreshness();const freshnessTimer=setInterval(updateFreshness,60000);window.addEventListener('pagehide',()=>clearInterval(freshnessTimer),{once:true});window.addEventListener('pageshow',updateFreshness);document.addEventListener('visibilitychange',updateFreshness);
    const input=section.querySelector('.current-search'),list=section.querySelector('.current-results'),more=section.querySelector('.current-more');
    if(input&&list){let limit=50;input.disabled=false;input.value=new URLSearchParams(location.search).get('source-q')||'';
      const render=()=>{const matches=filterSources(data.sources,input.value);list.replaceChildren(...matches.slice(0,limit).map(s=>renderCurrentSource(s,annotations.get(s))));
        section.querySelector('.current-count').textContent=`${Math.min(limit,matches.length)} of ${matches.length} matching files · ${data.sources.length} indexed public files. Excluded historical/recovery/operational paths are outside this discovery population.`;more.hidden=matches.length<=limit;};
      input.addEventListener('input',()=>{limit=50;render();const url=new URL(location.href);if(input.value)url.searchParams.set('source-q',input.value);else url.searchParams.delete('source-q');try{history.replaceState(null,'',url);}catch{}});
      window.addEventListener('popstate',()=>{input.value=new URLSearchParams(location.search).get('source-q')||'';limit=50;render();});
      more.addEventListener('click',()=>{limit+=50;render();});render();
    }
    return data;
  }catch(e){if(status)status.textContent=`Unavailable: ${e.message} No current result inferred.`;return null;}
}
export const currentSnapshot=typeof document==='undefined'?Promise.resolve(null):load();
