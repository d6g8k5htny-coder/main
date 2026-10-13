import {parseVerifiedSnapshot,freshness,filterSources,sourceURL} from './current-model.mjs?site-release=93fcc09ee90ea25c2ed68c5b59f063fa89d6543631ff98d686bd911b9873bc6b';
export function element(tag,text,className) {
  const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;
}
export function anchor(text,href) {const a=element('a',text);a.href=href;return a;}
async function load() {
  const section=document.getElementById('current-research');if(!section)return null;
  const status=section.querySelector('.current-state');
  try {
    const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),15000);
    let response;
    try {response=await fetch('current.json',{cache:'no-store',credentials:'omit',signal:controller.signal});}finally{clearTimeout(timer);}
    if(!response.ok)throw Error('Current source snapshot unavailable.');
    const data=await parseVerifiedSnapshot(await response.arrayBuffer(),{sha256:section.dataset.currentDigest,bytes:section.dataset.currentBytes});
    const updateFreshness=()=>{const state=freshness(data.observed_at);status.textContent=`Verified source observation: ${data.observed_at} · ${state}. ${state==='stale'?'This observation is more than two hours old; check upstream sources.':''}`;status.dataset.freshness=state;};
    updateFreshness();const freshnessTimer=setInterval(updateFreshness,60000);window.addEventListener('pagehide',()=>clearInterval(freshnessTimer),{once:true});window.addEventListener('pageshow',updateFreshness);document.addEventListener('visibilitychange',updateFreshness);
    const input=section.querySelector('.current-search'),list=section.querySelector('.current-results'),more=section.querySelector('.current-more');
    if(input&&list){let limit=50;input.disabled=false;input.value=new URLSearchParams(location.search).get('source-q')||'';
      const render=()=>{const matches=filterSources(data.sources,input.value);list.replaceChildren(...matches.slice(0,limit).map(s=>{const li=element('li');li.append(anchor(`${s.repository.split('/')[1]} · ${s.path}`,sourceURL(s)));return li;}));
        section.querySelector('.current-count').textContent=`${Math.min(limit,matches.length)} of ${matches.length} matching files · ${data.sources.length} indexed public files. Excluded historical/recovery/operational paths are outside this discovery population.`;more.hidden=matches.length<=limit;};
      input.addEventListener('input',()=>{limit=50;render();const url=new URL(location.href);if(input.value)url.searchParams.set('source-q',input.value);else url.searchParams.delete('source-q');try{history.replaceState(null,'',url);}catch{}});
      window.addEventListener('popstate',()=>{input.value=new URLSearchParams(location.search).get('source-q')||'';limit=50;render();});
      more.addEventListener('click',()=>{limit+=50;render();});render();
    }
    return data;
  }catch(e){if(status)status.textContent=`Unavailable: ${e.message} No current result inferred.`;return null;}
}
export const currentSnapshot=typeof document==='undefined'?Promise.resolve(null):load();
