// Restore an initial section link once source loading stops changing the layout.
// Reader interaction cancels this correction; later navigation stays native.
export function prepareWorkspaceFragment({window=globalThis.window,document=globalThis.document}={}) {
  const hash=window?.location?.hash;
  let id;try{id=decodeURIComponent((hash||'').replace(/^#/,''));}catch{return ()=>false;}
  if(!['board','coefficient','inventory','contribute'].includes(id)||!window?.addEventListener)return ()=>false;
  const events=['wheel','touchstart','touchmove','keydown','pointerdown','pointermove','focusin','hashchange','popstate','pagehide'];
  const options={capture:true,passive:true};
  let cancelled=false,finished=false;
  const cleanup=()=>events.forEach(event=>window.removeEventListener(event,cancel,options));
  const cancel=event=>{
    if(event.type==='pointermove'&&!event.buttons)return;
    cancelled=true;cleanup();
  };
  events.forEach(event=>window.addEventListener(event,cancel,options));
  return ()=>{
    if(finished)return false;
    finished=true;
    // Let the final source DOM updates reach the rendering boundary first.
    // Keep cancellation active until then so readers retain control.
    window.requestAnimationFrame(()=>{
      cleanup();
      if(cancelled||window.location.hash!==hash)return;
      const target=document.getElementById(id);
      if(!target?.scrollIntoView)return;
      target.tabIndex=-1;
      target.focus?.({preventScroll:true});
      target.scrollIntoView({block:'start',behavior:'instant'});
    });
    return true;
  };
}
