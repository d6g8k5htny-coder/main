// Optional dismissal only: native details/links work without this module.
export function startReaderJump(doc,win) {
  const menu=doc.querySelector('.reader-tools > .reader-jump');
  if(!menu||menu.dataset.enhanced)return null;
  const summary=menu.querySelector(':scope > summary');
  if(!summary)return null;
  menu.dataset.enhanced='true';
  menu.addEventListener('toggle',()=>{
    if(!menu.open)return;
    // Align the existing tools strip before showing a downward menu. Its CSS
    // viewport bound then works at initial page position and in landscape.
    const top=menu.closest('.reader-tools').getBoundingClientRect().top;
    if(top>0)win.scrollBy({top,behavior:'instant'});
  });
  menu.addEventListener('click',event=>{
    if(event.target.closest('a[href]'))menu.open=false;
  });
  menu.addEventListener('keydown',event=>{
    if(event.key==='Escape'&&menu.open){
      event.preventDefault();event.stopPropagation();menu.open=false;
      summary.focus({preventScroll:true});
    }
  });
  doc.addEventListener('pointerdown',event=>{
    if(menu.open&&!menu.contains(event.target))menu.open=false;
  });
  menu.addEventListener('focusout',event=>{
    if(event.relatedTarget&&!menu.contains(event.relatedTarget))menu.open=false;
  });
  win.addEventListener('pagehide',()=>{menu.open=false;});
  return menu;
}
if(typeof document!=='undefined'&&typeof window!=='undefined')startReaderJump(document,window);
