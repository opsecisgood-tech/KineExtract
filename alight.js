// vid2anim → Alight Motion exporter (.xml project). Default look: white stickman, thin black edge, round head.
// Pure browser, no dependencies. Wired up from app.js with initAlight(ctx).

/* ───────── helpers ───────── */
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const f6=v=>(Number.isFinite(v)?v:0).toFixed(6);
const num=v=>{const s=(+v).toFixed(8);return s.replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'.0')};
const AM_FPS=[12,15,18,20,24,25,30,48,50,60];
const even=n=>Math.max(2,Math.round(n/2)*2);
const LINEAR=[0,0,1,1];
const isLin=e=>Array.isArray(e)&&e.every((v,i)=>Math.abs(v-LINEAR[i])<1e-6);
const easeStr=e=>e&&!isLin(e)?`cubicBezier ${e.map(num).join(' ')}`:'';
const sideColor=d=>d.side==='L'?'#FF3DE8FF':d.side==='R'?'#FFFF8A4C':'#FFCFD3D9';
const thickBase=id=>id==='head'?16:/tor|pelvis|neck/.test(id)?14:9; // same proportions as the Rig preview (480 px wide)

/* ───────── key planning ─────────
   A transform channel (location = x+y, scale = len, rotation = rot) can only have ONE set of keyframes in Alight Motion.
   x and y have their own keyframes in vid2anim, so we take the union of their times. Where the original easing can't be carried over
   exactly (different curves on x and y, elastic/bounce, or a union key that cuts an eased segment in two) the segment is baked
   with one key per frame, which reproduces the curve exactly. */
function planKeys(arrs,n,dense){
  if(dense)return Array.from({length:n},(_,i)=>({i,e:null}));
  const set=new Set();for(const k of arrs)if(k)for(const q of k)set.add(q.i);
  if(!set.size)set.add(0);
  const idx=[...set].sort((a,b)=>a-b),out=[];
  for(let j=0;j<idx.length;j++){
    const a=idx[j],b=idx[j+1];
    if(b===undefined){out.push({i:a,e:null});break}
    const classes=new Set();let single=null;
    for(const k of arrs){
      if(!k||k.length<2)continue;
      let qi=-1;for(let q=0;q<k.length-1;q++)if(k[q].i<=a&&k[q+1].i>=b){qi=q;break}
      if(qi<0)continue;
      const s=k[qi],t=k[qi+1];
      if(Math.abs(t.v-s.v)<1e-9)continue;
      const exact=s.i===a&&t.i===b,e=s.e;
      if(typeof e==='string'){classes.add('S:'+e);single=e;continue}
      if(isLin(e)){classes.add('lin');continue}
      if(!exact){classes.add('partial');continue}
      classes.add('B:'+e.join(','));single=e;
    }
    if(classes.size===0||(classes.size===1&&classes.has('lin'))){out.push({i:a,e:null});continue}
    if(classes.size===1&&[...classes][0].startsWith('B:')){out.push({i:a,e:single});continue}
    if(classes.size===1&&[...classes][0]==='S:hold'){out.push({i:a,e:null});out.push({i:b-.05,e:null,holdFrom:a});continue}
    out.push({i:a,e:null});for(let f=a+1;f<b;f++)out.push({i:f,e:null}); // bake
  }
  return out;
}

/* ───────── scene builder ───────── */
export function buildScene(c,o){
  const fps=c.fps(),n=c.n(),K=c.K(),S=c.S(),D=c.D(),ids=c.act();
  if(!n||!ids.length)throw new Error('Analyze a video first.');
  const vw=c.W(),vh=c.H(),sc=o.long/Math.max(vw,vh),cw=even(vw*sc),ch=even(vh*sc);
  const totalMs=Math.max(200,Math.round(n/fps*1000)),dur=totalMs/1000;
  const projFps=o.fps==='match'?(AM_FPS.includes(fps)?fps:30):+o.fps;
  const T=i=>{const s=i/fps;return(o.tmode==='frac'?s/dur:s).toFixed(9)};
  const off=c.off(),sgn=o.inv?-1:1;
  const series=(id,p)=>{const arr=S[id]?.[p],k=K[id]?.[p];return{k:o.dense||!k||!k.length?null:k,s:arr}};
  const valAt=(id,p,i,hold)=>{const{k,s}=series(id,p);if(hold!==undefined)i=hold;if(k)return c.curve(k,i);const a=s[Math.max(0,Math.min(s.length-1,Math.round(i)))];return Number.isFinite(a)?a:0};
  // emit one keyable channel
  const channel=(tag,ids_,props,fmt)=>{
    const arrs=props.map(p=>series(ids_,p).k),keys=planKeys(arrs,n,o.dense),
      vals=keys.map(k=>props.map(p=>valAt(ids_,p,k.i,k.holdFrom))),
      cols=vals[0].map((_,ci)=>vals.map(v=>v[ci])),
      still=cols.every(col=>Math.max(...col)-Math.min(...col)<1e-7);
    if(still||keys.length<2)return `<${tag} value="${fmt(vals[0])}"/>`;
    const kf=keys.map((k,j)=>{
      const e=o.easeOn==='arrive'?(j>0?keys[j-1].e:null):(j<keys.length-1?k.e:null),es=easeStr(e);
      return `<kf t="${T(k.i)}" v="${fmt(vals[j])}"${es?` e="${es}"`:''}/>`});
    return `<${tag}>\n        ${kf.join('\n        ')}\n      </${tag}>`;
  };
  // ---- layer list, bottom to top
  const WHITE='#FFFFFFFF',BLACK='#FF000000',items=[],locFmt=v=>`${f6(v[0]*cw)},${f6(v[1]*ch)},0.000000`,rotCh=id=>channel('rotation',id,['rot'],v=>f6(sgn*(v[0]+off)));
  if(o.style==='classic'){
    for(const id of ids){const d=D[id],thick=thickBase(id)/480*cw*(o.thick/100);
      items.push({label:d.n,sid:'.rect',color:sideColor(d),tf:[channel('location',id,['x','y'],locFmt),channel('scale',id,['len'],v=>`${f6(Math.max(.001,v[0]*ch/100))},${f6(Math.max(.001,thick/100))}`),rotCh(id)]})}
  }else{
    // Stickman: every part is a thin white bar. The thin black edge is a slightly bigger black bar underneath (so it only uses the same
    // rectangle + colour tags as the classic look). All edges go under all white bars, which makes the joints merge into one clean outline.
    const lineW=Math.max(1,.0075*o.long*(o.thick/100)),edge=Math.max(0,(Number.isFinite(+o.edge)?+o.edge:2)*o.long/1080),
      lines=ids.filter(id=>id!=='head'),pre={};
    for(const id of lines)pre[id]={loc:channel('location',id,['x','y'],locFmt),rot:rotCh(id)};
    // each bar is lengthened by one line width so neighbouring bars overlap at the joints instead of leaving notches
    const sclOf=(id,pad)=>channel('scale',id,['len'],v=>`${f6(Math.max(.001,(v[0]*ch+lineW+pad)/100))},${f6(Math.max(.001,(lineW+pad)/100))}`);
    if(edge>0)for(const id of lines)items.push({label:D[id].n+' edge',sid:'.rect',color:BLACK,tf:[pre[id].loc,sclOf(id,2*edge),pre[id].rot]});
    for(const id of lines)items.push({label:D[id].n,sid:'.rect',color:WHITE,tf:[pre[id].loc,sclOf(id,0),pre[id].rot]});
    if(ids.includes('head')){
      // round head: a circle of steady size (the ear-to-nose line changes length every time the head turns, so it is not used for the size)
      const medv=a=>{const q=(a||[]).filter(Number.isFinite).sort((x,y)=>x-y);return q.length?q[q.length>>1]:0},tl=medv(S.torso?.len),hl=medv(S.head?.len);
      let hd=tl>0?.42*tl*ch:hl>0?2.4*hl*ch:lineW*8;hd=Math.max(lineW*4,hd*((+o.headPct||100)/100));
      const loc=channel('location','head',['x','y'],locFmt),rs=esc((o.round||'.ellipse').trim()||'.ellipse'),sc=d=>`<scale value="${f6(d/100)},${f6(d/100)}"/>`;
      if(edge>0)items.push({label:'Head edge',sid:rs,color:BLACK,tf:[loc,sc(hd+2*edge)]});
      items.push({label:'Head',sid:rs,color:WHITE,tf:[loc,sc(hd)]});
    }
  }
  if(o.order==='top')items.reverse();
  const layers=items.map((it,li)=>`  <shape id="${li+1}" label="${esc(it.label)}" startTime="0" endTime="${totalMs}" fillType="color" s="${it.sid}">
    <transform>
      ${it.tf.join('\n      ')}
    </transform>
    <fillColor value="${it.color}"/>
  </shape>`).join('\n');
  const title=(o.title||'vid2anim').slice(0,60);
  const xml=`<?xml version="1.0" encoding="UTF-8"?>
<scene title="${esc(title)}" width="${cw}" height="${ch}" exportWidth="${cw}" exportHeight="${ch}" bgcolor="${o.bg}" totalTime="${totalMs}" fps="${projFps}" modifiedTime="${Date.now()}" amver="1028425" ffver="106" am="com.alightcreative.motion/5.0.273.1028425" amplatform="android">
${layers}
</scene>
`;
  return{xml,info:{layers:items.length,canvas:`${cw}×${ch}`,ms:totalMs,fps:projFps,bytes:new Blob([xml]).size}};
}

/* Calibration project: tiny scene that shows which timing / easing / rotation conventions your Alight Motion build uses. */
export function buildCalibration(o){
  const totalMs=4000,dur=4,tt=x=>(o.tmode==='frac'?x:x*4).toFixed(9),mk=(id,label,y,color,loc,extra='',shape='.rect')=>`  <shape id="${id}" label="${esc(label)}" startTime="0" endTime="${totalMs}" fillType="color" s="${esc(shape)}">
    <transform>
      ${loc}${extra}
    </transform>
    <fillColor value="${color}"/>
  </shape>`;
  const mv=(y,k)=>`<location>\n        ${k.map(([t,x,e])=>`<kf t="${t}" v="${x}.000000,${y}.000000,0.000000"${e?` e="${e}"`:''}/>`).join('\n        ')}\n      </location>`;
  const sc='\n      <scale value="1.200000,1.200000"/>';
  const layers=[
    mk(1,'TIME A - fraction mode: must ARRIVE at right edge exactly when clip ENDS',0,'#FF3DE8FF',mv(200,[['0.000000000',160],['1.000000000',920]]),sc),
    mk(2,'TIME B - seconds mode: must ARRIVE at right edge exactly when clip ENDS',0,'#FFFF8A4C',mv(380,[['0.000000000',160],['4.000000000',920]]),sc),
    mk(3,'EASE arrive-key: sits almost still in first half, then rushes (= easing is on ARRIVING key)',0,'#FFFC01E6',mv(560,[[tt(0),160],[tt(.5),540,'cubicBezier 0.9 0 1 0.1'],[tt(1),920]]),sc),
    mk(4,'EASE depart-key: control, easing placed on DEPARTING key',0,'#FF3DFFA6',mv(740,[[tt(0),160,'cubicBezier 0.9 0 1 0.1'],[tt(.5),540],[tt(1),920]]),sc),
    mk(5,'ROTATION - bar must turn CLOCKWISE and point DOWN at the end',0,'#FFFFE45E',`<location value="540.000000,900.000000,0.000000"/>`,`\n      <scale value="2.500000,0.300000"/>\n      <rotation>\n        <kf t="${tt(0)}" v="0.000000"/>\n        <kf t="${tt(1)}" v="90.000000"/>\n      </rotation>`),
    mk(6,'ROUND - must be a CIRCLE (missing or square: change "Round shape id" in the export panel)',0,'#FFFFFFFF',`<location value="160.000000,950.000000,0.000000"/>`,sc,o.round||'.ellipse'),
    mk(7,'ORDER first in file - RED square',0,'#FFFF2D2D',`<location value="780.000000,950.000000,0.000000"/>`,sc),
    mk(8,'ORDER second in file - BLUE square: if BLUE covers RED then first layer = bottom, if RED covers BLUE set "First layer = top"',0,'#FF2D6BFF',`<location value="830.000000,985.000000,0.000000"/>`,sc)
  ].join('\n');
  // note: layers 1,3,4 are fraction-mode; layer 2 is seconds-mode. The scene stays valid either way, the wrong one simply overshoots.
  return `<?xml version="1.0" encoding="UTF-8"?>
<scene title="vid2anim calibration" width="1080" height="1080" exportWidth="1080" exportHeight="1080" bgcolor="#FF0D0F24" totalTime="${totalMs}" fps="30" modifiedTime="${Date.now()}" amver="1028425" ffver="106" am="com.alightcreative.motion/5.0.273.1028425" amplatform="android">
${layers}
</scene>
`;
}

/* ───────── UI wiring ───────── */
export function initAlight(c){
  const $=c.$,st=c.st,LS='vid2anim.alight';
  if(['amXml','amCal','amShare','amMsg'].some(i=>!$(i))){console.warn('Alight panel not found in index.html, export disabled');return}
  const F=['amLong','amBg','amThick','amTime','amEase','amDense','amFps','amInv','amStyle','amEdge','amHead','amRound','amOrder'].filter(i=>$(i));
  const get=i=>!$(i)?undefined:$(i).type==='checkbox'?$(i).checked:$(i).value;
  try{const s=JSON.parse(localStorage.getItem(LS)||'{}');for(const i of F)if(i in s&&$(i)){if($(i).type==='checkbox')$(i).checked=s[i];else $(i).value=s[i]}}catch{}
  const save=()=>{try{localStorage.setItem(LS,JSON.stringify(Object.fromEntries(F.map(i=>[i,get(i)]))))}catch{}};
  for(const i of F)$(i).addEventListener('change',save);
  const opts=()=>({long:+get('amLong'),bg:get('amBg'),thick:+get('amThick')||100,tmode:get('amTime'),easeOn:get('amEase'),dense:get('amDense')==='dense',fps:get('amFps'),inv:get('amInv'),style:get('amStyle')||'stick',edge:$('amEdge')?+get('amEdge'):2,headPct:+get('amHead')||100,round:(get('amRound')||'.ellipse').trim()||'.ellipse',order:get('amOrder')||'bottom',title:c.title()});
  const safe=s=>(s||'vid2anim').replace(/\.[^.]+$/,'').replace(/[^\w\- ]+/g,'_').slice(0,40)||'vid2anim';
  const make=()=>{const o=opts();const r=buildScene(c,o);return{...r,name:safe(o.title)+'_alight.xml'}};
  const msg=t=>{$('amMsg').textContent=t;st(t)};

  $('amXml').onclick=()=>{try{const r=make();c.dl(r.name,r.xml,'application/xml');msg(`Saved ${r.name}: ${r.info.layers} layers · ${r.info.canvas} · ${(r.info.ms/1000).toFixed(2)}s @ ${r.info.fps} fps · ${(r.info.bytes/1024).toFixed(0)} KB`)}catch(e){msg(e.message)}};
  $('amCal').onclick=()=>{c.dl('vid2anim_calibration.xml',buildCalibration(opts()),'application/xml');msg('Calibration project saved. Import it in Alight Motion and follow the layer names.')};
  $('amShare').onclick=async()=>{try{const r=make(),f=new File([r.xml],r.name,{type:'application/xml'});
      if(navigator.canShare&&navigator.canShare({files:[f]})){await navigator.share({files:[f],title:r.name});msg('Shared.')}else{c.dl(r.name,r.xml,'application/xml');msg('This browser cannot share files, so the file was downloaded instead.')}}catch(e){if(e.name!=='AbortError')msg(e.message)}};
}
