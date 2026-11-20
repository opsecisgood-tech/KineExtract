import {PoseLandmarker,FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {initAlight} from './alight.js?v=5';
const $=i=>document.getElementById(i),V=$('v'),O=$('ov'),ox=O.getContext('2d'),G=$('graph'),EC=$('ezc'),ec=EC.getContext('2d'),PV=$('pv'),pc=PV.getContext('2d'),crop=document.createElement('canvas'),cx=crop.getContext('2d');
const WASM="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",MURL=m=>`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${m}/float16/1/pose_landmarker_${m}.task`;
let fs,delegate='GPU',LMs={},tsBase=0,frames=[],EF=[],R={},RF={},S={},K={},CORR=[],undo=[],redo=[],sel='head',prop='rot',selKf=-1,stop=false,view=[0,1],fps=15,start=0,dirty=true,endAt=null,lastT=-1,meta={},curFile=null,subj=null,picking=false,hov=null,rep={},lasso=null,lp=[],lassoing=false,drawingL=false,anchor=null,vel={x:0,y:0},BR=null,seed=null,ident=null,pickC=[],pickT=-1,lastPx=null;
const C={L:'#3de8ff',R:'#ff8a4c',N:'#b4b8e0',sel:'#ffe45e',brand:'#fc01e6',ok:'#3dffa6'};
const KP=['rot','x','y','jt','len'],PROPS={rot:'Rotation °',x:'Position X',y:'Position Y',jt:'Joint angle °',len:'Segment length',vrot:'Angular velocity °/s (read-only)',vpos:'Position velocity /s (read-only)'};
$('prop').innerHTML=Object.entries(PROPS).map(([k,v])=>`<option value=${k}>${v}</option>`).join('');
const NAMES='nose,eyeIL,eyeL,eyeOL,eyeIR,eyeR,eyeOR,earL,earR,mouthL,mouthR,shL,shR,elL,elR,wrL,wrR,pkL,pkR,ixL,ixR,thL,thR,hipL,hipR,knL,knR,anL,anR,heL,heR,toL,toR'.split(',');
// ---- easing: free cubic-bezier [x1,y1,x2,y2] (x clamped 0..1 so time can't reverse; y may overshoot -1..2) or special curves
const EZ={Linear:[0,0,1,1],'Ease-in':[.42,0,1,1],'Ease-out':[0,0,.58,1],'Ease-in-out':[.42,0,.58,1],'Sharp-out':[.05,.9,.2,1],'Sharp-in':[.8,0,.95,.1],'Back-out':[.2,1.6,.4,1],'Back-in':[.6,-.6,.8,0],Elastic:'elastic',Bounce:'bounce',Hold:'hold'};
$('ezs').innerHTML=Object.keys(EZ).concat('Custom').map(k=>`<option>${k}</option>`).join('');
function ez(e,x){if(e==='hold')return x>=1?1:0;if(e==='elastic')return x<=0?0:x>=1?1:2**(-10*x)*Math.sin((x*10-.75)*2*Math.PI/3)+1;
 if(e==='bounce'){const n=7.5625,d=2.75;if(x<1/d)return n*x*x;if(x<2/d)return n*(x-=1.5/d)*x+.75;if(x<2.5/d)return n*(x-=2.25/d)*x+.9375;return n*(x-=2.625/d)*x+.984375}
 const[a,b,c,d]=e;let lo=0,hi=1,t=x;for(let k=0;k<22;k++){const m=3*(1-t)*(1-t)*t*a+3*(1-t)*t*t*c+t**3;if(m<x)lo=t;else hi=t;t=(lo+hi)/2}return 3*(1-t)*(1-t)*t*b+3*(1-t)*t*t*d+t**3}
const easeName=e=>typeof e==='string'?Object.keys(EZ).find(n=>EZ[n]===e):Object.keys(EZ).find(n=>Array.isArray(EZ[n])&&EZ[n].every((v,i)=>Math.abs(v-e[i])<1e-6))||'Custom';
// ---- segments (point spec = landmark indices, averaged; repeats act as weights)
const D={},OFF=new Set();const A=(id,n,a,b,lv,j)=>D[id]={n,a,b,lv,j};
A('head','Head',[7,8],[0],'123');A('neck','Neck',[11,12],[7,8],'23');A('torso','Torso',[11,12],[23,24],'1');
A('utor','Upper torso',[11,12],[11,12,23,24],'2');A('ltor','Lower torso',[11,12,23,24],[23,24],'2');
A('utor3','Upper torso',[11,12],[11,12,11,12,23,24],'3');A('mtor','Mid torso',[11,12,11,12,23,24],[11,12,23,24,23,24],'3');A('ltor3','Lower torso',[11,12,23,24,23,24],[23,24],'3');A('pelvis','Pelvis',[23],[24],'23');
for(const[s,o,N]of[['L',0,'Left'],['R',1,'Right']]){const q=(id,n,a,b,lv,j)=>{A(s+id,N+' '+n,a,b,lv,j);D[s+id].side=s};
 q('sh','shoulder',[11,12],[11+o],'3');q('ua','upper arm',[11+o],[13+o],'123',[[23+o],[11+o],[13+o]]);q('fa','forearm',[13+o],[15+o],'123',[[11+o],[13+o],[15+o]]);
 q('wr','wrist',[17+o],[19+o],'3',[[13+o],[15+o],[17+o,19+o]]);q('hand','hand',[15+o],[17+o,19+o],'23');
 q('th','thigh',[23+o],[25+o],'123',[[11+o],[23+o],[25+o]]);q('sn','lower leg',[25+o],[27+o],'123',[[23+o],[25+o],[27+o]]);
 q('knee','knee chord',[23+o],[27+o],'3',[[23+o],[25+o],[27+o]]);q('foot','foot',[27+o],[31+o],'23',[[25+o],[27+o],[31+o]]);q('ank','heel-toe',[29+o],[31+o],'3')}
const CONN=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32],[15,17],[15,19],[17,19],[16,18],[16,20],[18,20],[0,7],[0,8]];
let SWA;const SW=[[1,4],[2,5],[3,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32]];
const lvIds=()=>Object.keys(D).filter(id=>D[id].lv.includes($('rig').value)),act=()=>lvIds().filter(id=>!OFF.has(id));
// removed parts: their landmarks are ignored when judging, cross-checking and verifying the track
let ALM=null;const ok=i=>!ALM||ALM.has(i),FACE=[1,2,3,4,5,6,9,10],lmOf=id=>{const d=D[id];return[...d.a,...d.b,...(d.j||[]).flat()]};
function applyOff(){ALM=null;if(OFF.size){const s=new Set(FACE);for(const id in D)if(!OFF.has(id))lmOf(id).forEach(i=>s.add(i));if(OFF.has('head'))FACE.forEach(i=>s.delete(i));ALM=s}
 MAIN=MAIN0.filter(ok);if(MAIN.length<3)MAIN=MAIN0;BONES=BONES0.filter(([a,b])=>ok(a)&&ok(b));if(BONES.length<3)BONES=BONES0;SWA=SW.filter(([i,j])=>ok(i)&&ok(j));if(!SWA.length)SWA=SW;BR=null}
const pt=(lm,s)=>{let x=0,y=0,v=1;for(const i of s){const p=lm[i];x+=p.x;y+=p.y;v=Math.min(v,p.v)}return{x:x/s.length,y:y/s.length,v}};
const ang=(p,W,H)=>{const ax=(p[0].x-p[1].x)*W,ay=(p[0].y-p[1].y)*H,bx=(p[2].x-p[1].x)*W,by=(p[2].y-p[1].y)*H,m=Math.hypot(ax,ay)*Math.hypot(bx,by);return m?Math.acos(Math.max(-1,Math.min(1,(ax*bx+ay*by)/m)))*180/Math.PI:NaN};
const VW=()=>V.videoWidth||meta.w||1280,VH=()=>V.videoHeight||meta.h||720;
// ---- series helpers
function fill(a){const o=a.slice(),n=o.length,f=o.findIndex(v=>!isNaN(v));if(f<0)return o.fill(0);for(let i=0;i<f;i++)o[i]=o[f];let l=f;for(let i=f+1;i<n;i++)if(!isNaN(o[i])){for(let j=l+1;j<i;j++)o[j]=o[l]+(o[i]-o[l])*(j-l)/(i-l);l=i}for(let i=l+1;i<n;i++)o[i]=o[l];return o}
const unwrap=a=>{if(!a.length)return a;const o=[a[0]];for(let i=1;i<a.length;i++){let d=a[i]-a[i-1];d-=360*Math.round(d/360);o.push(o[i-1]+d)}return o};
function sm(a,m,s){const n=a.length;if(m==='none'||!s||n<3)return a.slice();
 if(m==='ma'){const w=[0,1,2,4][s];return a.map((_,i)=>{let t=0,c=0;for(let j=i-w;j<=i+w;j++)if(j>=0&&j<n){t+=a[j];c++}return t/c})}
 if(m==='sg'){const h=[0,2,3,5][s],d=(2*h+3)*(2*h+1)*(2*h-1),c=[];for(let j=-h;j<=h;j++)c.push(3*(3*h*h+3*h-1-5*j*j)/d);return a.map((_,i)=>{let t=0;for(let j=-h;j<=h;j++)t+=c[j+h]*a[Math.min(n-1,Math.max(0,i+j))];return t})}
 let f=[],b=[];if(m==='ema'){const al=[1,.6,.4,.2][s];a.forEach((v,i)=>f.push(i?al*v+(1-al)*f[i-1]:v));for(let i=n-1;i>=0;i--)b[i]=i<n-1?al*a[i]+(1-al)*b[i+1]:a[i]}
 else{const dd=a.slice(1).map((v,i)=>v-a[i]),Rr=Math.max(1e-12,dd.reduce((t,v)=>t+v*v,0)/dd.length/2),Q=[.6,.2,.05][s-1]*Rr;
  const run=(arr)=>{let x=arr[0],P=Rr;return arr.map(v=>{P+=Q;const k=P/(P+Rr);x+=k*(v-x);P*=1-k;return x})};f=run(a);b=run(a.slice().reverse()).reverse()}
 return a.map((_,i)=>(f[i]+b[i])/2)}
const grad=a=>a.map((_,i)=>{const p=Math.max(0,i-1),n=Math.min(a.length-1,i+1);return(a[n]-a[p])/((n-p)||1)});
const px=()=>$('coord').value==='px',U=p=>px()?(p==='x'?VW():(p==='y'||p==='len'||p==='vpos')?VH():1):1;
const wrapA=v=>v-360*Math.round(v/360),dv=(p,v)=>p==='rot'&&$('wrap').checked?wrapA(v):v,f3=v=>isFinite(v)?(Math.abs(v)>=100?v.toFixed(1):v.toFixed(3)):'–';
const ci=()=>frames.length?Math.max(0,Math.min(frames.length-1,Math.round((V.currentTime-start)*fps))):-1;
const st=t=>$('status').textContent=t;
// ---- landmarks: adaptive filter + manual corrections (raw data is never overwritten)
function filt(){const o=frames.map(f=>f.lm?f.lm.map(p=>({...p})):null);const cs=camShifts(),sh=sg=>{if(cs)o.forEach((l,i)=>{if(l)for(const p of l){p.x+=sg*cs[i][0];p.y+=sg*cs[i][1]}})};sh(-1);if($('lf').checked)filtRun(o);sh(1);return o}
function filtRun(o){const Te=1/fps,al=c=>1/(1+1/(2*Math.PI*c)/Te);
 for(let k=0;k<33;k++)for(const ax of['x','y']){let xh=null,dh=0;for(const l of o){if(!l)continue;const x=l[k][ax];if(xh===null){xh=x;continue}dh+=al(1)*((x-xh)/Te-dh);xh+=al(1.5+8*Math.abs(dh))*(x-xh);l[k][ax]=xh}}}
function effective(){const E0=filt();EF=E0.map(l=>l?l.map(p=>({...p})):null);
 for(const c of CORR){const b=E0[c.i]?.[c.k];if(!b)continue;const dx=c.x-b.x,dy=c.y-b.y,r=c.r||0;
  for(let j=Math.max(0,c.i-r);j<=Math.min(EF.length-1,c.i+r);j++){const l=EF[j];if(!l)continue;const w=r?1-Math.abs(j-c.i)/(r+1):1;l[c.k].x+=dx*w;l[c.k].y+=dy*w;l[c.k].v=1;if(j===c.i)l[c.k].fixed=true}}}
// ---- compute segment data
function compute(){if(!frames.length)return;effective();const W=VW(),H=VH(),off=+$('off').value||0,mv=+$('mv').value,cs=camShifts();R={};
 for(const id in D){const d=D[id],r={rot:[],x:[],y:[],jt:[],len:[],c:[]};
  EF.forEach((lm,fi)=>{if(!lm){for(const k in r)r[k].push(NaN);return}const[sx,sy]=cs?cs[fi]:[0,0];
   const a=pt(lm,d.a),b=pt(lm,d.b),c=Math.min(a.v,b.v),ok=c>=mv,dx=(b.x-a.x)*W,dy=(b.y-a.y)*H;
   r.c.push(c);r.rot.push(ok?Math.atan2(dy,dx)*180/Math.PI-off:NaN);r.x.push(ok?(a.x+b.x)/2-sx:NaN);r.y.push(ok?(a.y+b.y)/2-sy:NaN);r.len.push(ok?Math.hypot(dx,dy)/H:NaN);
   r.jt.push(ok&&d.j?ang(d.j.map(s=>pt(lm,s)),W,H):NaN)});R[id]=r}
 smooth();genKF();probList()}
const SL=p=>+$(p==='rot'?'sR':p==='jt'?'sJ':'sP').value;
function smooth(){S={};RF={};for(const id in R){const r=R[id],o={},rf={};for(const p of KP){let a=fill(r[p]);if(p==='rot')a=unwrap(a);rf[p]=a;o[p]=sm(a,$('sm').value,SL(p))}
 o.c=r.c.map(c=>isNaN(c)?0:c);o.vrot=grad(o.rot).map(v=>v*fps);const gx=grad(o.x),gy=grad(o.y);o.vpos=gx.map((v,i)=>Math.hypot(v,gy[i])*fps);S[id]=o;RF[id]=rf}}
// ---- keyframes: RDP per property + max gap + validity fix + easing fit
const PR={min:[6,1.2,.3,0],bal:[3,.6,.15,0],acc:[1.2,.25,.05,0]};
$('preset').onchange=()=>{const p=PR[$('preset').value];if(p){$('kR').value=p[0];$('kP').value=p[1];$('kMin').value=p[2];$('kMax').value=p[3]}if(frames.length)compute()};
const epsOf=p=>p==='rot'||p==='jt'?+$('kR').value:+$('kP').value/100;
function rdp(y,eps,mn){const n=y.length,keep=new Set([0,n-1]);if(n<3)return[...keep];
 (function r(a,b){let m=-1,md=0;for(let i=a+1;i<b;i++){const e=Math.abs(y[i]-(y[a]+(y[b]-y[a])*(i-a)/(b-a)));if(e>md){md=e;m=i}}if(m>0&&md>eps&&m-a>=mn&&b-m>=mn){keep.add(m);r(a,m);r(m,b)}})(0,n-1);return[...keep].sort((a,b)=>a-b)}
const why=(y,i,p)=>i===0?'Start of clip':i===y.length-1?'End of clip':(y[i]-y[i-1])*(y[i+1]-y[i])<0?'Direction reversal':p==='rot'?'Significant rotation change':p==='jt'?'Significant angle change':'Significant position/length change';
function genKF(){K={};const mn=Math.max(1,Math.round(fps*+$('kMin').value)),mg=Math.round(fps*+$('kMax').value);
 for(const id in S){K[id]={};for(const p of KP){const y=S[id][p];if(!y.length)continue;let idx=rdp(y,epsOf(p),mn);
  if(mg>0){const o=[];idx.forEach((i,j)=>{o.push([i,why(y,i,p)]);const nx=idx[j+1];if(nx!==undefined&&nx-i>mg){const c=Math.ceil((nx-i)/mg);for(let q=1;q<c;q++)o.push([Math.round(i+(nx-i)*q/c),'Max time gap'])}});K[id][p]=o.map(([i,w])=>({i,v:y[i],why:w,e:EZ.Linear}))}
  else K[id][p]=idx.map(i=>({i,v:y[i],why:why(y,i,p),e:EZ.Linear}));fix(id,p);fitAll(id,p)}}selKf=-1;dirty=true}
function segCost(a,b,y,e){const d=b.v-a.v;let s=0;for(let i=a.i+1;i<b.i;i++){const t=(y[i]-a.v)/d-ez(e,(i-a.i)/(b.i-a.i));s+=t*t}return s}
function fitAll(id,p){const k=K[id][p],y=S[id][p];for(let j=0;j<k.length-1;j++){const a=k[j],b=k[j+1];a.e=EZ.Linear;if(Math.abs(b.v-a.v)<1e-9||b.i-a.i<3)continue;const c0=segCost(a,b,y,EZ.Linear);let best=c0;
 for(const n in EZ){if(n==='Hold'||n==='Linear')continue;const e=EZ[n],c=segCost(a,b,y,e),need=typeof e==='string'?.5:.85;if(c<c0*need&&c<best){best=c;a.e=e}}}}
// Free-form fit: pattern search on the 4 bezier numbers against the real tracked data (not limited to presets). x in 0..1, y in -1..2.
function freeFit(id,p,j){const k=K[id][p],y=S[id][p],a=k[j],b=k[j+1];if(!b||Math.abs(b.v-a.v)<1e-9||b.i-a.i<3)return;
 let e=Array.isArray(a.e)?[...a.e]:[...EZ.Linear],c=segCost(a,b,y,e),st=.3;const lim=[[0,1],[-1,2],[0,1],[-1,2]];
 for(let g=0;g<60&&st>.01;g++){let imp=false;for(let q=0;q<4;q++)for(const s of[st,-st]){const n=[...e];n[q]=Math.min(lim[q][1],Math.max(lim[q][0],n[q]+s));const cc=segCost(a,b,y,n);if(cc<c-1e-9){c=cc;e=n;imp=true}}if(!imp)st/=2}
 if(c<segCost(a,b,y,a.e)*.97)a.e=e.map(v=>Math.round(v*100)/100)}
// A segment between two keyframes is ONE curve through time: it can't turn back on itself. Insert a keyframe wherever the data leaves the segment's value range (small overshoot is left to the easing).
function fix(id,p){const k=K[id][p],y=S[id][p];let n=0;for(let g=0;g<500;g++){let ins=null;
 for(let j=0;j<k.length-1&&!ins;j++){const a=k[j],b=k[j+1],t=Math.max(epsOf(p),.12*Math.abs(b.v-a.v)),lo=Math.min(a.v,b.v)-t,hi=Math.max(a.v,b.v)+t;let w=0,m=-1;for(let i=a.i+1;i<b.i;i++){const x=Math.max(lo-y[i],y[i]-hi);if(x>w){w=x;m=i}}if(m>0)ins=[j+1,m]}
 if(!ins)break;k.splice(ins[0],0,{i:ins[1],v:y[ins[1]],why:'Added: a segment cannot turn back on itself',e:EZ.Linear});n++}return n}
function curve(k,i){if(i<=k[0].i)return k[0].v;for(let j=0;j<k.length-1;j++){const a=k[j],b=k[j+1];if(i<=b.i)return a.v+(b.v-a.v)*ez(a.e,(i-a.i)/(b.i-a.i))}return k[k.length-1].v}
const KS=()=>K[sel]?.[prop];
// ---- smart tracking: multi-pass measurement, cross-check, verify sweeps, lasso focus
const vis=l=>MAIN.reduce((a,k)=>a+l[k].v,0)/MAIN.length,nm=l=>l.map(p=>({x:p.x,y:p.y,v:p.visibility??1})),tag=(l,s)=>{l.src=s;return l};
function box(l,all){let q=l.filter((p,i)=>p.v>.3&&(all||ok(i)));if(q.length<4)q=l;return{x0:Math.min(...q.map(p=>p.x)),x1:Math.max(...q.map(p=>p.x)),y0:Math.min(...q.map(p=>p.y)),y1:Math.max(...q.map(p=>p.y))}}
const bs=b=>Math.max(30,Math.hypot((b.x1-b.x0)*VW(),(b.y1-b.y0)*VH())),dd=(a,b)=>Math.hypot((a.x-b.x)*VW(),(a.y-b.y)*VH());
const cd=(a,b)=>{let z=[11,12,23,24].filter(ok);if(!z.length)z=[11,12,23,24];return z.reduce((t,i)=>t+dd(a[i],b[i]),0)/z.length};
const MAIN0=[0,11,12,13,14,15,16,23,24,25,26,27,28],BONES0=[[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28],[11,12],[23,24],[11,23],[12,24]];let MAIN=MAIN0,BONES=BONES0;SWA=SW;
const med=a=>{const s=[...a].sort((x,y)=>x-y),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
const tolV=()=>(+$('dtol').value||6)/100;
// ---- identity lock: keeps the chosen person when others stand close or cross in front.
// Three cues are combined: (1) clothing colour fingerprint of torso + thighs, learned from the chosen person and refreshed only on confident frames,
// (2) motion prediction (where the torso should be given its last speed), (3) body size. Position alone is what swaps people when they overlap.
const lockOn=()=>!!$('lock')&&$('lock').checked;
const fcv=document.createElement('canvas'),fg=fcv.getContext('2d',{willReadFrequently:true});let fKey='',fData=null;
function frameData(){const key=V.currentTime+'|'+V.videoWidth;if(key===fKey&&fData)return fData;
 try{const w=VW(),h=VH(),k=240/w;fcv.width=240;fcv.height=Math.max(8,Math.round(h*k));fg.drawImage(V,0,0,fcv.width,fcv.height);fData=fg.getImageData(0,0,fcv.width,fcv.height);fKey=key;return fData}catch{fData=null;fKey='';return null}}
function shrinkP(P,f){let cx_=0,cy_=0;for(const p of P){cx_+=p[0]/P.length;cy_+=p[1]/P.length}return P.map(p=>[cx_+(p[0]-cx_)*f,cy_+(p[1]-cy_)*f])}
// 28-bin colour histogram: 1 black, 3 grey levels, 12 hues x (dark|bright). Hue bins are smoothed so a small colour shift does not break the match.
function colBin(R,G,B){const r=R/255,g=G/255,bl=B/255,mx=Math.max(r,g,bl),mn=Math.min(r,g,bl),d=mx-mn,s=mx?d/mx:0;
 if(mx<.18)return 0;if(s<.22)return mx<.45?1:mx<.75?2:3;const hu=d?(mx===r?((g-bl)/d+6)%6:mx===g?(bl-r)/d+2:(r-g)/d+4):0;return 4+(Math.floor(hu*2)%12)*2+(mx<.55?0:1)}
function smoothH(Hh,n){if(n<12)return null;for(let i=0;i<28;i++)Hh[i]/=n;const o=Hh.slice();for(let i=4;i<28;i++)o[i]=.6*Hh[i]+.2*Hh[4+(i-4+2)%24]+.2*Hh[4+(i-4-2+24)%24];return o}
function histOf(P,im){const w=im.width,h=im.height,b=bbOf(P),x0=Math.max(0,Math.floor(b.x0*w)),x1=Math.min(w-1,Math.ceil(b.x1*w)),y0=Math.max(0,Math.floor(b.y0*h)),y1=Math.min(h-1,Math.ceil(b.y1*h)),Hh=new Float32Array(28);let n=0;
 for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){if(!inPoly((x+.5)/w,(y+.5)/h,P))continue;const o=4*(y*w+x);Hh[colBin(im.data[o],im.data[o+1],im.data[o+2])]++;n++}
 return smoothH(Hh,n)}
function sigOf(l){const im=frameData();if(!im)return null;const P=ids=>ids.every(i=>l[i].v>.3)?ids.map(i=>[l[i].x,l[i].y]):null,t=P([11,12,24,23]),g=P([23,24,26,25]);
 const s={t:t?histOf(shrinkP(t,.55),im):null,l:g?histOf(shrinkP(g,.55),im):null};return s.t||s.l?s:null}
function simSig(a,b){if(!a||!b)return null;let w=0,s=0;for(const[k,q]of[['t',.6],['l',.4]])if(a[k]&&b[k]){let m=0;for(let i=0;i<28;i++)m+=Math.min(a[k][i],b[k][i]);s+=q*m;w+=q}return w?s/w:null}
const copySig=s=>s?{t:s.t?s.t.slice():null,l:s.l?s.l.slice():null}:null;
function learnIdent(lm,r,flag){if(!lockOn())return;const s=sigOf(lm);if(!s)return;if(!ident){ident=copySig(s);return}
 const sm=simSig(s,ident);if(sm!==null&&(sm<.45||vis(lm)<.6||r.disp||/jump/.test(flag)))return; // only learn from confident frames that still look like our person
 for(const k of['t','l'])if(s[k]){ident[k]=ident[k]?ident[k].map((v,i)=>.88*v+.12*s[k][i]):s[k].slice()}}
// distance (in body sizes) from a point to the nearest bone of a pose; 0 when it is inside the torso
function skelDist(l,p){const W=VW(),H=VH(),s=bs(box(l)),px=p.x*W,py=p.y*H;let m=1e9;
 for(const[a,b]of CONN){if(l[a].v<.3||l[b].v<.3)continue;const ax=l[a].x*W,ay=l[a].y*H,dx=l[b].x*W-ax,dy=l[b].y*H-ay,L2=dx*dx+dy*dy,t=L2?Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/L2)):0;m=Math.min(m,Math.hypot(px-ax-t*dx,py-ay-t*dy))}
 if([11,12,24,23].every(i=>l[i].v>.3)&&inPoly(p.x,p.y,[11,12,24,23].map(i=>[l[i].x,l[i].y])))m=0;return m/s}
const cdS=(a,b,dx,dy)=>{let z=[11,12,23,24].filter(ok);if(!z.length)z=[11,12,23,24];return z.reduce((t,i)=>t+Math.hypot((a[i].x-b[i].x-dx)*VW(),(a[i].y-b[i].y-dy)*VH()),0)/z.length};
// held frames: animation is often drawn on twos or threes. If the subject area did not change since the last measured frame, reuse that result.
function isHeld(bb){const im=frameData();if(!im)return false;const d=im.data,w=im.width,h=im.height;
 if(!lastPx||lastPx.length!==d.length){lastPx=new Uint8ClampedArray(d);return false}
 let x0=0,x1=w,y0=0,y1=h;if(bb){x0=Math.max(0,Math.floor((bb.x0-.04)*w));x1=Math.min(w,Math.ceil((bb.x1+.04)*w));y0=Math.max(0,Math.floor((bb.y0-.04)*h));y1=Math.min(h,Math.ceil((bb.y1+.04)*h))}
 let s=0,c=0,big=0;for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const o=4*(y*w+x),df=Math.abs(d[o]-lastPx[o])+Math.abs(d[o+1]-lastPx[o+1])+Math.abs(d[o+2]-lastPx[o+2]);s+=df;c++;if(df>60)big++}
 const held=c>0&&s/c/3<1.2&&big/c<.006;if(!held)lastPx=new Uint8ClampedArray(d);return held}
// ---- camera motion: how the whole picture moved between two analysed frames.
// Block matching on a small grey copy of the frame, coarse to fine, background pixels only (the person's box is left out so their own
// movement is not mistaken for camera movement). Result is a translation in frame units (+x = picture moved right).
// @@cam-start
const camG=im=>{const w=im.width,h=im.height,d=im.data,g=new Uint8Array(w*h);for(let i=0;i<w*h;i++)g[i]=(d[4*i]*77+d[4*i+1]*150+d[4*i+2]*29)>>8;return{w,h,g}};
const camDown=(a,k)=>{const w=Math.floor(a.w/k),h=Math.floor(a.h/k),g=new Uint8Array(w*h);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let t=0;for(let j=0;j<k;j++)for(let i=0;i<k;i++)t+=a.g[(y*k+j)*a.w+x*k+i];g[y*w+x]=t/(k*k)}return{w,h,g}};
// 1 = background pixel, 0 = inside the (grown) subject box
const camMask=(w,h,bb)=>{if(!bb)return null;const gx=(bb.x1-bb.x0)*.25+.02,gy=(bb.y1-bb.y0)*.25+.02,x0=Math.floor((bb.x0-gx)*w),x1=Math.ceil((bb.x1+gx)*w),y0=Math.floor((bb.y0-gy)*h),y1=Math.ceil((bb.y1+gy)*h),m=new Uint8Array(w*h).fill(1);let z=0;
 for(let y=Math.max(0,y0);y<Math.min(h,y1);y++)for(let x=Math.max(0,x0);x<Math.min(w,x1);x++){m[y*w+x]=0;z++}return z>w*h*.7?null:m};
// mean clipped |difference| between a(x,y) and b(x+dx,y+dy), over background pixels; 1e9 when too little overlaps
function camSad(a,b,dx,dy,m,T,st){const w=a.w,h=a.h,x0=Math.max(0,-dx),x1=Math.min(w,w-dx),y0=Math.max(0,-dy),y1=Math.min(h,h-dy);let s=0,c=0;
 for(let y=y0;y<y1;y+=st){const ra=y*w,rb=(y+dy)*w+dx;for(let x=x0;x<x1;x+=st){if(m&&!m[ra+x])continue;let d=a.g[ra+x]-b.g[rb+x];if(d<0)d=-d;s+=d>T?T:d;c++}}
 return c>(w/st)*(h/st)*.15?s/c:1e9}
function camEst(P,C,bb,R=14){if(!P||!C||P.w!==C.w||P.h!==C.h)return{x:0,y:0,c:0};
 const k=4,p4=camDown(P,k),c4=camDown(C,k),m4=camMask(p4.w,p4.h,bb);let bx=0,by=0,bs_=1e18;
 for(let dy=-R;dy<=R;dy++)for(let dx=-R;dx<=R;dx++){const v=camSad(p4,c4,dx,dy,m4,40,1);if(v>=1e9)continue;const sc=v*(1+.004*Math.hypot(dx,dy));if(sc<bs_){bs_=sc;bx=dx;by=dy}}
 if(bs_>=1e17)return{x:0,y:0,c:0};
 const m1=camMask(P.w,P.h,bb),cx=bx*k,cy=by*k;let fx=cx,fy=cy,fb=1e18;
 for(let dy=cy-5;dy<=cy+5;dy++)for(let dx=cx-5;dx<=cx+5;dx++){const v=camSad(P,C,dx,dy,m1,40,2);if(v<fb){fb=v;fx=dx;fy=dy}}
 const s0=camSad(P,C,0,0,m1,40,2),gain=s0<1e9?(s0-fb)/(s0+1e-3):0;
 if(fb>=1e9||(gain<.03&&Math.hypot(fx,fy)>2))return{x:0,y:0,c:0};
 const sub=(l,c,r)=>{const d=l-2*c+r;return d>1e-9?Math.max(-.5,Math.min(.5,.5*(l-r)/d)):0},
  ax=camSad(P,C,fx-1,fy,m1,40,2),bx2=camSad(P,C,fx+1,fy,m1,40,2),ay=camSad(P,C,fx,fy-1,m1,40,2),by2=camSad(P,C,fx,fy+1,m1,40,2);
 const sx=ax<1e9&&bx2<1e9?sub(ax,fb,bx2):0,sy=ay<1e9&&by2<1e9?sub(ay,fb,by2):0;
 return{x:(fx+sx)/P.w,y:(fy+sy)/P.h,c:Math.max(0,Math.min(1,gain*4))}}
// @@cam-end
let camPrevG=null,camCum={x:0,y:0},camUnc=0,camFast=false;
function camStep(prevLM){const im=frameData();if(!im){camPrevG=null;return{x:0,y:0,c:0}}const cur=camG(im),P=camPrevG;camPrevG=cur;return camEst(P,cur,prevLM?box(prevLM,true):null)}
const shiftLM=(l,dx,dy)=>{const o=l.map(p=>({...p,x:p.x+dx,y:p.y+dy}));o.src=l.src;return o};
const ccOf=j=>frames[j]?.cc||[0,0];
// per-frame camera offset to remove from positions (null = leave as filmed)
const lpZ=(a0,tau)=>{const al=1-Math.exp(-1/(fps*tau)),n0=a0.length,P=Math.min(n0-1,Math.ceil(4*tau*fps)),a=[]; // odd reflection at both ends keeps a steady pan from leaving a start-up error
 for(let k=P;k>=1;k--)a.push(2*a0[0]-a0[k]);for(const v of a0)a.push(v);for(let k=1;k<=P;k++)a.push(2*a0[n0-1]-a0[n0-1-k]);
 const n=a.length,f=[],b=[];a.forEach((v,i)=>f.push(i?al*v+(1-al)*f[i-1]:v));for(let i=n-1;i>=0;i--)b[i]=i<n-1?al*a[i]+(1-al)*b[i+1]:a[i];return a0.map((_,i)=>(f[i+P]+b[i+P])/2)};
function camShifts(){const m=$('stab')?.value||'off',n=frames.length;if(m==='off'||!n||!frames.some(f=>f.cc))return null;
 const cx=fill(frames.map(f=>f.cc?f.cc[0]:NaN)),cy=fill(frames.map(f=>f.cc?f.cc[1]:NaN));if(m==='lock')return cx.map((v,i)=>[v,cy[i]]);
 const tau=Math.max(.1,+$('stabT').value||.6),lx=lpZ(cx,tau),ly=lpZ(cy,tau);return cx.map((v,i)=>[v-lx[i],cy[i]-ly[i]])}
// ---- lasso geometry
const inPoly=(x,y,P)=>{let c=false;for(let i=0,j=P.length-1;i<P.length;j=i++){const[xi,yi]=P[i],[xj,yj]=P[j];if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)c=!c}return c};
const bbOf=P=>({x0:Math.min(...P.map(p=>p[0])),x1:Math.max(...P.map(p=>p[0])),y0:Math.min(...P.map(p=>p[1])),y1:Math.max(...P.map(p=>p[1]))});
function cenOf(P){let a=0,x=0,y=0;for(let i=0;i<P.length;i++){const[x0,y0]=P[i],[x1,y1]=P[(i+1)%P.length],c=x0*y1-x1*y0;a+=c;x+=(x0+x1)*c;y+=(y0+y1)*c}if(Math.abs(a)<1e-9){const b=bbOf(P);return{x:(b.x0+b.x1)/2,y:(b.y0+b.y1)/2}}return{x:x/(3*a),y:y/(3*a)}}
const torsoC=l=>({x:(l[11].x+l[12].x+l[23].x+l[24].x)/4,y:(l[11].y+l[12].y+l[23].y+l[24].y)/4});
// The lasso and the subject circle are FIXED in video space. They never follow the person and never move on playback or scrubbing:
// they only change when you draw a new lasso or drag the centre handle yourself.
function lassoNow(){if(!lasso)return null;const dx=camCum.x,dy=camCum.y;if(!dx&&!dy)return lasso;return{...lasso,pts:lasso.pts.map(([x,y])=>[x+dx,y+dy]),c:{x:lasso.c.x+dx,y:lasso.c.y+dy},bb:{x0:lasso.bb.x0+dx,x1:lasso.bb.x1+dx,y0:lasso.bb.y0+dy,y1:lasso.bb.y1+dy}}}
function maskPoly(ref){const L=lassoNow(ref);if(!L)return null;const f=1+(+$('fpad').value||35)/100;return L.pts.map(([x,y])=>[L.c.x+(x-L.c.x)*f,L.c.y+(y-L.c.y)*f])}
function finishLasso(){let P=lp;lp=[];$('lasso').classList.remove('on');O.style.cursor='';lastT=-1;
 if(P.length>90)P=P.filter((_,i)=>i%Math.ceil(P.length/90)===0);
 if(P.length<6||(()=>{const b=bbOf(P);return b.x1-b.x0<.01||b.y1-b.y0<.01})())return st('Lasso too small. Drag a closed loop around the spot you want tracked.');
 lasso={pts:P,bb:bbOf(P),c:cenOf(P),t:V.currentTime};anchor=null;
 st(`Lasso set at ${V.currentTime.toFixed(2)}s. Press Analyze (works best when drawn on the first frame of the clip).`)}
function drawLassoOv(w,h,lm){const lw=Math.max(2,w/320);
 if(lp.length>1){ox.strokeStyle=C.brand;ox.lineWidth=lw*1.5;ox.beginPath();lp.forEach(([x,y],j)=>j?ox.lineTo(x*w,y*h):ox.moveTo(x*w,y*h));ox.stroke()}
 const L=lasso&&lassoNow(lm);if(!L)return;ox.save();
 ox.fillStyle='rgba(0,0,0,.35)';ox.beginPath();ox.rect(0,0,w,h);L.pts.forEach(([x,y],j)=>j?ox.lineTo(x*w,y*h):ox.moveTo(x*w,y*h));ox.closePath();ox.fill('evenodd');
 ox.beginPath();L.pts.forEach(([x,y],j)=>j?ox.lineTo(x*w,y*h):ox.moveTo(x*w,y*h));ox.closePath();ox.setLineDash([lw*4,lw*3]);ox.strokeStyle=C.brand;ox.lineWidth=lw*1.5;ox.stroke();ox.setLineDash([]);
 const r=w/45,cx_=L.c.x*w,cy_=L.c.y*h;ox.setLineDash([]);ox.shadowColor=C.brand;ox.shadowBlur=lw*4;ox.beginPath();ox.moveTo(cx_-r,cy_);ox.lineTo(cx_+r,cy_);ox.moveTo(cx_,cy_-r);ox.lineTo(cx_,cy_+r);ox.stroke();ox.beginPath();ox.arc(cx_,cy_,w/90,0,7);ox.stroke();ox.restore()}
// ---- subject choice (lasso > clicked point > largest/clearest), then continuity
function choose(c,prev,miss){if(!c.length)return null;const L=lasso&&lassoNow(prev),LK=lockOn(),
 fo=l=>{if(!L)return 0;const q=l.filter(p=>p.v>.3);return q.length?q.filter(p=>inPoly(p.x,p.y,L.pts)).length/q.length:0},
 ap=l=>{if(!LK||!ident)return null;if(l.sg===undefined)l.sg=sigOf(l);return simSig(l.sg,ident)};
 if(!prev){
  if(LK&&seed){let b=null,bc=1e9;const near=Math.abs(seed.t-start)<.3;for(const l of c){const q=box(l),tc=torsoC(l),d=Math.hypot((tc.x-seed.c.x)*VW(),(tc.y-seed.c.y)*VH())/bs(q),sm=ap(l),cost=d*(near?1:.25)+(sm===null?0:2.2*(1-sm))-.5*vis(l)-(L?2*fo(l):0);if(cost<bc){bc=cost;b=l}}return b}
  if(L){let b=null,bc=1e9;for(const l of c){const q=box(l),d=Math.hypot(((q.x0+q.x1)/2-L.c.x)*VW(),((q.y0+q.y1)/2-L.c.y)*VH())/bs(q),cost=d-3*fo(l)-.5*vis(l);if(cost<bc){bc=cost;b=l}}return b}
  if(subj){let b=null,bd=9;for(const l of c){const d=skelDist(l,subj);if(d<bd){bd=d;b=l}}return b}
  return c.map(l=>[l,vis(l)*bs(box(l))]).sort((a,b)=>b[1]-a[1])[0][0]}
 const pb=bs(box(prev));let b=null,bc=1e9;
 for(const l of c){const d0=cd(l,prev)/pb,d=LK?Math.min(d0,cdS(l,prev,vel.x,vel.y)/pb):d0,sm=ap(l),
   sr=LK?.4*Math.max(0,Math.abs(Math.log(bs(box(l))/pb))-.3):0,app=sm===null?0:2*(1-sm),lim=miss>8?99:(sm!==null&&sm>.7?2.4:1.6);
  if(sm!==null&&sm<.15&&d>.5&&miss<=8)continue; // clearly a different person and not where ours should be
  const cost=d-.5*vis(l)-(miss>8?.8*fo(l):0)+sr+app;if(d<lim&&cost<bc){bc=cost;b=l}}
 return b}
function deswap(l,p){if(!p)return false;let a=0,b=0;for(const[i,j]of SWA){a+=dd(l[i],p[i])+dd(l[j],p[j]);b+=dd(l[j],p[i])+dd(l[i],p[j])}if(b<a*.6){for(const[i,j]of SW){const t=l[i];l[i]=l[j];l[j]=t}return true}return false}
// ---- crop around the focus: centred on the lasso centre (or the subject), widened to always include the whole subject
function roiFor(ref,mul=1){const f=(+$('fpad').value||35)/100,sb=ref?box(ref,true):null,L=lasso&&lassoNow(ref);let x0,x1,y0,y1;
 if(L){const C=L.c;let hx=Math.max(C.x-L.bb.x0,L.bb.x1-C.x)*(1+f),hy=Math.max(C.y-L.bb.y0,L.bb.y1-C.y)*(1+f);
  if(sb){hx=Math.max(hx,Math.max(C.x-sb.x0,sb.x1-C.x)*1.15);hy=Math.max(hy,Math.max(C.y-sb.y0,sb.y1-C.y)*1.15)}
  hx=Math.max(.06,hx*mul);hy=Math.max(.06,hy*mul);x0=C.x-hx;x1=C.x+hx;y0=C.y-hy;y1=C.y+hy}
 else if(sb){const px_=((sb.x1-sb.x0)*.4+.04)*mul,py_=((sb.y1-sb.y0)*.4+.04)*mul;x0=sb.x0-px_;x1=sb.x1+px_;y0=sb.y0-py_;y1=sb.y1+py_}
 else return null;
 if(!L){x0+=Math.min(0,vel.x)*1.5;x1+=Math.max(0,vel.x)*1.5;y0+=Math.min(0,vel.y)*1.5;y1+=Math.max(0,vel.y)*1.5}
 const q={x0:Math.max(0,x0),y0:Math.max(0,y0),x1:Math.min(1,x1),y1:Math.min(1,y1)};return q.x1-q.x0<.05||q.y1-q.y0<.05?null:q}
function runCrop(im,q,src,size,enh,mirror,mask){q=q||{x0:0,y0:0,x1:1,y1:1};const W=VW(),H=VH(),cw=(q.x1-q.x0)*W,ch=(q.y1-q.y0)*H,k=Math.min(size/Math.max(cw,ch),4);
 crop.width=Math.max(32,Math.round(cw*k));crop.height=Math.max(32,Math.round(ch*k));
 cx.save();cx.filter=enh?'contrast(1.4) saturate(1.3)':'none';if(mirror){cx.translate(crop.width,0);cx.scale(-1,1)}cx.drawImage(V,q.x0*W,q.y0*H,cw,ch,0,0,crop.width,crop.height);cx.restore();
 if(mask){cx.save();cx.fillStyle='#808080';cx.beginPath();cx.rect(0,0,crop.width,crop.height);mask.forEach(([x,y],j)=>{const X=(mirror?q.x1-x:x-q.x0)/(q.x1-q.x0)*crop.width,Y=(y-q.y0)/(q.y1-q.y0)*crop.height;j?cx.lineTo(X,Y):cx.moveTo(X,Y)});cx.closePath();cx.fill('evenodd');cx.restore()}
 return im.detect(crop).landmarks.map(l=>{const p=nm(l).map(p=>({x:q.x0+(mirror?1-p.x:p.x)*(q.x1-q.x0),y:q.y0+p.y*(q.y1-q.y0),v:p.v}));if(mirror)for(const[a,b]of SW){const t=p[a];p[a]=p[b];p[b]=t}return tag(p,src)})}
// ---- cross-checking
function agree(a,b){const s=bs(box(a));let t=0,n=0;for(const k of MAIN){if(a[k].v<.3||b[k].v<.3)continue;t+=dd(a[k],b[k]);n++}return n>=4?t/n/s:NaN}
function fuse(cs,tol,ref){if(cs.length<2)return{lm:cs[0]||null,n:cs.length,tot:cs.length};
 let best=null,bg=[];for(const a of cs){const g=cs.filter(b=>b===a||agree(a,b)<tol);if(!best||g.length>bg.length||(g.length===bg.length&&vis(a)>vis(best))){best=a;bg=g}}
 if(bg.length<2){const b=choose(cs,ref,0)||best;return{lm:tag(b.map(p=>({...p})),b.src),n:1,tot:cs.length,disp:true}}
 const sz=bs(box(best)),out=best.map((_,k)=>{let P=bg.map(l=>l[k]);if(P.length>=3){const mx=med(P.map(p=>p.x)),my=med(P.map(p=>p.y)),T=P.filter(p=>Math.hypot((p.x-mx)*VW(),(p.y-my)*VH())<tol*1.5*sz);if(T.length)P=T}
  let w=0,x=0,y=0,v=0;for(const p of P){const q=Math.max(.02,p.v)**2;w+=q;x+=p.x*q;y+=p.y*q;v+=p.v*q}return{x:x/w,y:y/w,v:v/w}});
 let t=0,c=0;for(const a of bg)for(const b of bg)if(a!==b){const d=agree(a,b);if(d>=0){t+=d;c++}}
 return{lm:tag(out,'multi'),n:bg.length,tot:cs.length,dis:c?t/c:NaN}}
// ---- advanced engine: YOLOX person detector + DWPose/RTMW (133 whole-body keypoints) on ONNX Runtime Web (WebGPU, WASM fallback)
const engine=()=>$('eng').value,ORTV='1.22.0',HF='https://huggingface.co/yzd-v/DWPose/resolve/main/',DW={W:288,H:384};
let ort=null,dwP=null,dwD=null,dwEP='';
// 133 COCO-WholeBody points -> the 33 MediaPipe landmarks the rest of the app uses (face = iBUG-68 at 23.., left hand 91.., right hand 112..)
const MAP33=[[0],[23+42],[42,43,44,45,46,47].map(n=>23+n),[23+45],[23+39],[36,37,38,39,40,41].map(n=>23+n),[23+36],[3],[4],[23+54],[23+48],[5],[6],[7],[8],[9],[10],[91+17],[112+17],[91+5],[112+5],[91+3],[112+3],[11],[12],[13],[14],[15],[16],[19],[22],[17],[20]];
function simccDecode(tx,ty,K,NX,NY){const r=[];for(let k=0;k<K;k++){let bx=0,vx=-1e9,by=0,vy=-1e9;for(let j=0;j<NX;j++){const v=tx[k*NX+j];if(v>vx){vx=v;bx=j}}for(let j=0;j<NY;j++){const v=ty[k*NY+j];if(v>vy){vy=v;by=j}}r.push({x:bx/2,y:by/2,s:Math.max(0,Math.min(vx,vy))})}return r}
const to33=k=>MAP33.map(ids=>{let x=0,y=0,s=0;for(const i of ids){x+=k[i].x;y+=k[i].y;s+=k[i].s}const n=ids.length;return{x:x/n,y:y/n,v:Math.min(1,s/n*1.1)}});
const iou=(a,b)=>{const w=Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0),h=Math.min(a.y1,b.y1)-Math.max(a.y0,b.y0);if(w<=0||h<=0)return 0;const i=w*h;return i/((a.x1-a.x0)*(a.y1-a.y0)+(b.x1-b.x0)*(b.y1-b.y0)-i)};
function yNms(c,t){c.sort((a,b)=>b.s-a.s);const k=[];for(const b of c)if(k.every(q=>iou(q,b)<t))k.push(b);return k}
let yG=null;const yGrid=()=>{if(yG)return yG;const g=[];for(const s of[8,16,32]){const n=640/s;for(let y=0;y<n;y++)for(let x=0;x<n;x++)g.push([x,y,s])}return yG=g};
// YOLOX output [1,8400,85]: raw (grid-relative) or already decoded; person = class 0
function yoloDecode(dat,N,C,r,W,H,thr){let mx=0;for(let i=0;i<N;i++)mx=Math.max(mx,dat[i*C+2]);const dec=mx>20,G=N===8400?yGrid():null;if(!dec&&!G)throw new Error('Unsupported detector output size');
 const c=[];for(let i=0;i<N;i++){const o=i*C,sc=dat[o+4]*dat[o+5];if(sc<thr)continue;let X,Y,w,h;if(dec){X=dat[o];Y=dat[o+1];w=dat[o+2];h=dat[o+3]}else{const[gx,gy,s]=G[i];X=(dat[o]+gx)*s;Y=(dat[o+1]+gy)*s;w=Math.exp(dat[o+2])*s;h=Math.exp(dat[o+3])*s}
  const b={x0:Math.max(0,(X-w/2)/r/W),y0:Math.max(0,(Y-h/2)/r/H),x1:Math.min(1,(X+w/2)/r/W),y1:Math.min(1,(Y+h/2)/r/H),s:sc};if(b.x1-b.x0>.02&&b.y1-b.y0>.05)c.push(b)}
 return yNms(c,.45)}
async function loadORT(){if(ort)return ort;for(const f of['ort.webgpu.min.mjs','ort.all.min.mjs','ort.min.mjs']){try{const m=await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORTV}/dist/${f}`);m.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORTV}/dist/`;m.env.wasm.numThreads=Math.min(4,navigator.hardwareConcurrency||2);ort=m;return ort}catch(e){console.warn('ORT',f,e)}}throw new Error('could not load ONNX Runtime Web')}
async function fetchModel(url,label){let cache=null;try{cache=await caches.open('vid2anim-models');const hit=await cache.match(url);if(hit)return new Uint8Array(await hit.arrayBuffer())}catch{}
 const r=await fetch(url);if(!r.ok)throw new Error(label+' download failed ('+r.status+')');const tot=+r.headers.get('content-length')||0,rd=r.body.getReader(),ch=[];let got=0;
 for(;;){const{done,value}=await rd.read();if(done)break;ch.push(value);got+=value.length;st(`Downloading ${label}: ${(got/1e6).toFixed(0)}${tot?' / '+(tot/1e6).toFixed(0):''} MB (first run only)`);$('prog').value=tot?got/tot:0}
 const buf=new Uint8Array(got);let o=0;for(const c of ch){buf.set(c,o);o+=c.length}try{await cache?.put(url,new Response(buf))}catch{}return buf}
async function mkSess(buf,label,dims){for(const ep of dwEP==='wasm'?[['wasm']]:[['webgpu','wasm'],['wasm']]){try{const s=await ort.InferenceSession.create(buf,{executionProviders:ep,graphOptimizationLevel:'all'});st(`Warming up ${label}…`);
  await s.run({[s.inputNames[0]]:new ort.Tensor('float32',new Float32Array(dims.reduce((a,b)=>a*b)),dims)});dwEP=ep[0];return s}catch(e){console.warn(label,ep,e);if(ep.length===1)throw e}}}
async function ensureDW(){await loadORT();
 if(!dwP)dwP=await mkSess(await fetchModel(HF+'dw-ll_ucoco_384.onnx','pose model'),'pose model',[1,3,DW.H,DW.W]);
 if(!dwD&&$('ddet').value==='yolox')dwD=await mkSess(await fetchModel(HF+'yolox_l.onnx','person detector'),'person detector',[1,3,640,640])}
const pcv=document.createElement('canvas'),pg=pcv.getContext('2d',{willReadFrequently:true}),dcv=document.createElement('canvas'),dg=dcv.getContext('2d',{willReadFrequently:true});pcv.width=DW.W;pcv.height=DW.H;dcv.width=dcv.height=640;
// top-down pose on a person box (normalized). o: mirror, enh, pad, mask. Returns 33 landmarks in frame coordinates.
async function dwPose(bx,o={}){const W=VW(),H=VH(),pad=o.pad||1.25,ccx=(bx.x0+bx.x1)/2*W,ccy=(bx.y0+bx.y1)/2*H,asp=DW.W/DW.H;
 let sw=Math.max(8,(bx.x1-bx.x0)*W*pad),sh=Math.max(8,(bx.y1-bx.y0)*H*pad);if(sw>sh*asp)sh=sw/asp;else sw=sh*asp;const sx=ccx-sw/2,sy=ccy-sh/2;
 pg.save();pg.fillStyle='#000';pg.fillRect(0,0,DW.W,DW.H);pg.filter=o.enh?'contrast(1.35) saturate(1.2)':'none';if(o.mirror){pg.translate(DW.W,0);pg.scale(-1,1)}pg.drawImage(V,sx,sy,sw,sh,0,0,DW.W,DW.H);pg.restore();
 if(o.mask){pg.save();pg.fillStyle='#808080';pg.beginPath();pg.rect(0,0,DW.W,DW.H);o.mask.forEach(([x,y],j)=>{let X=(x*W-sx)/sw*DW.W;if(o.mirror)X=DW.W-X;const Y=(y*H-sy)/sh*DW.H;j?pg.lineTo(X,Y):pg.moveTo(X,Y)});pg.closePath();pg.fill('evenodd');pg.restore()}
 const d=pg.getImageData(0,0,DW.W,DW.H).data,n=DW.W*DW.H,f=new Float32Array(3*n);for(let i=0;i<n;i++){f[i]=(d[4*i]-123.675)/58.395;f[n+i]=(d[4*i+1]-116.28)/57.12;f[2*n+i]=(d[4*i+2]-103.53)/57.375}
 const res=await dwP.run({[dwP.inputNames[0]]:new ort.Tensor('float32',f,[1,3,DW.H,DW.W])});let tx=null,ty=null;
 for(const nme of dwP.outputNames){const t=res[nme];if(t.dims[2]===DW.W*2)tx=t;else if(t.dims[2]===DW.H*2)ty=t}
 if(!tx||!ty||tx.dims[1]<133)throw new Error('unexpected pose model output');
 const k=simccDecode(tx.data,ty.data,tx.dims[1],tx.dims[2],ty.dims[2]).map(p=>{let u=p.x/DW.W;if(o.mirror)u=1-u;return{x:(sx+u*sw)/W,y:(sy+p.y/DW.H*sh)/H,s:p.s}});
 const lm=to33(k);if(o.mirror)for(const[a,b]of SW){const t=lm[a];lm[a]=lm[b];lm[b]=t}return lm}
async function yoloBoxes(thr=.3,maxN=4){if(!dwD)return[];const W=VW(),H=VH(),r=Math.min(640/W,640/H);dg.fillStyle='rgb(114,114,114)';dg.fillRect(0,0,640,640);dg.drawImage(V,0,0,Math.round(W*r),Math.round(H*r));
 const d=dg.getImageData(0,0,640,640).data,n=640*640,f=new Float32Array(3*n);for(let i=0;i<n;i++){f[i]=d[4*i];f[n+i]=d[4*i+1];f[2*n+i]=d[4*i+2]}
 const res=await dwD.run({[dwD.inputNames[0]]:new ort.Tensor('float32',f,[1,3,640,640])}),t=res[dwD.outputNames[0]];return yoloDecode(t.data,t.dims[1],t.dims[2],r,W,H,thr).slice(0,maxN)}
// person box carried over from the previous frame's pose, leaning in the direction of motion
function trackBox(l,useVel=true){const b=box(l,true),vx=useVel?vel.x:0,vy=useVel?vel.y:0,m=useVel?camUnc:0;return{x0:Math.max(0,b.x0+Math.min(0,vx)*1.2-m),y0:Math.max(0,b.y0+Math.min(0,vy)*1.2-m),x1:Math.min(1,b.x1+Math.max(0,vx)*1.2+m),y1:Math.min(1,b.y1+Math.max(0,vy)*1.2+m)}}
// one frame with the advanced engine. detector runs on the first frame, when lost, and every few frames; in between the box follows the pose (like a pose tracker)
async function detectDW(i,prev,miss){const lvl=$('smart').checked?$('thr').value:'off',tol=tolV(),np=Math.max(2,+$('np').value||1);
 const tb=prev?trackBox(prev):null,every={off:8,fast:8,care:5,max:2}[lvl],needDet=$('ddet').value==='yolox'&&(!prev||miss>0||i%every===0||vis(prev)<.5||camFast);
 const dets=needDet?await yoloBoxes(.3,np):[],mask=lasso&&$('lmask').checked?maskPoly(prev):null;let seen=dets.length;
 const props=[];if(tb)props.push({b:tb,s:'track'});for(const d of dets)if(!tb||lvl==='max'||iou(d,tb)<.6)props.push({b:d,s:'det'});
 if(!props.length)props.push({b:roiFor(prev)||{x0:0,y0:0,x1:1,y1:1},s:'ctx'});
 const A0=[];for(const p of props.slice(0,np+1)){const l=await dwPose(p.b,{mask});if(l){l.bx=p.b;A0.push(tag(l,'dw-'+p.s))}}seen+=A0.length;
 const A=choose(A0,prev,miss),done=(lm,o)=>({lm,src:lm?.src||'dw',rej:!lm&&seen>0,n:1,tot:lm?1:0,...o});
 if(lvl==='off')return done(A);
 if(lvl==='fast'){let b=A;if(!b||vis(b)<.4){const ex=[],bxs=(await yoloBoxes(.15,np)).concat(tb?[tb]:[]);for(const q of bxs){const l=await dwPose(q,{mask,enh:true,pad:1.4});if(l)ex.push(tag(l,'dw-retry'))}seen+=ex.length;b=choose(A0.concat(ex),prev,miss)||b}return done(b,{resc:!!b&&b!==A})}
 const cs=[],ref=prev||A,bx=A?.bx||tb||props[0].b;if(A)cs.push(A);
 const add=async(o,src,b=bx)=>{const l=await dwPose(b,{mask,...o});if(!l)return;seen++;const c=choose([l],ref,miss);if(c)cs.push(tag(c,src))};
 await add({mirror:true},'dw-mirror');
 const g=cs.length>=2?agree(cs[0],cs[1]):NaN;
 if(lvl==='max'||!(g<tol)){await add({enh:true,pad:1.5},'dw-enh');await add({mirror:true,enh:true,pad:1.1},'dw-mirror-enh');
  if(lvl==='max'){await add({pad:1.8},'dw-wide');const q=roiFor(ref);if(q)await add({},'dw-ctx',q);for(const p of props)if(p.s==='det'&&p.b!==bx)await add({},'dw-det',p.b)}}
 if(!cs.length)await add({enh:true,pad:1.4},'dw-full',{x0:0,y0:0,x1:1,y1:1});
 if(ref)for(const c of cs)deswap(c,ref);
 const F=fuse(cs,tol,ref);return{lm:F.lm,src:F.lm?.src||'dw',rej:!F.lm&&seen>0,n:F.n,tot:F.tot,dis:F.dis,disp:F.disp,resc:!!F.lm&&(!A||vis(A)<.6)}}
async function dwGather(ref,mask){const cs=[],bx=ref?trackBox(ref,false):{x0:0,y0:0,x1:1,y1:1};
 const add=async(b,o,src)=>{const l=await dwPose(b,{mask,...o});if(!l)return;const c=choose([l],ref,0);if(c)cs.push(tag(c,src))};
 await add(bx,{},'dw');await add(bx,{mirror:true},'dw-mirror');await add(bx,{enh:true,pad:1.5},'dw-enh');await add(bx,{mirror:true,enh:true,pad:1.1},'dw-mirror-enh');await add(bx,{pad:1.8},'dw-wide');
 if($('ddet').value==='yolox')for(const d of await yoloBoxes(.25,3))await add(d,{},'dw-det');
 return cs}
async function getLM(mode){const np=+$('np').value||1,key=[$('mdl').value,$('cd').value,$('cp').value,$('ct').value,np,delegate].join();
 if(LMs.key!==key){for(const m of['VIDEO','IMAGE']){try{LMs[m]?.close()}catch{}}LMs={key};tsBase=0}
 if(!LMs[mode])LMs[mode]=await PoseLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:MURL($('mdl').value),delegate},runningMode:mode,numPoses:np,minPoseDetectionConfidence:+$('cd').value,minPosePresenceConfidence:+$('cp').value,minTrackingConfidence:+$('ct').value});return LMs[mode]}
// one frame. off = single video pass; fast = retry only when lost; care = 2 independent passes that must agree (+ tie-breakers); max = 5+ passes incl. mirror/enhanced/wide
async function detect(i,prev,miss){if(engine()==='dw')return detectDW(i,prev,miss);const lvl=$('smart').checked?$('thr').value:'off',tol=tolV(),vm=await getLM('VIDEO');
 const all=vm.detectForVideo(V,tsBase+Math.round(i*1000/fps)).landmarks.map(l=>tag(nm(l),'full'));let seen=all.length;const A=choose(all,prev,miss);
 const done=(lm,o)=>({lm,src:lm?.src||'full',rej:!lm&&seen>0,n:1,tot:lm?1:0,...o});
 if(lvl==='off')return done(A);
 const im=await getLM('IMAGE'),ref=prev||A;
 if(lvl==='fast'){let b=A;if(!b||vis(b)<.6){const q=roiFor(ref),ex=q?runCrop(im,q,'roi',512):[];seen+=ex.length;b=choose(all.concat(ex),prev,miss);
   if(!b){const e2=runCrop(im,null,'enh',1024,true);seen+=e2.length;b=choose(all.concat(ex,e2),prev,miss)}}return done(b,{resc:!!b&&b!==A})}
 const cs=[],mask=lasso&&$('lmask').checked?maskPoly(ref):null,q=roiFor(ref);if(A)cs.push(A);
 const add=(list,src)=>{seen+=list.length;const b=choose(list,ref,miss);if(b)cs.push(tag(b,src))};
 add(runCrop(im,q,'roi',512,false,false,mask),'roi');
 const g=cs.length>=2?agree(cs[0],cs[1]):NaN;
 if(lvl==='max'||!(g<tol)){add(runCrop(im,q,'roi-mirror',768,false,true,mask),'roi-mirror');add(runCrop(im,q,'roi-enh',1024,true,false,mask),'roi-enh');
  if(lvl==='max'){const qw=roiFor(ref,1.8);add(runCrop(im,qw,'wide',640,false,false,null),'wide');add(runCrop(im,qw,'wide-enh',896,true,false,null),'wide-enh')}}
 if(!cs.length){add(runCrop(im,null,'full-enh',1024,true),'full-enh');if(!cs.length)add(runCrop(im,null,'full-img',1280),'full-img')}
 if(ref)for(const c of cs)deswap(c,ref);
 const F=fuse(cs,tol,ref);return{lm:F.lm,src:F.lm?.src||'full',rej:!F.lm&&seen>0,n:F.n,tot:F.tot,dis:F.dis,disp:F.disp,resc:!!F.lm&&(!A||vis(A)<.6)}}
// ---- verify sweeps: find frames that look wrong and re-measure them with every method
function boneRatios(lm){const s=bs(box(lm));return BONES.map(([a,b])=>lm[a].v>.5&&lm[b].v>.5?dd(lm[a],lm[b])/s:NaN)}
function buildBR(){const cols=BONES.map(()=>[]);for(const f of frames)if(f.lm&&vis(f.lm)>.5)boneRatios(f.lm).forEach((r,j)=>{if(r>=0)cols[j].push(r)});
 BR=cols.map(c=>{if(c.length<5)return NaN;c.sort((a,b)=>a-b);return c[Math.floor(c.length*.8)]})}
const stretch=lm=>{if(!BR)return 1;let m=1;boneRatios(lm).forEach((r,j)=>{if(r>=0&&BR[j]>0)m=Math.max(m,r/BR[j])});return m};
function devOf(i,lm){const nb=[],ci0=ccOf(i);for(const j of[i-2,i-1,i+1,i+2])if(frames[j]?.lm){const cj=ccOf(j);nb.push(shiftLM(frames[j].lm,ci0[0]-cj[0],ci0[1]-cj[1]))}if(nb.length<2)return NaN;const s=bs(box(lm));let t=0,n=0;
 for(const k of MAIN){if(lm[k].v<.3)continue;const q=nb.filter(l=>l[k].v>.3);if(q.length<2)continue;t+=Math.hypot((lm[k].x-med(q.map(l=>l[k].x)))*VW(),(lm[k].y-med(q.map(l=>l[k].y)))*VH())/s;n++}return n>=4?t/n:NaN}
const bad=(i,lm)=>{const d=devOf(i,lm);return(isNaN(d)?0:d)+Math.max(0,stretch(lm)-1.4)*.3+(1-vis(lm))*.1};
function suspects(){buildBR();const th=Math.max(.08,tolV()*1.7),out=[];
 frames.forEach((f,i)=>{if(!f.lm){out.push([i,'missing']);return}const r=[],fl=f.flag||'';
  if(/jump/.test(fl))r.push('jump');if(/low confidence/.test(fl))r.push('low confidence');if(/disputed/.test(fl))r.push('passes disagree');
  if(devOf(i,f.lm)>th)r.push('off-trend');if(stretch(f.lm)>1.4)r.push('stretched limb');if(r.length)out.push([i,r.join(', ')])});return out}
async function recheck(i,why){const f=frames[i];await seek(start+i/fps);const c0=ccOf(i);camCum={x:c0[0],y:c0[1]};
 const nb=[i-1,i+1].filter(j=>frames[j]?.lm).map(j=>{const cj=ccOf(j);return shiftLM(frames[j].lm,c0[0]-cj[0],c0[1]-cj[1])});let ref=null;
 if(nb.length===2)ref=nb[0].map((p,k)=>({x:(p.x+nb[1][k].x)/2,y:(p.y+nb[1][k].y)/2,v:Math.min(p.v,nb[1][k].v)}));else if(nb.length===1)ref=nb[0];else if(f.lm)ref=f.lm;
 const dw=engine()==='dw',im=dw?null:await getLM('IMAGE'),cs=[],mask=lasso&&$('lmask').checked?maskPoly(ref):null,tol=tolV(),q=dw?null:roiFor(ref),qw=dw?null:roiFor(ref,1.8);
 if(f.lm)cs.push(tag(f.lm.map(p=>({...p})),'orig'));
 if(dw)cs.push(...await dwGather(ref,mask));else{const add=(list,src)=>{const b=choose(list,ref,0);if(b)cs.push(tag(b,src))};
 add(im.detect(V).landmarks.map(l=>tag(nm(l),'full-img')),'full-img');
 add(runCrop(im,q,'roi',512,false,false,mask),'roi');add(runCrop(im,q,'roi-mirror',768,false,true,mask),'roi-mirror');add(runCrop(im,q,'roi-enh',1024,true,false,mask),'roi-enh');add(runCrop(im,qw,'wide',640,false,false,null),'wide')}
 if(ref)for(const c of cs)deswap(c,ref);
 rep.rechecked++;const F=fuse(cs,tol,ref);
 if(!F.lm){if(!f.lm)f.flag='missing';return false}
 const nl=F.lm.map(p=>({...p}));
 if(!f.lm||bad(i,nl)<bad(i,f.lm)*.8){f.lm=nl;f.src='verified';f.n=F.n;rep.fixed++;const d=devOf(i,nl);f.flag=d>Math.max(.08,tol*1.7)||stretch(nl)>1.4?'review: '+why:'';return true}
 f.flag='review: '+why;return false}
async function verifySweeps(rounds){for(let r=1;r<=rounds&&!stop;r++){const sus=suspects().slice(0,400);if(!sus.length)break;
 for(let k=0;k<sus.length&&!stop;k++){st(`Double-check sweep ${r}/${rounds}: re-measuring frame ${sus[k][0]} (${k+1}/${sus.length}) · ${sus[k][1]}`);$('prog').value=(k+1)/sus.length;await recheck(sus[k][0],sus[k][1]);await new Promise(r=>setTimeout(r))}}}
const seek=t=>new Promise(r=>{t=Math.min(t,V.duration);let d=false;const fin=()=>{if(d)return;d=true;V.onseeked=null;r()};
 if(Math.abs(V.currentTime-t)<1e-4)return setTimeout(fin,0);
 V.onseeked=()=>{if(V.requestVideoFrameCallback){V.requestVideoFrameCallback(fin);setTimeout(fin,150)}else setTimeout(fin,50)};V.currentTime=t});
async function analyze(){if(!V.src)return st('Load a video first.');picking=false;O.style.cursor='';if(engine()==='mp'&&!fs)return st('MediaPipe model not ready yet.');
 stop=false;V.pause();fps=+$('fps').value;start=+$('t0').value||0;const end=Math.min(+$('t1').value||V.duration,V.duration),n=Math.floor((end-start)*fps)+1;if(n<2)return st('Clip too short.');
 const lvl=$('smart').checked?$('thr').value:'off',rounds={off:0,fast:0,care:1,max:2}[lvl];
 if(engine()==='dw'){$('analyze').disabled=true;try{await ensureDW();st('Advanced tracker ready ('+dwEP+').')}catch(e){console.error(e);$('analyze').disabled=false;$('eng').value='mp';return st('Advanced tracker could not load ('+e.message+'). It needs internet access to download its models. Switched to MediaPipe: press Analyze to continue with that.')}}
 $('analyze').disabled=true;$('cancel').disabled=false;frames=[];CORR=[];undo=[];redo=[];rep={held:0,checked:0,disputed:0,rescued:0,swaps:0,rejected:0,missing:0,rechecked:0,fixed:0};
 let prev=null,miss=0,pc_=null,acc={x:0,y:0},camPeak=0;const camOn=!!$('cam')&&$('cam').checked;anchor=null;vel={x:0,y:0};ident=seed&&seed.sig?copySig(seed.sig):null;lastPx=null;camPrevG=null;camCum={x:0,y:0};camUnc=0;camFast=false;const t0=performance.now();
 try{for(let i=0;i<n&&!stop;i++){await seek(start+i/fps);
  const cam=camOn?camStep(prev):{x:0,y:0,c:0};acc.x+=cam.x;acc.y+=cam.y;camCum.x+=cam.x;camCum.y+=cam.y;const cs_=Math.max(Math.abs(cam.x),Math.abs(cam.y));camPeak=Math.max(camPeak,cs_);camFast=cs_>.012;camUnc=Math.min(.12,Math.hypot(cam.x,cam.y)*.7);
  const prevS=prev&&(acc.x||acc.y)?shiftLM(prev,acc.x,acc.y):prev; // where the last known pose should be now that the camera has moved
  const held=$('held').checked&&isHeld(null),hd=held&&prevS,r=hd?{lm:prevS.map(p=>({...p})),src:'held',n:1,held:true}:await detect(i,prevS,miss),lm=r.lm;let flag='';if(hd)rep.held++;
  if(lm){if(deswap(lm,prevS)){rep.swaps++;flag+='L/R swap '}if(prevS&&cd(lm,prevS)/bs(box(prevS))>.6)flag+='jump ';else if(vis(lm)<.5)flag+='low confidence ';
   if(r.disp){flag+='disputed ';rep.disputed++}else if(r.n>=2)rep.checked++;if(r.resc)rep.rescued++;
   const c=torsoC(lm);if(prev)vel={x:c.x-pc_.x-acc.x,y:c.y-pc_.y-acc.y};pc_=c;if(!anchor)anchor={c,s:bs(box(lm,true))};learnIdent(lm,r,flag);prev=lm;miss=0;acc={x:0,y:0}}
  else{miss++;flag='missing ';if(r.rej)rep.rejected++;vel={x:vel.x*.5,y:vel.y*.5}}
  frames.push({t:i/fps,lm:lm?lm.map(p=>({...p})):null,src:r.src,flag:flag.trim(),n:r.n,dis:r.dis,...(camOn?{cc:[+camCum.x.toFixed(5),+camCum.y.toFixed(5)]}:{})});$('prog').value=(i+1)/n;
  if(i%2==0){const el=(performance.now()-t0)/1000;st(`Tracking ${i+1}/${n} (${lvl}) · ~${Math.round(el/(i+1)*(n-i-1))}s left${rounds?' + verify sweeps':''}`);await new Promise(r=>setTimeout(r))}}
  if(!stop&&rounds>0&&frames.length>2)await verifySweeps(rounds)}
 catch(e){console.error(e);st('Tracking error: '+e.message)}
 camCum={x:0,y:0};camUnc=0;camFast=false;tsBase+=Math.round(n*1000/fps)+1000;$('analyze').disabled=false;$('cancel').disabled=true;if(!frames.length)return;rep.missing=frames.filter(f=>!f.lm).length;
 meta={...meta,w:V.videoWidth,h:V.videoHeight,duration:V.duration,fps,start,end};compute();view=[0,Math.max(1,frames.length-1)];
 const note=lasso&&Math.abs(lasso.t-start)>.3?` · Lasso was drawn at ${lasso.t.toFixed(2)}s, not the clip start: redraw it on the first frame for best focus.`:'';
 st(`${stop?'Cancelled, partial. ':'Done. '}${frames.length} frames · ${rep.checked} confirmed by 2+ agreeing passes · ${rep.rescued} rescued by retry · ${rep.swaps} L/R swaps fixed · ${rep.rechecked} re-measured in verify sweeps (${rep.fixed} corrected) · ${rep.rejected} wrong-subject jumps rejected · ${rep.held||0} repeated frames reused · ${rep.missing} untracked${camOn&&camPeak>.004?` · camera shake compensated (peak ${(camPeak*100).toFixed(1)}% of the frame in one step)`:''}${note}`);V.currentTime=start}
async function init(){try{fs=await FilesetResolver.forVisionTasks(WASM)}catch(e){return st('Could not load MediaPipe (internet / http required).')}
 for(const del of['GPU','CPU']){delegate=del;try{await getLM('VIDEO');st(`Model ready (${del}). Upload a video.`);return}catch(e){console.warn(e)}}st('Model failed to load.')}
// ---- UI
const quality=id=>{const c=S[id]?.c;if(!c||!c.length)return'Missing';const m=c.reduce((a,b)=>a+b,0)/c.length;return m>.8?'Good':m>.5?'Moderate':m>.1?'Poor':'Missing'};
function parts(){const all=lvIds(),on=act();if(!on.includes(sel))sel=on[0]||all[0];const g=[['Head & torso',id=>!D[id].side],['Arms',id=>/^[LR](sh|ua|fa|wr|hand)$/.test(id)],['Legs',id=>/^[LR](th|sn|knee|foot|ank)$/.test(id)]];
 $('parts').innerHTML=g.map(([t,f])=>`<h5>${t}</h5>`+all.filter(f).map(id=>{const o=!OFF.has(id);return`<div class="pr ${D[id].side||''} ${o?'':'off'}"><input type=checkbox data-t=${id} ${o?'checked':''} title="Track this part" aria-label="Track ${D[id].n}"><button data-id=${id} class="${D[id].side||''} ${id===sel?'on':''}" ${o?'':'disabled'}>${D[id].n}${frames.length&&o?' · '+quality(id):''}</button></div>`}).join('')).join('')}
const LEG=id=>/^[LR](th|sn|knee|foot|ank)$/.test(id),PRE={all:()=>1,upper:id=>!LEG(id),lower:id=>LEG(id)||id==='pelvis',left:id=>D[id].side!=='R',right:id=>D[id].side!=='L'};
function offChanged(){applyOff();dirty=true;lastT=-1;if(frames.length)st(OFF.size?`${OFF.size} part(s) removed. Press Analyze to re-track without them.`:'All body parts are tracked.')}
$('parts').onchange=e=>{const t=e.target.dataset.t;if(!t)return;if(e.target.checked)OFF.delete(t);else if(act().length>1)OFF.add(t);else{st('Keep at least one body part on.');dirty=true;return}offChanged()};
$('pchips').onclick=e=>{const f=PRE[e.target.dataset.p];if(!f)return;OFF.clear();for(const id in D)if(!f(id))OFF.add(id);if(!act().length)OFF.clear();offChanged()};
function probList(){const p=frames.map((f,i)=>[i,f.flag]).filter(x=>x[1]);$('probs').innerHTML=p.length?p.slice(0,60).map(([i,f])=>`<button data-f=${i}>f${i}: ${f}${CORR.some(c=>c.i===i)?' ✓':''}</button>`).join('')+(p.length>60?`<small>+${p.length-60} more</small>`:''):'<small>None flagged.</small>'}
$('probs').onclick=e=>{const f=e.target.dataset.f;if(f!==undefined){V.pause();V.currentTime=start+ +f/fps}};
function info(){const i=ci(),s=S[sel];if(i<0||!s){$('info').innerHTML='<p class=m>Upload a video and press Analyze.</p>';return}
 const u=px()?'px':'norm',rows=[['Rotation','°',dv('rot',s.rot[i])],['X',u,s.x[i]*U('x')],['Y',u,s.y[i]*U('y')],['Joint angle','°',s.jt[i]],['Ang. velocity','°/s',s.vrot[i]],['Pos. velocity',u+'/s',s.vpos[i]*U('vpos')],['Length',u,s.len[i]*U('len')],['Confidence','',s.c[i]]];
 const k=KS(),nk=k&&k.reduce((a,b)=>Math.abs(b.i-i)<Math.abs(a.i-i)?b:a),all=`${D[sel].n} f${i} t=${(i/fps).toFixed(3)}s `+rows.map(r=>r[0]+'='+f3(r[2])).join(' '),d=D[sel];
 $('info').innerHTML=`<div class=kv><span>${d.n}</span><b>${frames[i].lm?quality(sel):'Missing'}</b></div><div class=kv><span>Joints</span><span>${[...d.a,...d.b].filter((v,j,a)=>a.indexOf(v)===j).slice(0,4).map(n=>NAMES[n]).join(' → ')}</span></div>`+rows.map(r=>`<div class=kv><span>${r[0]}</span><span>${f3(r[2])} ${r[1]}<button data-c="${f3(r[2])}">⧉</button></span></div>`).join('')+
 `<div class=kv><span>Frame / time</span><span>${i} / ${(i/fps).toFixed(3)}s</span></div><div class=kv><span>Nearest KF (${prop})</span><span>${nk?'f'+nk.i:'–'}</span></div><button data-c="${all}">Copy all values</button>`}
function table(){const k=KS();if(!k){$('kft').innerHTML='<p class=m>No keyframes for this property.</p>';return}
 $('kft').innerHTML='<table><tr><th>Frame<th>Time<th>Value<th>Ease<th>Why</tr>'+k.map((q,j)=>`<tr data-j=${j} class="${j===selKf?'s':''}"><td><span class=dm>◇</span> ${q.i}<td>${(q.i/fps).toFixed(2)}<td>${f3(dv(prop,q.v*U(prop)))}<td>${easeName(q.e)}<td>${q.why.replace('Significant ','').replace('Added: a segment cannot turn back on itself','added: reversal')}</tr>`).join('')+'</table>'}
// easing editor with draggable handles (y range -1..2)
const EX=x=>20+x*120,EY=y=>120-(y+1)*34;
function ezEd(){const k=KS(),a=k&&k[selKf],ok=a&&selKf<k.length-1;for(const i of['ezs','e1','e2','e3','e4','ffit'])$(i).disabled=!ok;
 ec.clearRect(0,0,160,140);if(!ok){$('ezn').textContent='Select a keyframe (not the last) to edit the curve to the next one.';return}
 $('ezs').value=easeName(a.e);const arr=Array.isArray(a.e);[1,2,3,4].forEach(n=>{$('e'+n).disabled=!arr;$('e'+n).value=arr?a.e[n-1]:''});
 ec.fillStyle='#ffffff14';ec.fillRect(EX(0),EY(1),120,34);ec.strokeStyle='#8890b0';ec.strokeRect(EX(0),EY(1),120,34);ec.strokeRect(EX(0),EY(0),120,0);
 const b=k[selKf+1],d=b.v-a.v,y=S[sel][prop];ec.fillStyle='#ffffff77';if(Math.abs(d)>1e-9)for(let i=a.i+1;i<b.i;i++)ec.fillRect(EX((i-a.i)/(b.i-a.i))-1,EY((y[i]-a.v)/d)-1,2,2);
 ec.strokeStyle='#3de8ff';ec.lineWidth=2;ec.beginPath();for(let q=0;q<=60;q++){const x=q/60,yy=ez(a.e,x);q?ec.lineTo(EX(x),EY(yy)):ec.moveTo(EX(x),EY(yy))}ec.stroke();ec.lineWidth=1;
 if(arr){ec.strokeStyle='#fff';ec.fillStyle='#fff';for(const[x0,y0,hx,hy]of[[0,0,a.e[0],a.e[1]],[1,1,a.e[2],a.e[3]]]){ec.beginPath();ec.moveTo(EX(x0),EY(y0));ec.lineTo(EX(hx),EY(hy));ec.stroke();ec.beginPath();ec.arc(EX(hx),EY(hy),5,0,7);ec.fill()}}
 $('ezn').textContent='Drag the white handles. Dots = real motion. Time (x) is clamped to 0–1; value (y) can overshoot.'}
let hd=null;
EC.onpointerdown=e=>{const a=KS()?.[selKf];if(!a||selKf>=KS().length-1)return;if(!Array.isArray(a.e))a.e=[.33,.33,.66,.66];const r=EC.getBoundingClientRect(),x=(e.clientX-r.left)*160/r.width,y=(e.clientY-r.top)*140/r.height;
 const d1=Math.hypot(x-EX(a.e[0]),y-EY(a.e[1])),d2=Math.hypot(x-EX(a.e[2]),y-EY(a.e[3]));hd=d1<d2?0:2;EC.setPointerCapture(e.pointerId);EC.onpointermove(e)};
EC.onpointermove=e=>{if(hd===null)return;const a=KS()[selKf],r=EC.getBoundingClientRect(),x=(e.clientX-r.left)*160/r.width,y=(e.clientY-r.top)*140/r.height;
 a.e[hd]=Math.round(Math.min(1,Math.max(0,(x-20)/120))*100)/100;a.e[hd+1]=Math.round(Math.min(2,Math.max(-1,(120-y)/34-1))*100)/100;dirty=true;lastT=-1};
EC.onpointerup=()=>hd=null;
$('ezs').onchange=()=>{const a=KS()?.[selKf],v=$('ezs').value;if(!a||v==='Custom')return;a.e=Array.isArray(EZ[v])?[...EZ[v]]:EZ[v];dirty=true;lastT=-1};
for(const n of[1,2,3,4])$('e'+n).onchange=()=>{const a=KS()?.[selKf];if(!a||!Array.isArray(a.e))return;a.e=[1,2,3,4].map((m,q)=>{const v=+$('e'+m).value;return q%2?Math.min(2,Math.max(-1,v)):Math.min(1,Math.max(0,v))});dirty=true;lastT=-1};
$('ffit').onclick=()=>{if(selKf>=0){freeFit(sel,prop,selKf);dirty=true;lastT=-1}};
$('ffitA').onclick=()=>{const k=KS();if(!k)return;st('Free-fitting…');for(let j=0;j<k.length-1;j++)freeFit(sel,prop,j);st('Free-fit done for '+PROPS[prop]);dirty=true;lastT=-1};
// ---- overlay + corrections
const PCOL=['#ffe45e','#3dffa6','#ff8a4c','#3de8ff','#fc01e6','#b48cff'];
function drawPick(w,h){const lw=Math.max(2,w/260);ox.save();ox.lineWidth=lw*1.4;ox.textAlign='center';ox.textBaseline='middle';ox.font='700 '+Math.round(w/45)+'px sans-serif';
 pickC.forEach((l,n)=>{const col=PCOL[n%PCOL.length];ox.strokeStyle=col;ox.globalAlpha=.9;ox.beginPath();for(const[a,b]of CONN){if(l[a].v<.3||l[b].v<.3)continue;ox.moveTo(l[a].x*w,l[a].y*h);ox.lineTo(l[b].x*w,l[b].y*h)}ox.stroke();
  const t=torsoC(l);ox.globalAlpha=1;ox.fillStyle=col;ox.beginPath();ox.arc(t.x*w,t.y*h,w/46,0,7);ox.fill();ox.fillStyle='#000';ox.fillText(String(n+1),t.x*w,t.y*h+1)});ox.restore()}
function overlay(){const w=VW(),h=VH();if(O.width!==w){O.width=w;O.height=h;$('stage').style.aspectRatio=w+'/'+h}ox.clearRect(0,0,w,h);
 if(subj){ox.save();ox.strokeStyle=C.brand;ox.shadowColor=C.brand;ox.shadowBlur=8;ox.lineWidth=3;ox.beginPath();ox.arc(subj.x*w,subj.y*h,w/40,0,7);ox.stroke();ox.beginPath();ox.arc(subj.x*w,subj.y*h,w/260,0,7);ox.fillStyle=C.brand;ox.fill();ox.restore()}
 if(seed&&lockOn()&&subj&&!picking){ox.save();ox.font='600 '+Math.round(w/60)+'px sans-serif';ox.fillStyle=C.brand;ox.textAlign='center';ox.fillText('LOCKED',subj.x*w,subj.y*h+w/40+w/50);ox.restore()}
 if(picking&&pickC.length&&Math.abs(V.currentTime-pickT)<.02)drawPick(w,h);
 const i=ci(),lm=EF[i];drawLassoOv(w,h,lm);if(i<0||!lm||!$('skel').checked)return;const lw=Math.max(2,w/300);ox.lineWidth=lw;
 for(const[a,b]of CONN){if(!ok(a)||!ok(b))continue;const m=Math.max(a,b);ox.strokeStyle=m<11?C.N:m%2?C.L:C.R;ox.globalAlpha=Math.min(lm[a].v,lm[b].v)<.5?.3:1;ox.beginPath();ox.moveTo(lm[a].x*w,lm[a].y*h);ox.lineTo(lm[b].x*w,lm[b].y*h);ox.stroke()}
 ox.font=Math.round(w/70)+'px sans-serif';lm.forEach((p,n)=>{if(!ok(n))return;ox.globalAlpha=Math.max(.25,p.v);ox.fillStyle=p.fixed?'#fff':n<11?C.N:n%2?C.L:C.R;ox.beginPath();ox.arc(p.x*w,p.y*h,lw*(p.fixed?2.5:1.5),0,7);ox.fill();if($('lbl').checked){ox.fillStyle='#fff';ox.fillText(NAMES[n],p.x*w+6,p.y*h)}});ox.globalAlpha=1;
 const d=D[sel],a=pt(lm,d.a),b=pt(lm,d.b);ox.strokeStyle=C.sel;ox.lineWidth=lw*2;ox.beginPath();ox.moveTo(a.x*w,a.y*h);ox.lineTo(b.x*w,b.y*h);ox.stroke();
 if($('trail').checked&&S[sel]){const tl=+$('tl').value||30,q=S[sel],s0=Math.max(0,i-tl);ox.strokeStyle=C.sel+'88';ox.lineWidth=lw;ox.beginPath();for(let j=s0;j<=i;j++)j==s0?ox.moveTo(q.x[j]*w,q.y[j]*h):ox.lineTo(q.x[j]*w,q.y[j]*h);ox.stroke()}}
const rel=e=>{const r=O.getBoundingClientRect();return{x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}};let cdrag=null,fdrag=null;const focusHit=p=>{const w=VW(),h=VH(),near=c=>Math.hypot((p.x-c.x)*w,(p.y-c.y)*h)<w*.03;if(subj&&near(subj))return'subj';const L=lasso&&lassoNow(EF[ci()]);return L&&near(L.c)?'lasso':null};
function moveFocus(k,dx,dy){if(k==='subj'){subj={x:Math.min(1,Math.max(0,subj.x+dx)),y:Math.min(1,Math.max(0,subj.y+dy))};if(seed)seed.c={x:seed.c.x+dx,y:seed.c.y+dy};return}lasso={...lasso,pts:lasso.pts.map(([x,y])=>[x+dx,y+dy]),c:{x:lasso.c.x+dx,y:lasso.c.y+dy}};lasso.bb=bbOf(lasso.pts)}
O.onpointerdown=e=>{const p=rel(e),i=ci();if(lassoing){lp=[[p.x,p.y]];drawingL=true;O.setPointerCapture(e.pointerId);lastT=-1;return}
 if(picking){picking=false;const ok_=pickC.length&&Math.abs(V.currentTime-pickT)<.02;
  if(pickC.length&&!ok_){pickC=[];lastT=-1;return st('The frame changed after the people were detected. Press Pick subject again.')}
  let k=-1,bd=.7;if(ok_)pickC.forEach((l,n)=>{const d=skelDist(l,p);if(d<bd){bd=d;k=n}});
  if(k>=0)lockPerson(pickC[k],k+1);else{subj=p;seed=null;ident=null;st('Subject set by click (no person lock: nothing was detected near the click). Drag the circle to move it. Press Analyze.')}
  pickC=[];lastT=-1;return}
 if(!$('corr').checked){const k=focusHit(p);if(k){fdrag={k,x:p.x,y:p.y};O.setPointerCapture(e.pointerId);O.style.cursor='grabbing';return}}
 if(!frames.length||i<0)return;
 if($('corr').checked&&EF[i]){let b=-1,bd=.05;EF[i].forEach((q,k)=>{const d=Math.hypot((q.x-p.x)*VW(),(q.y-p.y)*VH())/VW();if(d<bd){bd=d;b=k}});if(b>=0){cdrag=b;O.setPointerCapture(e.pointerId)}return}
 let b=null,bd=.06;for(const id of act()){const d=Math.hypot(S[id].x[i]-p.x,S[id].y[i]-p.y);if(d<bd){bd=d;b=id}}if(b){sel=b;selKf=-1;dirty=true}};
O.onpointermove=e=>{if(fdrag){const p=rel(e);moveFocus(fdrag.k,p.x-fdrag.x,p.y-fdrag.y);fdrag.x=p.x;fdrag.y=p.y;lastT=-1;return}if(drawingL){const p=rel(e),l=lp[lp.length-1];if(Math.hypot((p.x-l[0])*VW(),(p.y-l[1])*VH())>4){lp.push([p.x,p.y]);lastT=-1}return}if(cdrag===null){if(!lassoing&&!picking&&!$('corr').checked)O.style.cursor=focusHit(rel(e))?'grab':'';return}const p=rel(e),q=EF[ci()][cdrag];q.x=Math.min(1,Math.max(0,p.x));q.y=Math.min(1,Math.max(0,p.y));q.fixed=true;lastT=-1};
O.onpointerup=e=>{if(fdrag){fdrag=null;O.style.cursor='';lastT=-1;st('Focus moved. It stays there until you move or clear it. Press Analyze to track with it.');return}if(drawingL){drawingL=false;lassoing=false;finishLasso();return}if(cdrag===null)return;const i=ci(),q=EF[i][cdrag];undo.push(JSON.stringify(CORR));redo=[];CORR=CORR.filter(c=>!(c.i===i&&c.k===cdrag));CORR.push({i,k:cdrag,x:q.x,y:q.y,r:+$('rad').value||0});cdrag=null;compute();lastT=-1};
O.onpointercancel=()=>{fdrag=null};
$('undo').onclick=()=>{if(undo.length){redo.push(JSON.stringify(CORR));CORR=JSON.parse(undo.pop());compute();lastT=-1}};$('redo').onclick=()=>{if(redo.length){undo.push(JSON.stringify(CORR));CORR=JSON.parse(redo.pop());compute();lastT=-1}};
// detect every person on the current frame (numbered on the video so the right one can be clicked, even when they overlap)
async function peopleNow(){const out=[];
 if(engine()==='dw'){await ensureDW();let bx=await yoloBoxes(.25,6);if(!bx.length)bx=[{x0:0,y0:0,x1:1,y1:1}];for(const b of bx){const l=await dwPose(b,{});if(l&&vis(l)>.25)out.push(l)}}
 else{const im=await getLM('IMAGE');for(const l of im.detect(V).landmarks)out.push(nm(l))}
 const keep=[];for(const l of out.sort((a,b)=>vis(b)-vis(a)))if(keep.every(k=>cd(k,l)/bs(box(k))>.35))keep.push(l);
 return keep.sort((a,b)=>torsoC(a).x-torsoC(b).x)}
function lockPerson(l,n){const s=sigOf(l),t=torsoC(l);seed={lm:l.map(q=>({...q})),t:V.currentTime,sig:s,c:{x:t.x,y:t.y}};ident=copySig(s);subj={x:t.x,y:t.y};
 st(`Locked on person ${n}${s?' (clothing colours captured)':''}. Tracking will keep this person even when others come close. `+(Math.abs(V.currentTime-start)>.3?'Tip: this is not the clip start, tracking will find them by clothing and size.':'Press Analyze.'))}
$('pick').onclick=async()=>{if(!V.src)return st('Load a video first.');picking=true;lassoing=false;$('lasso').classList.remove('on');subj=null;seed=null;ident=null;pickC=[];lastT=-1;const t=V.currentTime;
 st('Finding the people on this frame…');try{const c=await peopleNow();if(!picking||Math.abs(V.currentTime-t)>.02)return;pickC=c;pickT=t}catch(e){console.warn(e);pickC=[]}
 lastT=-1;st(pickC.length>1?`Found ${pickC.length} people, numbered on the video. Click the one you want to lock onto.`:pickC.length===1?'Found 1 person. Click them to lock on.':'No skeletons found on this frame. Click the person anyway, or move to a clearer frame.')};
$('pclear').onclick=()=>{subj=null;seed=null;ident=null;pickC=[];picking=false;O.style.cursor='';lastT=-1;st('Subject lock cleared. The largest, clearest person (or the lasso) will be used.')};
// ---- graph (also used offscreen for PNG / reference sheet)
const PL=48,PB=18;
function drawGraph(cv,pr,vw,live,id=sel,mini=false){const w=cv.width,h=cv.height,g=cv.getContext('2d');g.fillStyle='#0d0f24';g.fillRect(0,0,w,h);const s=S[id]?.[pr];if(!s||!s.length)return null;
 const u=U(pr),a=Math.max(0,Math.floor(vw[0])),b=Math.min(s.length-1,Math.ceil(vw[1])),k=K[id]?.[pr],raw=RF[id]?.[pr],sr=$('rawc').checked&&raw;let mn=1e9,mx=-1e9;
 for(let i=a;i<=b;i++){mn=Math.min(mn,s[i]*u);mx=Math.max(mx,s[i]*u);if(sr){mn=Math.min(mn,raw[i]*u);mx=Math.max(mx,raw[i]*u)}}if(mx-mn<1e-9){mx+=1;mn-=1}const pd=(mx-mn)*.08;mn-=pd;mx+=pd;
 const X=i=>PL+(i-vw[0])/(vw[1]-vw[0]||1)*(w-PL-8),Y=v=>h-PB-(v-mn)/(mx-mn)*(h-PB-8);
 g.font='10px sans-serif';g.fillStyle='#8a8fc0';g.strokeStyle='#242850';for(let t=0;t<=4;t++){const v=mn+(mx-mn)*t/4;g.beginPath();g.moveTo(PL,Y(v));g.lineTo(w,Y(v));g.stroke();g.fillText(f3(v),2,Y(v)+3)}
 for(let t=0;t<=6;t++){const i=vw[0]+(vw[1]-vw[0])*t/6;g.fillText((i/fps).toFixed(2)+'s',X(i)-12,h-4)}
 const line=(c,wd,f)=>{g.strokeStyle=c;g.lineWidth=wd;g.beginPath();for(let i=a;i<=b;i++){const v=f(i);i===a?g.moveTo(X(i),Y(v)):g.lineTo(X(i),Y(v))}g.stroke();g.lineWidth=1};
 if(sr)line('#666',1,i=>raw[i]*u);line('#3de8ff',1.5,i=>s[i]*u);if(k)line('#ff9f43',2,i=>curve(k,i)*u);
 if(k)k.forEach((q,j)=>{const on=live&&id===sel&&j===selKf,r=mini?4:on?8:6,kx=X(q.i),ky=Y(q.v*u);g.beginPath();g.moveTo(kx,ky-r);g.lineTo(kx+r,ky);g.lineTo(kx,ky+r);g.lineTo(kx-r,ky);g.closePath();if(on){g.fillStyle='#3dffa655';g.fill()}g.strokeStyle=C.ok;g.lineWidth=on?2.5:1.5;g.stroke();g.lineWidth=1});
 if(live||mini){const c=ci();g.strokeStyle=C.sel;g.beginPath();g.moveTo(X(c),0);g.lineTo(X(c),h-PB);g.stroke();g.fillStyle=C.sel;if(!mini)g.fillText(f3(dv(pr,s[c]*u)),X(c)+4,10);
  if(!mini&&hov!==null&&hov>=0&&hov<s.length){g.strokeStyle='#ffffff66';g.beginPath();g.moveTo(X(hov),0);g.lineTo(X(hov),h-PB);g.stroke();g.fillStyle='#fff';g.fillText(`f${hov} ${(hov/fps).toFixed(2)}s  smooth ${f3(dv(pr,s[hov]*u))}${raw?'  raw '+f3(dv(pr,raw[hov]*u)):''}`,PL+4,h-PB-6)}}
 return{X,Y,u,mn,mx,w,h}}
const gi=e=>view[0]+(e.offsetX-PL)/(G.width-PL-8)*(view[1]-view[0]);let drag=null;
G.onmousedown=e=>{const m=G._m;if(!m)return;const k=KS();let hit=-1;if(k)k.forEach((q,j)=>{if(Math.hypot(m.X(q.i)-e.offsetX,m.Y(q.v*m.u)-e.offsetY)<9)hit=j});
 if(hit>=0){selKf=hit;drag={kf:hit};dirty=true}else drag={pan:view[0],x:e.offsetX,moved:false,i:gi(e)}};
G.onmousemove=e=>{const m=G._m;if(!m)return;if(!drag){hov=Math.round(gi(e));lastT=-1;return}const k=KS();
 if(drag.kf!==undefined&&k){const q=k[drag.kf],lo=drag.kf?k[drag.kf-1].i+1:0,hi=drag.kf<k.length-1?k[drag.kf+1].i-1:S[sel][prop].length-1;
  if(drag.kf===0)q.i=0;else if(drag.kf===k.length-1)q.i=S[sel][prop].length-1;else q.i=Math.max(lo,Math.min(hi,Math.round(gi(e))));q.v=(m.mn+(m.h-PB-e.offsetY)/(m.h-PB-8)*(m.mx-m.mn))/m.u;q.why='Manually edited';dirty=true;lastT=-1}
 else if(drag.pan!==undefined){const dx=(e.offsetX-drag.x)/(m.w-PL-8)*(view[1]-view[0]);if(Math.abs(e.offsetX-drag.x)>3)drag.moved=true;const sp=view[1]-view[0],n=S[sel][prop].length-1,a=Math.max(0,Math.min(n-sp,drag.pan-dx));view=[a,a+sp];lastT=-1}};
G.onmouseup=()=>{if(drag&&drag.pan!==undefined&&!drag.moved)V.currentTime=start+Math.max(0,Math.min(frames.length-1,Math.round(drag.i)))/fps;if(drag?.kf!==undefined)V.currentTime=start+KS()[drag.kf].i/fps;drag=null};G.onmouseleave=()=>{drag=null;hov=null;lastT=-1};
const zoom=(f,c)=>{if(!frames.length)return;const n=frames.length-1;c??=(view[0]+view[1])/2;let a=c-(c-view[0])*f,b=c+(view[1]-c)*f;if(b-a>n){a=0;b=n}if(b-a<4)return;view=[Math.max(0,a),Math.min(n,b)];lastT=-1};
G.onwheel=e=>{e.preventDefault();zoom(e.deltaY>0?1.25:.8,gi(e))};$('zin').onclick=()=>zoom(.7);$('zout').onclick=()=>zoom(1.4);$('zoomR').onclick=G.ondblclick=()=>{view=[0,Math.max(1,frames.length-1)];lastT=-1};
// ---- rig preview (reconstructed from data or from eased keyframes)
function preview(){if(!$('pvon').checked)return;const w=PV.width=480,h=PV.height=Math.round(480*VH()/VW()),i=ci();pc.fillStyle='#0d0f24';pc.fillRect(0,0,w,h);if($('pvbg').checked&&V.videoWidth){pc.globalAlpha=.35;pc.drawImage(V,0,0,w,h);pc.globalAlpha=1}
 if(i<0||!S.head)return;const W=VW(),H=VH(),off=(+$('off').value||0)*Math.PI/180,kf=$('pvs').value==='kf',skel=$('pvm').value==='skel';pc.lineCap='round';
 for(const id of act()){const d=D[id],val=p=>kf&&K[id]?.[p]?curve(K[id][p],i):S[id][p][i],mx=val('x')*w,my=val('y')*h,L=val('len')*H*w/W,an=val('rot')*Math.PI/180+off,dx=Math.cos(an)*L/2,dy=Math.sin(an)*L/2;
  pc.strokeStyle=id===sel?C.sel:d.side==='L'?C.L:d.side==='R'?C.R:'#cfd3d9';pc.lineWidth=skel?2:id==='head'?16:/tor|pelvis|neck/.test(id)?14:9;pc.beginPath();pc.moveTo(mx-dx,my-dy);pc.lineTo(mx+dx,my+dy);pc.stroke();
  if(skel){pc.fillStyle=pc.strokeStyle;pc.beginPath();pc.arc(mx-dx,my-dy,3,0,7);pc.arc(mx+dx,my+dy,3,0,7);pc.fill()}}}
$('pvon').onchange=()=>{$('pvw').hidden=!$('pvon').checked;lastT=-1};for(const i of['pvm','pvs','pvbg'])$(i).onchange=()=>lastT=-1;
// ---- export
const dl=(n,t,m)=>{const a=document.createElement('a');a.href=URL.createObjectURL(t instanceof Blob?t:new Blob([t],{type:m}));a.download=n;a.click()};
const kcsv=ids=>'part,property,frame,time_s,value,easing,reason\n'+ids.flatMap(id=>KP.flatMap(p=>(K[id]?.[p]||[]).map(q=>[D[id].n,p,q.i,(q.i/fps).toFixed(4),q.v*U(p),easeName(q.e),q.why].join(',')))).join('\n');
const fcsv=ids=>'part,frame,time_s,rotation,x,y,joint_angle,length,ang_vel,pos_vel,confidence\n'+ids.flatMap(id=>S[id].rot.map((_,i)=>[D[id].n,i,(i/fps).toFixed(4),...['rot','x','y','jt','len','vrot','vpos','c'].map(p=>S[id][p][i]*U(p))].join(','))).join('\n');
$('csv').onclick=()=>{if(!frames.length)return st('Analyze first.');const sc=$('xs').value,ids=sc.startsWith('sel')?[sel]:act();dl('vid2anim.csv',sc.endsWith('all')?fcsv(ids):kcsv(ids),'text/csv')};
const SET=['rig','mdl','smart','np','cd','cp','ct','mv','fps','sm','sP','sR','sJ','lf','preset','kR','kP','kMin','kMax','coord','off','wrap','t0','t1','xs','thr','dtol','lmask','fpad','eng','ddet','held','lock','cam','stab','stabT'];
const gs=()=>Object.fromEntries(SET.map(i=>[i,$(i).type==='checkbox'?$(i).checked:$(i).value])),ss=o=>{for(const i in o){if(!$(i))continue;if($(i).type==='checkbox')$(i).checked=o[i];else $(i).value=o[i]}};
$('json').onclick=()=>frames.length?dl('vid2anim_analysis.json',JSON.stringify({meta,settings:gs(),removedParts:[...OFF],landmarksRaw:frames,corrections:CORR,landmarksCorrected:EF,segments:D,rawSeries:RF,smoothedSeries:S,keyframes:K}),'application/json'):st('Analyze first.');
$('gpng').onclick=()=>{if(!S[sel])return st('Analyze first.');const c=document.createElement('canvas');c.width=1400;c.height=420;drawGraph(c,prop,[0,frames.length-1]);c.toBlob(b=>dl(`graph_${sel}_${prop}.png`,b))};
$('copyT').onclick=()=>{const k=KS();if(k)navigator.clipboard.writeText('frame,time,value,easing\n'+k.map(q=>[q.i,(q.i/fps).toFixed(3),q.v*U(prop),easeName(q.e)].join(',')).join('\n'))};
async function sheet(){if(!S[sel])return st('Analyze first.');st('Building reference sheet…');const n=frames.length,c=document.createElement('canvas');c.width=900;c.height=260;
 const tbl=p=>'<table><tr><th>Frame<th>Time (s)<th>Value<th>Easing (to next)<th>Why</tr>'+K[sel][p].map(q=>`<tr><td>${q.i}<td>${(q.i/fps).toFixed(3)}<td>${f3(dv(p,q.v*U(p)))}<td>${easeName(q.e)}${Array.isArray(q.e)?' ['+q.e.join(', ')+']':''}<td>${q.why}</tr>`).join('')+'</table>';
 const gh=KP.filter(p=>K[sel]?.[p]).map(p=>{drawGraph(c,p,[0,n-1]);return`<h3>${PROPS[p]}</h3><img src="${c.toDataURL('image/png')}" width=900>${tbl(p)}`}).join('');
 const k=K[sel].rot,pick=k.length>16?k.filter((_,j)=>j%Math.ceil(k.length/16)===0):k,tc=document.createElement('canvas'),tw=200,th=Math.round(tw*VH()/VW());tc.width=tw;tc.height=th;const t0=V.currentTime;let ft='';
 if(V.videoWidth)for(const q of pick){await seek(start+q.i/fps);tc.getContext('2d').drawImage(V,0,0,tw,th);ft+=`<figure><img src="${tc.toDataURL('image/jpeg',.7)}"><figcaption>f${q.i} · ${(q.i/fps).toFixed(2)}s · ${f3(dv('rot',q.v))}°</figcaption></figure>`}await seek(t0);
 dl(`reference_${sel}.html`,`<!doctype html><meta charset=utf-8><title>vid2anim ${D[sel].n}</title><style>body{font:13px sans-serif;margin:20px}table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #bbb;padding:3px 8px}figure{display:inline-block;margin:4px}</style><h1>vid2anim — ${D[sel].n}</h1><p>${meta.name||''} · ${fps} fps · rotation offset ${$('off').value}° · units: ${$('coord').value}</p><h2>Frames at rotation keyframes</h2>${ft}${gh}`,'text/html');st('Reference sheet saved (open it and print / save as PDF).')}
$('sheet').onclick=sheet;
// ---- sessions (IndexedDB)
const tx=(m,f)=>new Promise((res,rej)=>{const q=indexedDB.open('motiontrace',1)/* DB name kept so sessions saved before the rename still open */;q.onupgradeneeded=()=>q.result.createObjectStore('s',{keyPath:'name'});q.onerror=()=>rej(q.error);q.onsuccess=()=>{const t=q.result.transaction('s',m),r=f(t.objectStore('s'));t.oncomplete=()=>res(r.result);t.onerror=t.onabort=()=>rej(t.error)}});
const listS=async()=>{try{const k=await tx('readonly',s=>s.getAllKeys());$('sess').innerHTML=k.map(n=>`<option>${n}</option>`).join('')}catch{}};
$('save').onclick=async()=>{if(!frames.length)return st('Analyze first.');const name=prompt('Session name',meta.name||'session');if(!name)return;const o={name,meta,fps,start,frames,CORR,set:gs(),K,sel,prop,lasso,anchor,subj,seed,off:[...OFF]};
 try{await tx('readwrite',s=>s.put({...o,video:curFile}));st('Session saved with video.')}catch(e){try{await tx('readwrite',s=>s.put({...o,video:null}));st('Saved without the video (storage limit). Reselect the video when reopening.')}catch(e2){st('Save failed: '+e2.message)}}listS()};
$('open').onclick=async()=>{const n=$('sess').value;if(!n)return;const o=await tx('readonly',s=>s.get(n));if(!o)return;ss(o.set);OFF.clear();(o.off||[]).forEach(id=>OFF.add(id));applyOff();subj=o.subj||null;seed=o.seed||null;ident=null;meta=o.meta;fps=o.fps;start=o.start;frames=(o.frames||[]).map(({cm,cb,...f})=>f);CORR=o.CORR||[];lasso=o.lasso||null;anchor=o.anchor||null;undo=[];redo=[];sel=o.sel;prop=o.prop;$('prop').value=prop;view=[0,Math.max(1,frames.length-1)];
 const fin=()=>{compute();K=o.K;selKf=-1;dirty=true;V.currentTime=start;st('Session opened.')};
 if(o.video){curFile=o.video;V.src=URL.createObjectURL(o.video);V.onloadedmetadata=()=>{$('scrub').max=V.duration;fin()}}else{fin();st('Opened without video — reselect the original video file to see playback.')}};
$('del').onclick=async()=>{const n=$('sess').value;if(n&&confirm('Delete '+n+'?')){await tx('readwrite',s=>s.delete(n));listS()}};
const clear=()=>{frames=[];EF=[];S={};R={};RF={};K={};CORR=[];undo=[];redo=[];selKf=-1;subj=null;seed=null;ident=null;pickC=[];lasso=null;anchor=null;lp=[];dirty=true;lastT=-1};
$('reset').onclick=()=>{const l=lasso,sj=subj,sd=seed;clear();lasso=l;subj=sj;seed=sd;st('Analysis reset.')};$('newS').onclick=()=>{clear();V.removeAttribute('src');V.load();curFile=null;meta={};st('New session.')};
// ---- wiring
$('file').onchange=e=>{const f=e.target.files[0];if(!f)return;curFile=f;V.src=URL.createObjectURL(f);V.onloadedmetadata=()=>{const keep=frames.length&&f.name===meta.name;if(!keep){clear();$('t0').value=0;$('t1').value=V.duration.toFixed(2)}meta={...meta,name:f.name,w:V.videoWidth,h:V.videoHeight,duration:V.duration};$('scrub').max=V.duration;st(`${f.name} · ${V.videoWidth}×${V.videoHeight} · ${V.duration.toFixed(1)}s · ${(f.size/1e6).toFixed(1)}MB`);if(keep)compute()}};
$('analyze').onclick=analyze;$('cancel').onclick=()=>stop=true;
$('setA').onclick=()=>$('t0').value=V.currentTime.toFixed(2);$('setB').onclick=()=>$('t1').value=V.currentTime.toFixed(2);
$('prevClip').onclick=()=>{V.currentTime=+$('t0').value;endAt=+$('t1').value||V.duration;V.play()};
$('play').onclick=()=>{endAt=null;V.paused?V.play():V.pause()};$('back').onclick=()=>{V.pause();V.currentTime-=1/fps};$('fwd').onclick=()=>{V.pause();V.currentTime+=1/fps};
$('spd').onchange=()=>V.playbackRate=+$('spd').value;$('loop').onchange=()=>V.loop=$('loop').checked;$('scrub').oninput=()=>V.currentTime=+$('scrub').value;
for(const i of['sm','sP','sR','sJ','lf','off','mv','kR','kP','kMin','kMax','stab','stabT'])$(i).onchange=()=>{if(i.startsWith('k'))$('preset').value='cus';if(frames.length)compute()};
for(const i of['rig','coord','wrap','skel','trail','tl','lbl','rawc'])$(i).onchange=()=>{dirty=true;lastT=-1};
$('prop').onchange=()=>{prop=$('prop').value;selKf=-1;dirty=true};
$('parts').onclick=e=>{const id=e.target.dataset.id;if(id){sel=id;selKf=-1;dirty=true}};
$('kft').onclick=e=>{const r=e.target.closest('tr[data-j]');if(r){selKf=+r.dataset.j;V.currentTime=start+KS()[selKf].i/fps;dirty=true}};
$('info').onclick=e=>{if(e.target.dataset.c!==undefined)navigator.clipboard.writeText(e.target.dataset.c)};
$('addK').onclick=()=>{const k=KS(),i=ci();if(!k||k.some(q=>q.i===i))return;const j=k.findIndex(q=>q.i>i);k.splice(j<0?k.length:j,0,{i,v:curve(k,i),why:'Manually added',e:EZ.Linear});selKf=j<0?k.length-1:j;dirty=true};
$('delK').onclick=()=>{const k=KS();if(k&&selKf>=0&&k.length>2){k.splice(selKf,1);selKf=-1;dirty=true}};
$('regen').onclick=()=>frames.length&&compute();
$('fitK').onclick=()=>{if(KS()){fitAll(sel,prop);dirty=true}};
$('fixK').onclick=()=>{if(KS()){const n=fix(sel,prop);$('gmsg').textContent=n?`Added ${n} keyframe(s) where a segment would have to reverse.`:'All segments are valid.';dirty=true}};
(function loop(){requestAnimationFrame(loop);if(endAt!==null&&V.currentTime>=endAt){V.pause();endAt=null}
 if(dirty){parts();table();ezEd();dirty=false;lastT=-1}
 if(V.currentTime!==lastT||lastT===-1){lastT=V.currentTime;overlay();info();const w=G.clientWidth,h=G.clientHeight;if(G.width!==w||G.height!==h){G.width=w;G.height=h}G._m=drawGraph(G,prop,view,true);preview();agDraw();if(!dirty)ezEd();
  $('scrub').value=V.currentTime;$('tinfo').textContent=`${V.currentTime.toFixed(2)}s · f${Math.max(0,ci())}`;$('play').textContent=V.paused?'Play':'Pause'}})();

// ---- lasso buttons
$('eng').onchange=()=>st(engine()==='dw'?'Advanced tracker selected. Its models (about 350 MB) download once when you press Analyze, then stay cached.':'MediaPipe engine selected.');
$('lasso').onclick=()=>{if(!V.src)return st('Load a video first.');lassoing=!lassoing;picking=false;lp=[];$('lasso').classList.toggle('on',lassoing);O.style.cursor=lassoing?'crosshair':'';lastT=-1;st(lassoing?'Drag a loop around the spot to focus on, then release.':'Lasso cancelled.')};
$('lclear').onclick=()=>{lasso=null;anchor=null;lp=[];lassoing=false;$('lasso').classList.remove('on');O.style.cursor='';lastT=-1;st('Lasso cleared.')};
// ---- all-parts graph wall (every part for one property)
let agKey='';
$('agp').innerHTML=$('prop').innerHTML;
function agDraw(force){if($('ag').hidden)return;const ids=S.head?act():[],n=frames.length,key=ids.join()+'|'+n;
 if(key!==agKey||force){agKey=key;$('agg').innerHTML=ids.length&&n>1?ids.map(id=>`<div class=agc data-id=${id}><div class=agt>${D[id].n} <small>· ${quality(id)}</small></div><canvas></canvas></div>`).join(''):'<p class=m>Analyze a video first.</p>'}
 if(!ids.length||n<2)return;const pr=$('agp').value;
 for(const el of $('agg').children){const id=el.dataset.id,cv=el.querySelector('canvas');if(!cv)continue;el.classList.toggle('on',id===sel);const w=cv.clientWidth||320,h=cv.clientHeight||150;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}drawGraph(cv,pr,[0,n-1],false,id,true)}}
$('allg').onclick=()=>{$('agp').value=prop;$('ag').hidden=false;agDraw(true)};$('agx').onclick=()=>$('ag').hidden=true;$('agp').onchange=()=>agDraw(true);
$('agg').onclick=e=>{const c=e.target.closest('.agc');if(!c)return;sel=c.dataset.id;selKf=-1;const p=$('agp').value;if(KP.includes(p)){prop=p;$('prop').value=p}dirty=true};
$('agg').ondblclick=()=>$('ag').hidden=true;
$('agpng').onclick=()=>{const n=frames.length;if(!S.head||n<2)return st('Analyze first.');const ids=act(),pr=$('agp').value,cols=3,cw=620,ch=230,th=20,rows=Math.ceil(ids.length/cols),c=document.createElement('canvas');c.width=cols*cw;c.height=rows*(ch+th);const g=c.getContext('2d');g.fillStyle='#0d0f24';g.fillRect(0,0,c.width,c.height);
 ids.forEach((id,j)=>{const t=document.createElement('canvas');t.width=cw;t.height=ch;drawGraph(t,pr,[0,n-1],false,id,true);const x=j%cols*cw,y=Math.floor(j/cols)*(ch+th);g.drawImage(t,x,y+th);g.fillStyle='#d6d9de';g.font='13px sans-serif';g.fillText(`${D[id].n} · ${PROPS[pr]}`,x+8,y+14)});
 c.toBlob(b=>dl(`vid2anim_all_parts_${pr}.png`,b))};
listS();init();

// ---- sidebar tabs
function showTab(t){for(const b of $('tabs').querySelectorAll('button'))b.classList.toggle('on',b.dataset.tab===t);for(const d of document.querySelectorAll('#lt .tab'))d.hidden=d.dataset.tab!==t;try{localStorage.setItem('vid2anim.tab',t)}catch{}}
$('tabs').onclick=e=>{const t=e.target.dataset?.tab;if(t)showTab(t)};
try{showTab(localStorage.getItem('vid2anim.tab')||'scan')}catch{showTab('scan')}
// ---- Alight Motion export (alight.js): XML project and calibration file
initAlight({$,st,dl,curve,act,K:()=>K,S:()=>S,D:()=>D,fps:()=>fps,n:()=>frames.length,W:VW,H:VH,off:()=>+$('off').value||0,title:()=>(meta&&meta.name)||'vid2anim'});
