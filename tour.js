// KineExtract guided tour: an optional, skippable walkthrough for new users.
// It only highlights things and moves between pages and sidebar tabs. It never loads a video or changes a project.
// Start: first visit (offered once), the "Tour" button in the top bar, or Settings > Get started tutorial.
// Keys: Right / Enter = next, Left = back, Esc = skip.
const $ = id => document.getElementById(id);
const q = (s, r = document) => r.querySelector(s);
const reduce = matchMedia('(prefers-reduced-motion: reduce)');
const KEY = 'kineextract.tour';
const seen = () => { try { return !!JSON.parse(localStorage.getItem(KEY) || 'null')?.done; } catch { return false; } };
const mark = () => { try { localStorage.setItem(KEY, JSON.stringify({ done: true, t: Date.now() })); } catch {} };
const shell = () => window.kineextractShell || {};
const toast = m => window.dispatchEvent(new CustomEvent('kx:toast', { detail: m }));

/* ---------- page / tab helpers used by the steps ---------- */
let openedTools = false;
const home = () => shell().go?.('home');
const projects = () => shell().go?.('projects');
const tools = () => {
  const s = shell();
  if (s.current?.() === 'tools') return;
  if (!s.isActive?.()) openedTools = true;
  s.openTools?.();
};
const tab = t => { const b = q(`#tabs button[data-tab="${t}"]`); if (b && !b.classList.contains('on')) b.click(); };
const inTools = t => () => { tools(); tab(t); };

/* ---------- the steps ---------- */
const STEPS = [
  { welcome: true, title: 'Welcome to KineExtract',
    text: 'KineExtract turns a video into clean, editable animation, right in your browser. Take a one minute tour, or skip it and dive in. You can restart it any time from the Tour button or from Settings.' },
  { title: 'Find your way around', target: '.side', place: 'right', before: home,
    text: 'Use the sidebar to move between Home, Projects, Settings and About. A Tools tab appears here while a project is open.' },
  { title: 'Start a new project', target: '#qsNew', place: 'bottom', before: home,
    text: 'New Project opens the editor and asks for a video file. Everything is processed on your device, so nothing is uploaded.' },
  { title: 'Your dashboard', target: '.stats', place: 'top', before: home,
    text: 'These cards show how many projects you have saved and what you did last.' },
  { title: 'Your projects', target: '#pjGrid', place: 'bottom', before: projects,
    text: 'Saved sessions appear here. Press Open to continue one or Delete to remove it. Leave the editor with unsaved edits and they are autosaved as an Unnamed project. The search bar at the top filters projects by name.' },
  { title: 'Load a video', target: '#v-tools header', place: 'bottom', before: inTools('scan'),
    text: 'In the editor, press Upload video, then Analyze. The first analysis downloads the tracking models (about 350 MB) once and keeps them cached. The round arrow on the left leaves the editor.' },
  { title: 'Status and progress', target: '#v-tools .stat', place: 'bottom', before: inTools('scan'),
    text: 'Messages and a progress bar show here while a video loads and is analyzed. Press Cancel, next to Analyze, to stop a run at any time.' },
  { title: 'Four sidebar tabs', target: '#tabs', place: 'right', before: inTools('scan'),
    text: 'Scan sets tracking quality. Subject picks who to follow. Clean-up smooths the motion and builds keyframes. Export saves your results.' },
  { title: 'Tracking quality', target: '#lt .tab[data-tab="scan"] details', place: 'right', before: inTools('scan'),
    text: 'The defaults work for most clips. Careful measures every frame twice. Pick Fast for a quick test, or Exhaustive for hard footage. Higher quality takes longer.' },
  { title: 'Choose who to track', target: '#mid .vtools', place: 'bottom', before: inTools('subj'),
    text: 'Go to the first frame, press Pick subject and click the person you want. Draw paints the area to focus on, Erase rubs it out and Clear removes your choices.' },
  { title: 'Video and overlay', target: () => union('#stage', '#scrub'), place: 'left', before: inTools('subj'),
    text: 'Your video plays here with the tracked skeleton on top. Use Play, the step buttons and the slider under it to move through the clip. Tick Correct landmarks to drag a wrong point back into place.' },
  { title: 'Smooth and simplify', target: '#lt .tab[data-tab="clean"] details:nth-of-type(2)', place: 'right', before: inTools('clean'),
    text: 'Smoothing removes jitter. The Keyframes presets (Minimal, Balanced, Accurate) decide how many keyframes your animation gets. Camera shake and problem frames are in this tab too.' },
  { title: 'Graphs and keyframes', target: '#v-tools footer', place: 'top', before: inTools('clean'),
    text: 'Pick a property to see its curve. Drag the diamonds to edit keyframes, scroll or use the plus and minus buttons to zoom, and click to jump to a frame. All parts graphs shows every part at once.' },
  { title: 'Selected part and easing', target: '#rt', place: 'left', before: inTools('clean'),
    text: 'Pick a body part in the Subject tab to inspect it here. Select a keyframe to shape the easing curve that leads to the next one.' },
  { title: 'Export to Alight Motion', target: '#lt .tab[data-tab="out"] details:nth-of-type(2)', place: 'right', before: inTools('out'),
    text: 'Builds an Alight Motion project (.xml) with one layer per body part. Two automatic checks run, then you preview it and confirm before downloading. Data export above gives you CSV or JSON.' },
  { title: 'Save your work', target: '#lt .tab[data-tab="out"] details:nth-of-type(3)', place: 'right', before: inTools('out'),
    text: 'Save session stores the project on this device. It then shows up under Projects, ready to reopen later.' },
  { done: true, title: "You're all set", before: home,
    text: 'That is everything you need. Press New Project on the Home page and load a video. You can replay this tour any time from the Tour button or Settings.' }
];

/* ---------- DOM ---------- */
let root, hole, card, el = {};
let idx = -1, running = false, raf = 0, snap = true, missingSince = 0, prevTab = 'scan';
let scrollEl = null;
const cur = { hx: 0, hy: 0, hw: 0, hh: 0, cx: 0, cy: 0 };

function build() {
  if (root) return;
  root = document.createElement('div');
  root.id = 'tour';
  root.hidden = true;
  root.className = 'center';
  root.innerHTML = `<div class="tour-hole"></div>
<div class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tourT" aria-describedby="tourB">
 <div class="tour-spark" hidden><svg class="ic st" viewBox="0 0 24 24"><path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2z"/></svg></div>
 <div class="tour-step"></div>
 <h3 id="tourT"></h3>
 <p id="tourB" aria-live="polite"></p>
 <div class="tour-bar"><i></i></div>
 <div class="tour-actions"><button type="button" class="tour-skip">Skip tour</button><button type="button" class="tour-back">Back</button><button type="button" class="go tour-next">Next</button></div>
</div>`;
  document.body.append(root);
  hole = q('.tour-hole', root);
  card = q('.tour-card', root);
  el = { spark: q('.tour-spark', root), step: q('.tour-step', root), t: $('tourT'), b: $('tourB'), bar: q('.tour-bar i', root),
    skip: q('.tour-skip', root), back: q('.tour-back', root), next: q('.tour-next', root) };
  el.skip.onclick = () => end(true);
  el.back.onclick = () => go(idx - 1);
  el.next.onclick = () => (idx >= STEPS.length - 1 ? end(false) : go(idx + 1));
  root.addEventListener('pointerdown', e => { if (!card.contains(e.target)) e.preventDefault(); });
}

const narrow = () => innerWidth < 700;

function go(i) {
  if (i < 0 || i >= STEPS.length) return;
  clearScrollMargin();
  idx = i;
  const s = STEPS[i];
  try { s.before?.(); } catch (e) { console.warn('tour step hook failed', e); }
  missingSince = performance.now();
  // content
  const first = i === 0, last = i === STEPS.length - 1;
  card.classList.toggle('welcome', !!(s.welcome || s.done));
  el.spark.hidden = !(s.welcome || s.done);
  el.step.textContent = (first || last) ? '' : `Step ${i} of ${STEPS.length - 2}`;
  el.step.hidden = first || last;
  el.t.textContent = s.title;
  el.b.textContent = s.text;
  el.bar.style.width = Math.round(i / (STEPS.length - 1) * 100) + '%';
  el.back.hidden = first;
  el.skip.hidden = last;
  el.next.textContent = first ? 'Start tour' : last ? 'Finish' : 'Next';
  card.classList.remove('swap'); void card.offsetWidth; card.classList.add('swap');
  el.next.focus({ preventScroll: true });
  // bring the target into view once the page / tab switch has laid out
  const reveal = () => {
    if (!running || idx !== i) return;
    let t = targetOf(s);
    if (t && t.group) t = t.el;
    if (!t) return;
    if (narrow()) { t.style.scrollMarginTop = '76px'; scrollEl = t; }
    try { t.scrollIntoView({ block: narrow() ? 'start' : 'nearest', inline: 'nearest', behavior: reduce.matches ? 'auto' : 'smooth' }); } catch {}
  };
  setTimeout(reveal, 80);
  setTimeout(reveal, 520);
}
function clearScrollMargin() { if (scrollEl) { scrollEl.style.scrollMarginTop = ''; scrollEl = null; } }

const targetOf = s => (typeof s.target === 'function' ? s.target() : s.target ? q(s.target) : null);
// a target can also be a group of elements: the highlight then covers all of them
const union = (...sels) => ({
  el: q(sels[0]),
  group: sels.map(x => q(x)),
});
function measure(t) {
  if (t && t.group) {
    const rs = t.group.filter(e => e && e.isConnected && getComputedStyle(e).visibility !== 'hidden').map(e => e.getBoundingClientRect()).filter(r => r.width > 1 && r.height > 1);
    if (!rs.length) return null;
    const l = Math.min(...rs.map(r => r.left)), tp = Math.min(...rs.map(r => r.top)), rt = Math.max(...rs.map(r => r.right)), b = Math.max(...rs.map(r => r.bottom));
    return { left: l, top: tp, right: rt, bottom: b, width: rt - l, height: b - tp };
  }
  if (!t || !t.isConnected) return null;
  if (getComputedStyle(t).visibility === 'hidden') return null;
  const r = t.getBoundingClientRect();
  return r.width < 2 || r.height < 2 ? null : r;
}

function place(r, cw, ch, pref) {
  const vw = innerWidth, vh = innerHeight, m = 12, g = 16;
  const fits = (x, y) => x >= m && y >= m && x + cw <= vw - m && y + ch <= vh - m;
  const cxm = Math.min(Math.max(r.left + r.width / 2 - cw / 2, m), vw - cw - m);
  const cym = Math.min(Math.max(r.top + r.height / 2 - ch / 2, m), vh - ch - m);
  const opts = { bottom: [cxm, r.bottom + g], top: [cxm, r.top - ch - g], right: [r.right + g, cym], left: [r.left - cw - g, cym] };
  const order = [pref, 'bottom', 'top', 'right', 'left'].filter((k, i, a) => k && a.indexOf(k) === i);
  for (const k of order) { const [x, y] = opts[k]; if (fits(x, y)) return [x, y]; }
  // target is too big to sit beside: park the card over the quieter edge
  return [(vw - cw) / 2, r.top + r.height / 2 > vh / 2 ? m : vh - ch - m];
}

function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  const s = STEPS[idx];
  if (!s) return;
  const cw = card.offsetWidth, ch = card.offsetHeight, vw = innerWidth, vh = innerHeight;
  let r = measure(targetOf(s));
  const hasTarget = !!s.target;
  if (!r && hasTarget && performance.now() - missingSince < 700) return; // page is still switching: hold position
  let hx, hy, hw, hh, px, py;
  if (r) {
    const pad = 8;
    hx = Math.max(r.left - pad, 4); hy = Math.max(r.top - pad, 4);
    hw = Math.min(r.right + pad, vw - 4) - hx; hh = Math.min(r.bottom + pad, vh - 4) - hy;
    const rr = { left: hx, top: hy, right: hx + hw, bottom: hy + hh, width: hw, height: hh };
    if (narrow()) { px = (vw - cw) / 2; py = vh - ch - 12; if (rr.bottom > py - 8 && rr.top < py + ch) { /* sheet covers the target: nudge to top */ if (rr.top > ch + 24) py = 12; } }
    else [px, py] = place(rr, cw, ch, s.place);
    root.classList.remove('center');
  } else {
    hx = vw / 2; hy = vh / 2; hw = 0; hh = 0;
    px = (vw - cw) / 2; py = Math.max(12, (vh - ch) / 2);
    root.classList.add('center');
  }
  const k = snap || reduce.matches ? 1 : 0.22;
  cur.hx += (hx - cur.hx) * k; cur.hy += (hy - cur.hy) * k; cur.hw += (hw - cur.hw) * k; cur.hh += (hh - cur.hh) * k;
  cur.cx += (px - cur.cx) * k; cur.cy += (py - cur.cy) * k;
  snap = false;
  hole.style.cssText = `left:${cur.hx}px;top:${cur.hy}px;width:${Math.max(0, cur.hw)}px;height:${Math.max(0, cur.hh)}px`;
  card.style.left = cur.cx + 'px';
  card.style.top = cur.cy + 'px';
}

/* ---------- start / end ---------- */
function onKey(e) {
  if (!running) return;
  if (e.key === 'Escape') { e.preventDefault(); end(true); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); el.next.click(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); if (!el.back.hidden) el.back.click(); }
  else if (e.key === 'Tab') {
    const f = [...card.querySelectorAll('button')].filter(b => !b.hidden && !b.disabled);
    if (!f.length) return;
    const i = f.indexOf(document.activeElement);
    e.preventDefault();
    f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length].focus();
  }
}

function start() {
  if (running) return;
  build();
  running = true;
  openedTools = false;
  prevTab = (() => { try { return localStorage.getItem('vid2anim.tab') || 'scan'; } catch { return 'scan'; } })();
  document.body.classList.add('touring');
  root.hidden = false;
  snap = true;
  addEventListener('keydown', onKey, true);
  go(0);
  raf = requestAnimationFrame(loop);
}

function end(skipped) {
  if (!running) return;
  running = false;
  cancelAnimationFrame(raf);
  removeEventListener('keydown', onKey, true);
  clearScrollMargin();
  const wasFirst = !seen();
  mark();
  root.classList.add('closing');
  setTimeout(() => { root.hidden = true; root.classList.remove('closing'); }, reduce.matches ? 0 : 240);
  document.body.classList.remove('touring');
  // put the editor back the way it was and close the empty project the tour opened
  tab(prevTab);
  if (!skipped || (openedTools && shell().current?.() === 'tools')) home();
  if (skipped && wasFirst) toast('Tour skipped. Restart it any time from the Tour button.');
  $('tourBtn')?.classList.remove('hint');
  idx = -1;
}

/* ---------- entry points ---------- */
$('tourBtn')?.addEventListener('click', start);
$('stTour')?.addEventListener('click', start);
if (!seen()) {
  $('tourBtn')?.classList.add('hint');
  setTimeout(() => { if (!seen() && !running) start(); }, 900);
}
window.kineextractTour = { start, end: () => end(true) };
