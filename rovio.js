// Rovio shell: page navigation, dashboard cards, project list.
// The editor itself (app.js / alight.js) is untouched by this file; it only clicks its buttons.
const $ = id => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} }
};
const PAGES = ['home', 'projects', 'tools', 'settings', 'about'];
const ABOUT_SECTIONS = ['privacy', 'terms', 'contact'];
const NEWS_ID = 'welcome-2026-11-20';

/* ---------- navigation (hash based, so pages can be bookmarked) ---------- */
let current = 'home';
// Tools only exist while a project is active: started with New Project (or Open) and ended by Save.
let active = false;
const toolsBtn = document.querySelector('.side .nv[data-go="tools"]');
function setActive(v) {
  active = v;
  toolsBtn.hidden = !v;
  $('pjSave').hidden = !v;
}
function show(page, section) {
  if (!PAGES.includes(page)) page = 'home';
  current = page;
  for (const p of PAGES) $('v-' + p).classList.toggle('on', p === page);
  document.querySelectorAll('.side .nv').forEach(b => b.classList.toggle('on', b.dataset.go === page));
  if (page === 'home') refreshStats();
  if (page === 'projects') renderProjects();
  if (page === 'settings') refreshStorage();
  if (page === 'tools') requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  if (section && $(section)) $(section).scrollIntoView({ block: 'start' });
  else if (page !== 'tools') $('v-' + page).scrollTop = 0;
}
function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h === 'tools' && !active) { history.replaceState(null, '', '#home'); return show('home'); }
  if (ABOUT_SECTIONS.includes(h)) return show('about', h);
  show(PAGES.includes(h) ? h : 'home');
}
const go = page => { if (location.hash === '#' + page) route(); else location.hash = page; };
window.addEventListener('hashchange', route);
document.addEventListener('click', e => {
  const t = e.target.closest('[data-go]');
  if (t) { e.preventDefault(); go(t.dataset.go); }
});

/* ---------- recent activity ---------- */
function note(label) { store.set('rovio.recent', { label, t: Date.now() }); refreshStats(); }
function ago(t) {
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s / 60) + ' min ago';
  if (s < 86400) return Math.floor(s / 3600) + ' h ago';
  return Math.floor(s / 86400) + ' d ago';
}
const on = (id, ev, label) => { const el = $(id); if (el) el.addEventListener(ev, () => note(typeof label === 'function' ? label() : label)); };
on('file', 'change', 'Video loaded');
on('analyze', 'click', 'Analysis started');
on('save', 'click', 'Session saved');
on('open', 'click', 'Session opened');
on('csv', 'click', 'Exported CSV');
on('json', 'click', 'Exported JSON');
on('sheet', 'click', 'Reference sheet');
on('amXml', 'click', 'Alight export');
on('amShare', 'click', 'Alight export');

/* ---------- projects (mirrors the editor's saved-session list) ---------- */
const sess = $('sess');
const names = () => [...sess.options].map(o => o.value || o.text).filter(Boolean);
const favs = () => new Set(store.get('rovio.favs', []));
function refreshStats() {
  const all = names(), f = favs();
  $('stP').textContent = all.length;
  $('stF').textContent = all.filter(n => f.has(n)).length;
  const r = store.get('rovio.recent', null);
  $('stA').textContent = r ? r.label : 'None';
  $('stAt').textContent = r ? ago(r.t) : '';
  $('rvDot').hidden = store.get('rovio.newsSeen', '') === NEWS_ID;
}
const star = '<svg class="ic"><use href="#i-star"/></svg>';
function renderProjects() {
  const q = $('rvq').value.trim().toLowerCase(), f = favs(), grid = $('pjGrid');
  const list = names().filter(n => n.toLowerCase().includes(q)).sort((a, b) => (f.has(b) - f.has(a)) || a.localeCompare(b));
  grid.textContent = '';
  if (!list.length) {
    const d = document.createElement('div');
    d.className = 'empty';
    d.style.gridColumn = '1/-1';
    d.textContent = q ? 'No projects match your search.' : 'No saved projects yet. Press New Project and load a video. When you are done, save it from the Export tab.';
    grid.append(d);
    return;
  }
  for (const n of list) {
    const c = document.createElement('div'); c.className = 'pj'; c.dataset.n = n;
    const top = document.createElement('div'); top.className = 'top';
    const nm = document.createElement('div'); nm.className = 'nm'; nm.textContent = n; nm.title = n;
    const st = document.createElement('button'); st.className = 'star' + (f.has(n) ? ' on' : ''); st.dataset.a = 'fav'; st.innerHTML = star;
    st.setAttribute('aria-label', 'Favorite ' + n); st.setAttribute('aria-pressed', f.has(n));
    top.append(nm, st);
    const acts = document.createElement('div'); acts.className = 'acts';
    const o = document.createElement('button'); o.className = 'open'; o.dataset.a = 'open'; o.textContent = 'Open';
    const d = document.createElement('button'); d.dataset.a = 'del'; d.textContent = 'Delete';
    acts.append(o, d); c.append(top, acts); grid.append(c);
  }
}
$('pjGrid').addEventListener('click', e => {
  const b = e.target.closest('button[data-a]'); if (!b) return;
  const n = b.closest('.pj').dataset.n;
  if (b.dataset.a === 'fav') {
    const f = favs(); f.has(n) ? f.delete(n) : f.add(n);
    store.set('rovio.favs', [...f]); renderProjects(); refreshStats();
  } else {
    sess.value = n;
    if (b.dataset.a === 'open') {
      if (active && hasVideo() && !confirm('Open this project? Unsaved changes to the current project will be lost.')) return;
      $('open').click(); setActive(true); go('tools');
    }
    else $('del').click();              // the editor asks for confirmation itself
  }
});
new MutationObserver(() => { refreshStats(); if (current === 'projects') renderProjects(); }).observe(sess, { childList: true });

/* ---------- new project / save ---------- */
const hasVideo = () => { const v = $('v'); return !!(v && v.currentSrc); };
function newProject() {
  if (active && hasVideo() && !confirm('Start a new project? Unsaved analysis will be cleared.')) return;
  if (hasVideo()) $('newS').click();   // always start from a clean editor
  setActive(true);
  note('New project');
  go('tools');
  $('file').click();
}
$('qsNew').onclick = newProject;
$('pjNew').onclick = newProject;
$('pjSave').onclick = () => $('save').click();
// A successful save finishes the project: Tools goes away and the saved project shows in Projects.
new MutationObserver(() => {
  if (active && /^(Session saved|Saved without)/.test($('status').textContent)) { setActive(false); go('projects'); }
}).observe($('status'), { childList: true, characterData: true, subtree: true });

/* ---------- top bar ---------- */
$('rvq').addEventListener('input', () => { if (current !== 'projects') go('projects'); else renderProjects(); });
$('rvBell').onclick = () => {
  store.set('rovio.newsSeen', NEWS_ID); refreshStats(); go('home');
  requestAnimationFrame(() => $('newsBox').scrollIntoView({ behavior: 'smooth', block: 'center' }));
};
$('rvUser').onclick = () => go('settings');

/* ---------- settings ---------- */
const MB = n => (n / 1048576).toFixed(n > 1e8 ? 0 : 1) + ' MB';
async function refreshStorage() {
  const el = $('stStore');
  try {
    const has = 'caches' in window && await caches.has('vid2anim-models');
    const est = navigator.storage?.estimate ? await navigator.storage.estimate() : null;
    el.textContent = (has ? 'Models are cached on this device.' : 'No models cached yet.') + (est ? ' Total site storage in use: ' + MB(est.usage || 0) + '.' : '');
  } catch { el.textContent = 'Storage information is not available in this browser.'; }
}
$('stClear').onclick = async () => {
  if (!confirm('Delete the downloaded models? They will download again the next time you analyze a video.')) return;
  try { await caches.delete('vid2anim-models'); $('stMsg').textContent = 'Models cleared.'; } catch { $('stMsg').textContent = 'Could not clear the cache in this browser.'; }
  refreshStorage();
};
$('stReset').onclick = () => { store.del('rovio.recent'); store.del('rovio.favs'); refreshStats(); if (current === 'projects') renderProjects(); };

setActive(false);
route();
refreshStats();
setInterval(() => { if (current === 'home') refreshStats(); }, 30000);
