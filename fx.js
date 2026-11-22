// KineExtract motion layer: small behaviours that make the UI feel alive.
// Purely cosmetic. It never touches the editor. All motion is skipped when the system asks for reduced motion.
const $ = id => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)');

/* cursor glow on cards (CSS reads --mx / --my) */
document.addEventListener('pointermove', e => {
  if (e.pointerType === 'touch') return;
  const t = e.target.closest && e.target.closest('.sc,.pn,.pj,.box');
  if (!t) return;
  const r = t.getBoundingClientRect();
  t.style.setProperty('--mx', (e.clientX - r.left) + 'px');
  t.style.setProperty('--my', (e.clientY - r.top) + 'px');
}, { passive: true });

/* ripple on primary buttons and nav items */
document.addEventListener('pointerdown', e => {
  if (reduce.matches) return;
  const b = e.target.closest && e.target.closest('.bigbtn,button.go,.pj .acts button.open,.side .nv,.tour-actions button');
  if (!b || b.disabled) return;
  const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height) * 2;
  const s = document.createElement('span');
  s.className = 'kx-rip';
  s.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px`;
  b.append(s);
  s.addEventListener('animationend', () => s.remove());
  setTimeout(() => s.remove(), 900);
});

/* toasts: the shell and the tour send 'kx:toast' with a message */
const box = document.createElement('div');
box.id = 'kxToasts';
box.setAttribute('aria-live', 'polite');
document.body.append(box);
function toast(msg, ms = 2600) {
  if (!msg) return;
  const t = document.createElement('div');
  t.className = 'kx-toast';
  t.textContent = msg;
  box.append(t);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 340); }, ms);
}
window.addEventListener('kx:toast', e => toast(e.detail));

/* thin accent line that sweeps across the top when you change page */
const line = document.createElement('div');
line.id = 'kxLine';
document.body.append(line);
addEventListener('hashchange', () => {
  if (reduce.matches) return;
  line.classList.remove('go');
  void line.offsetWidth;
  line.classList.add('go');
});

/* project counter counts up / down to its new value */
const stP = $('stP');
if (stP) {
  let shown = Number(stP.textContent) || 0, lastSet = stP.textContent, raf = 0;
  const mo = new MutationObserver(() => {
    const txt = stP.textContent;
    if (txt === lastSet) return;
    const to = Number(txt);
    if (!Number.isFinite(to) || reduce.matches) { shown = to || 0; lastSet = txt; return; }
    cancelAnimationFrame(raf);
    const from = shown, t0 = performance.now(), dur = 650;
    stP.classList.remove('bump'); void stP.offsetWidth; stP.classList.add('bump');
    const step = now => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      shown = Math.round(from + (to - from) * e);
      lastSet = String(shown);
      stP.textContent = lastSet;
      if (k < 1) raf = requestAnimationFrame(step);
      else { shown = to; lastSet = String(to); }
    };
    lastSet = String(from);
    raf = requestAnimationFrame(step);
  });
  mo.observe(stP, { childList: true, characterData: true, subtree: true });
}

/* project cards: stagger index, and stay quiet while you type in the search box */
const grid = $('pjGrid');
if (grid) {
  new MutationObserver(() => {
    grid.classList.toggle('quiet', document.activeElement === $('rvq'));
    [...grid.children].forEach((c, i) => c.style.setProperty('--i', Math.min(i, 10)));
  }).observe(grid, { childList: true });
}

/* status line flashes softly when its text changes (throttled so analysis doesn't strobe) */
const status = $('status');
if (status) {
  let last = 0;
  new MutationObserver(() => {
    const n = performance.now();
    if (reduce.matches || n - last < 700) return;
    last = n;
    status.classList.remove('tick'); void status.offsetWidth; status.classList.add('tick');
  }).observe(status, { childList: true, characterData: true, subtree: true });
}
