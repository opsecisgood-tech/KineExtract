// KineExtract small extras: toast messages and a project counter that counts to its new value.
// Purely cosmetic. It never touches the editor. Skipped when the system asks for reduced motion.
const $ = id => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)');

/* toasts: the shell and the tour send 'kx:toast' with a message */
const box = document.createElement('div');
box.id = 'kxToasts';
box.setAttribute('aria-live', 'polite');
document.body.append(box);
function toast(msg, ms = 2400) {
  if (!msg) return;
  const t = document.createElement('div');
  t.className = 'kx-toast';
  t.textContent = msg;
  box.append(t);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 320); }, ms);
}
window.addEventListener('kx:toast', e => toast(e.detail));

/* project counter eases to its new value */
const stP = $('stP');
if (stP) {
  let shown = Number(stP.textContent) || 0, lastSet = stP.textContent, raf = 0;
  new MutationObserver(() => {
    const txt = stP.textContent;
    if (txt === lastSet) return;
    const to = Number(txt);
    if (!Number.isFinite(to) || reduce.matches) { shown = to || 0; lastSet = txt; return; }
    cancelAnimationFrame(raf);
    const from = shown, t0 = performance.now(), dur = 350;
    const step = now => {
      const k = Math.min(1, (now - t0) / dur);
      shown = Math.round(from + (to - from) * k);
      lastSet = String(shown);
      stP.textContent = lastSet;
      if (k < 1) raf = requestAnimationFrame(step);
      else { shown = to; lastSet = String(to); }
    };
    lastSet = String(from);
    raf = requestAnimationFrame(step);
  }).observe(stP, { childList: true, characterData: true, subtree: true });
}
