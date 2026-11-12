/* "A note on ethics, from the author": a small pill that opens the full note in a dialog.
 *
 * Source of truth for every site (Tonic front page, showcase, Listening Archive hub, survey, interactive demo).
 * Pages carry it inline between <!--author-note--> markers; demo/author_note/sync.py copies this file in.
 *
 * Usage: any element with a data-author-note attribute becomes the trigger (clicks are delegated, so it can
 * be added after this script runs). The language follows <html lang>: "it…" gives Italian, otherwise English.
 */
(() => {
  if (window.__authorNote) return;
  window.__authorNote = true;

  const GIGAMIDI = "https://huggingface.co/datasets/Metacreation/GigaMIDI";
  const T = {
    en: {
      pill: "A note on ethics, from the author",
      eyebrow: "A note on ethics, from the author",
      title: "A tool for musicians to work with",
      personal: [
        "This work is <b>not</b> meant to replace musicians and artists. It aims to give them better tools.",
        "I have been a musician all my life, and many of my friends make their living from music. I know how much effort, love and passion go into it.",
      ],
      h1: "What this work does",
      p1: "Most models that generate music today work like a black box: you get a result, but little say in how it turns out. " +
          "This work looks at how to give the person using the model more control, and a clearer view of what it is doing. " +
          "You can ask it, for instance, for",
      chips: ["softer or louder playing", "longer or shorter notes", "a higher or lower register"],
      p1b: "and hear how it responds.",
      h2: "Where it could lead",
      p2: "Decades ago, musicians took up synthesizers and electronic instruments and made them their own.</p><p>I see this work as a small step " +
          "in the same direction: models as instruments for shaping your own ideas, and one day perhaps for <i>painting</i> music with them, " +
          "interacting live, almost jamming together.",
      dataT: "Trained on open data",
      data: `All the music used to train these models comes from <a href="${GIGAMIDI}" target="_blank" rel="noopener">GigaMIDI</a>, ` +
            "an open-source collection of MIDI files that is publicly and legally available online.",
      thanks: "Thank you for taking the time to listen.",
      close: "Close",
    },
    it: {
      pill: "Una nota sull'etica",
      eyebrow: "Una nota sull'etica, di chi ha realizzato questo lavoro",
      title: "Uno strumento al servizio dei musicisti",
      personal: [
        "Questo lavoro <b>non</b> vuole sostituire musicisti e artisti: vuole dare loro strumenti migliori.",
        "Sono musicista da tutta la vita, e molti miei amici vivono di musica. So quanto impegno, amore e passione ci siano dietro.",
      ],
      h1: "Cosa fa questo lavoro",
      p1: "Oggi la maggior parte dei modelli che generano musica funziona come una scatola chiusa: si ottiene un risultato, ma si può decidere " +
          "ben poco di come viene fuori. Questo lavoro studia come dare più controllo a chi usa il modello, e capire meglio cosa sta facendo. " +
          "Gli si può chiedere, per esempio,",
      chips: ["un suono più piano o più forte", "note più lunghe o più corte", "un registro più acuto o più grave"],
      p1b: "e ascoltare come risponde.",
      h2: "Dove potrebbe portare",
      p2: "Decenni fa i musicisti hanno fatto propri i sintetizzatori e gli strumenti elettronici.</p><p>Vedo questo lavoro come un piccolo passo " +
          "nella stessa direzione: modelli come strumenti per dare forma alle proprie idee e, un giorno, magari per <i>dipingere</i> la musica, " +
          "interagendo con loro dal vivo, quasi come in una jam session.",
      dataT: "Addestrati su dati open source",
      data: `Tutta la musica usata per addestrare questi modelli proviene da <a href="${GIGAMIDI}" target="_blank" rel="noopener">GigaMIDI</a>, ` +
            "una raccolta open source di file MIDI disponibile pubblicamente e legalmente su internet.",
      thanks: "Grazie per il tuo tempo.",
      close: "Chiudi",
    },
  };
  const t = (document.documentElement.lang || "").toLowerCase().startsWith("it") ? T.it : T.en;

  const CSS = `
.an-pill { all: unset; box-sizing: border-box; display: inline-flex; align-items: center; gap: 8px; cursor: pointer;
  padding: 6px 13px 6px 9px; border-radius: 999px; font: 500 13px/1.2 Inter, system-ui, sans-serif; color: #d9d4ff;
  background: linear-gradient(90deg, rgba(139,124,255,.16), rgba(63,208,201,.10)); border: 1px solid rgba(139,124,255,.38);
  transition: border-color .2s, background .2s, transform .2s; white-space: nowrap; }
/* Hosts with their own button styles (Gradio's .prose, etc.): the pill's look must win. */
html body .an-pill { font: 500 13px/1.2 Inter, system-ui, sans-serif !important; color: #d9d4ff !important; padding: 6px 13px 6px 9px !important;
  border: 1px solid rgba(139,124,255,.38) !important; border-radius: 999px !important; box-shadow: none !important; min-width: 0 !important;
  background: linear-gradient(90deg, rgba(139,124,255,.16), rgba(63,208,201,.10)) !important; margin: 0; }
html body .an-pill:hover { border-color: rgba(139,124,255,.8) !important; background: linear-gradient(90deg, rgba(139,124,255,.26), rgba(63,208,201,.16)) !important; }
.an-pill:hover { border-color: rgba(139,124,255,.8); background: linear-gradient(90deg, rgba(139,124,255,.26), rgba(63,208,201,.16)); transform: translateY(-1px); }
.an-pill:focus-visible { outline: 2px solid #8b7cff; outline-offset: 2px; }
.an-pill .an-dot { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; font-size: 11px;
  background: rgba(139,124,255,.3); color: #fff; }
dialog.an-dialog { box-sizing: border-box; width: min(660px, calc(100vw - 24px)); max-height: min(88vh, 900px); padding: 0; margin: auto;
  border: 1px solid rgba(255,255,255,.12); border-radius: 22px; background: #12141c; color: #e8eaf2; overflow: hidden;
  box-shadow: 0 40px 120px -20px rgba(0,0,0,.8), 0 0 0 1px rgba(139,124,255,.15); font: 400 15px/1.6 Inter, system-ui, sans-serif; }
dialog.an-dialog[open] { display: flex; flex-direction: column; animation: an-in .22s ease-out; }
dialog.an-dialog::backdrop { background: rgba(5,6,10,.62); backdrop-filter: blur(4px); }
@keyframes an-in { from { opacity: 0; transform: translateY(10px) scale(.985); } }
.an-bar { height: 3px; flex: none; background: linear-gradient(90deg, #8b5cf6, #22d3ee); }
.an-body { overflow-y: auto; padding: 26px 28px 24px; }
.an-x { all: unset; position: absolute; top: 14px; right: 14px; width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
  cursor: pointer; color: #9298ab; font-size: 20px; background: rgba(255,255,255,.04); }
.an-x:hover { color: #fff; background: rgba(255,255,255,.1); }
.an-x:focus-visible { outline: 2px solid #8b7cff; }
.an-eyebrow { margin: 0 40px 10px 0; font: 600 11.5px/1.4 "JetBrains Mono", ui-monospace, monospace; letter-spacing: .12em; text-transform: uppercase; color: #3fd0c9; }
.an-title { margin: 0 0 16px; font: 750 clamp(22px, 4vw, 28px)/1.15 Inter, system-ui, sans-serif; letter-spacing: -.02em; color: #f2f3f8; }
.an-personal { margin: 0 0 18px; padding: 2px 0 2px 16px; border-left: 3px solid #8b7cff; }
.an-personal p { margin: 0; color: #e8eaf2; font-size: 15.5px; }
.an-personal p + p { margin-top: 8px; color: #b4b9c8; }
.an-personal b { color: #fff; }
.an-block { position: relative; padding: 15px 18px 15px 52px; margin: 0 0 10px; border-radius: 14px; background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08); }
.an-num { position: absolute; left: 18px; top: 16px; font: 600 12px "JetBrains Mono", ui-monospace, monospace; color: #3fd0c9; }
.an-block h4 { margin: 0 0 4px; font: 650 15.5px/1.3 Inter, system-ui, sans-serif; color: #f2f3f8; }
.an-block p { margin: 0; color: #a3a8b8; font-size: 14.5px; }
.an-block i { color: #e8eaf2; }
.an-block p + p { margin-top: 8px; }
.an-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.an-chips span { padding: 5px 11px; border-radius: 999px; font-size: 13px; color: #e8eaf2; background: rgba(139,124,255,.12); border: 1px solid rgba(139,124,255,.32); }
.an-data { display: flex; gap: 12px; margin: 14px 0 0; padding: 13px 15px; border-radius: 14px; border: 1px dashed rgba(63,208,201,.35); background: rgba(63,208,201,.04); }
.an-data .an-ico { flex: none; width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-size: 15px; background: rgba(63,208,201,.1); border: 1px solid rgba(63,208,201,.3); }
.an-data b { display: block; color: #f2f3f8; font-weight: 650; margin-bottom: 2px; }
.an-data p { margin: 0; color: #a3a8b8; font-size: 14px; }
.an-data a { color: #3fd0c9; }
.an-thanks { margin: 20px 0 0; text-align: right; color: #a3a8b8; font-size: 14.5px; }
.an-sign { display: inline-block; margin-top: 2px; font: 600 38px/1.1 "Caveat", "Segoe Script", "Bradley Hand", cursive; color: #c4b5fd; transform: rotate(-3deg); }
@media (max-width: 520px) { .an-body { padding: 22px 18px 20px; } .an-block { padding: 14px 14px; } .an-num { position: static; display: block; margin-bottom: 2px; } }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) body .an-pill.an-auto { color: #4b3fb0 !important; background: linear-gradient(90deg, rgba(139,124,255,.12), rgba(63,208,201,.08)) !important; border-color: rgba(110,92,230,.45) !important; }
  :root:not([data-theme="dark"]) .an-pill.an-auto .an-dot { background: #8b7cff; }
}
@media (prefers-reduced-motion: reduce) { dialog.an-dialog[open] { animation: none; } .an-pill { transition: none; } }
`;

  function setup() {
    // Caveat for the signature; Inter for pages that don't load it already (e.g. the archive hub).
    const fam = [];
    if (!document.querySelector('link[href*="family=Caveat"]')) fam.push("family=Caveat:wght@600");
    if (!document.querySelector('link[href*="family=Inter"]')) fam.push("family=Inter:wght@400;500;600;650;750");
    if (fam.length) {
      const f = document.createElement("link");
      f.rel = "stylesheet";
      f.href = `https://fonts.googleapis.com/css2?${fam.join("&")}&display=swap`;
      document.head.append(f);
    }
    const st = document.createElement("style");
    st.textContent = CSS;
    document.head.append(st);
  }

  let dlg = null;
  function dialog() {
    if (dlg) return dlg;
    dlg = document.createElement("dialog");
    dlg.className = "an-dialog";
    dlg.setAttribute("aria-labelledby", "an-title");
    dlg.innerHTML = `
      <div class="an-bar"></div>
      <button class="an-x" type="button" aria-label="${t.close}">×</button>
      <div class="an-body">
        <p class="an-eyebrow">${t.eyebrow}</p>
        <h3 class="an-title" id="an-title">${t.title}</h3>
        <div class="an-personal">${t.personal.map((p) => `<p>${p}</p>`).join("")}</div>
        <div class="an-block"><span class="an-num">01</span><h4>${t.h1}</h4><p>${t.p1}</p>
          <div class="an-chips">${t.chips.map((c) => `<span>${c}</span>`).join("")}</div><p>${t.p1b}</p></div>
        <div class="an-block"><span class="an-num">02</span><h4>${t.h2}</h4><p>${t.p2}</p></div>
        <div class="an-data"><span class="an-ico" aria-hidden="true">🎼</span><div><b>${t.dataT}</b><p>${t.data}</p></div></div>
        <p class="an-thanks">${t.thanks}<br><span class="an-sign">Rebecca</span></p>
      </div>`;
    document.body.append(dlg);
    dlg.querySelector(".an-x").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });  // click on the backdrop
    return dlg;
  }

  // Turn every [data-author-note] into the pill (once), now and whenever pages add one later.
  function decorate(root) {
    (root.querySelectorAll ? root : document).querySelectorAll("[data-author-note]:not(.an-pill)").forEach((el) => {
      el.classList.add("an-pill");
      if (el.dataset.authorNote === "auto") el.classList.add("an-auto");  // page has a light theme too
      if (el.tagName !== "BUTTON") { el.setAttribute("role", "button"); el.tabIndex = 0; }
      if (!el.textContent.trim()) el.innerHTML = `<span class="an-dot" aria-hidden="true">♪</span>${t.pill}`;
      el.setAttribute("aria-haspopup", "dialog");
    });
  }
  document.addEventListener("click", (e) => {
    const el = e.target.closest && e.target.closest("[data-author-note]");
    if (!el) return;
    e.preventDefault();
    dialog().showModal();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches && e.target.matches("[data-author-note]:not(button)")) {
      e.preventDefault(); dialog().showModal();
    }
  });

  const start = () => {
    setup();
    decorate(document);
    new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => n.nodeType === 1 && (n.matches("[data-author-note]") ? decorate(n.parentNode || document) : decorate(n)))))
      .observe(document.body, { childList: true, subtree: true });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
