// Italian copy of ../survey.js (same trials, audio, survey.json and endpoint;
// responses carry lang: "it"). Keep in sync when the English survey changes.
// Blind listening survey. Trials are sampled per participant from survey.json
// (built by demo/survey/build_survey.py); clips are opaque hashes, options are
// shown as A, B, C, D in random order. Responses go to SURVEY_CONFIG.endpoint.
(() => {
  const CFG = window.SURVEY_CONFIG || {};
  const STORE = "ebt-survey-v1-it";
  const app = document.getElementById("app");
  const player = document.getElementById("player");
  const LETTERS = ["A", "B", "C", "D", "E", "F"];
  const MIN_LISTEN = 0.7;   // share of a clip that must be heard before answering
  // ?test on localhost skips the listening requirement (for automated checks only).
  const TEST = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) && new URLSearchParams(location.search).has("test");

  // Italian text for the strings that live in survey.json.
  const IT = {
    scale: {
      velocity: ["decisamente più piano (più delicata)", "leggermente più piano", "nessuna differenza", "leggermente più forte", "decisamente più forte (più intensa)"],
      duration: ["note decisamente più brevi (più staccate)", "note leggermente più brevi", "nessuna differenza", "note leggermente più lunghe", "note decisamente più lunghe (più legate)"],
      pitch_register: ["decisamente più grave (suoni più profondi)", "leggermente più grave", "nessuna differenza", "leggermente più acuta", "decisamente più acuta (suoni più brillanti)"],
    },
    attr_name: {
      velocity: 'l\'intensità <span class="ends">piano ↔ forte</span>',
      duration: 'la durata delle note <span class="ends">staccato ↔ legato</span>',
      pitch_register: 'l\'altezza <span class="ends">suoni gravi ↔ suoni acuti</span>',
    },
    attr_help: {
      velocity: "Piano = suono delicato e leggero; forte = suono intenso e potente.",
      duration: "Staccato = note brevi, staccate l'una dall'altra; legato = note lunghe, che si collegano senza stacchi.",
      pitch_register: "Suoni gravi = note basse e profonde; suoni acuti = note alte e brillanti.",
    },
  };

  // ------------------------------------------------------------------ utils
  const h = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  function rng(seed) {  // mulberry32
    let a = 0;
    for (const c of seed) a = (Math.imul(a ^ c.charCodeAt(0), 2654435761) >>> 0);
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const shuffle = (arr, r) => {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  };
  const row = (cls, ...kids) => { const e = h("div", cls); e.append(...kids); return e; };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const save = () => localStorage.setItem(STORE, JSON.stringify(state));

  // ------------------------------------------------------------- sampling
  // Round-robin over groups so each participant covers many conditions.
  function sampleGrouped(trials, keyFn, n, r) {
    const groups = {};
    for (const [i, t] of trials.entries()) (groups[keyFn(t)] ||= []).push(i);
    const keys = shuffle(Object.keys(groups), r);
    const queues = Object.fromEntries(keys.map((k) => [k, shuffle(groups[k], r)]));
    const out = [];
    while (out.length < n && keys.some((k) => queues[k].length)) {
      for (const k of keys) {
        if (out.length >= n) break;
        if (queues[k].length) out.push(queues[k].pop());
      }
    }
    return out;
  }

  function makePlan(survey, id) {
    const r = rng(id);
    const P = survey.pool, N = survey.plan;
    const insert = (list, item) => { const at = Math.floor(r() * (list.length + 1)); list.splice(at, 0, item); };
    const withOrder = (kind, idx, t) => ({
      kind, pool_index: idx,
      order: t.options ? shuffle(t.options, r) : null,
    });

    const unguided = sampleGrouped(P.unguided, (t) => t.tok, N.unguided, r).map((i) => withOrder("unguided", i, P.unguided[i]));
    if (P.catch_unguided.length && N.catch_unguided) {
      const i = Math.floor(r() * P.catch_unguided.length);
      insert(unguided, withOrder("catch_unguided", i, P.catch_unguided[i]));
    }
    // group = opaque system index, so each participant hears many systems.
    const changeIdx = sampleGrouped(P.change, (t) => `${t.tok}|${t.attribute}|${t.group}`, N.change, r);
    const changeTrials = changeIdx.map((i) => withOrder("change", i, P.change[i]));
    if (P.catch_change.length && N.catch_change) {
      const i = Math.floor(r() * P.catch_change.length);
      insert(changeTrials, withOrder("catch_change", i, P.catch_change[i]));
    }
    const compare = sampleGrouped(P.compare, (t) => `${t.tok}|${t.attribute}`, N.compare, r).map((i) => withOrder("compare", i, P.compare[i]));
    return [
      { part: "Parte 1 · Continuazioni libere", intro: "intro1", trials: unguided },
      { part: "Parte 2 · Cogliere le differenze", intro: "intro2a", trials: changeTrials },
      { part: "Parte 3 · Confronto tra versioni", intro: "intro2b", trials: compare },
    ];
  }

  // ---------------------------------------------------------------- audio
  let current = null;   // {hash, card}
  const listened = {};  // hash -> seconds heard in this trial
  let lastT = null;

  function playCard(card, hash) {
    if (current && current.hash === hash && !player.paused) { player.pause(); return; }
    if (!current || current.hash !== hash) {
      if (current) setCard(current.card, false);
      current = { hash, card };
      player.src = `../audio/${hash}.mp3`;
      lastT = 0;
    }
    player.play();
  }
  function setCard(card, on) {
    card.classList.toggle("playing", on);
    card.querySelector(".play").innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  }
  const ICON_PLAY = '<svg viewBox="0 0 16 16"><path fill="currentColor" d="M4 2.5v11l9.5-5.5z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 16 16"><path fill="currentColor" d="M3.5 2h3v12h-3zM9.5 2h3v12h-3z"/></svg>';
  player.addEventListener("play", () => current && setCard(current.card, true));
  player.addEventListener("pause", () => current && setCard(current.card, false));
  player.addEventListener("ended", () => {
    if (!current) return;
    listened[current.hash] = Math.max(listened[current.hash] || 0, player.duration || 0);
    setCard(current.card, false);
    onListen();
  });
  (function tick() {
    if (current && !player.paused) {
      const t = player.currentTime;
      if (lastT != null && t > lastT && t - lastT < 0.5) listened[current.hash] = (listened[current.hash] || 0) + (t - lastT);
      lastT = t;
      const d = survey.clips[current.hash].audio || player.duration || 1;
      current.card.querySelector(".prog-fill").style.width = `${Math.min(100, (t / d) * 100)}%`;
      current.card.querySelector(".time").textContent = `${fmt(t)} / ${fmt(d)}`;
      const inPrompt = t < survey.clips[current.hash].prompt_end;
      const ph = current.card.querySelector(".phase");
      ph.textContent = inPrompt ? "ora: incipit" : "ora: continuazione";
      ph.classList.toggle("cont", !inPrompt);
      onListen();
    } else if (current) {
      lastT = player.currentTime;
    }
    requestAnimationFrame(tick);
  })();
  let onListen = () => {};

  function clipCard(hash, label, extraCls = "") {
    const c = survey.clips[hash];
    const card = h("div", "clip scard " + extraCls);
    const top = h("div", "clip-top");
    const btn = h("button", "play", ICON_PLAY);
    btn.setAttribute("aria-label", "Riproduci " + label);
    top.append(btn, h("div", "clip-label", label), h("span", "heard", ""));
    const dur = c.audio || c.duration;
    const share = Math.min(100, (c.prompt_end / dur) * 100);
    // Labels sit above the bar so "prompt" is always readable, however short it is.
    const labels = h("div", "prog-labels");
    const lp = h("span", "lab-prompt", "incipit");
    const lc = h("span", "lab-cont", '<span class="long">continuazione · da valutare</span><span class="short">da valutare</span>');
    lc.style.left = `max(${share}%, 64px)`;
    labels.append(lp, lc);
    const prog = h("div", "prog");
    const pr = h("div", "prog-prompt");
    pr.style.width = `${share}%`;
    prog.append(pr, h("div", "prog-fill"));
    const foot = h("div", "clip-foot");
    foot.append(h("span", "time", fmt(dur)), h("span", "phase", ""));
    card.append(top, labels, prog, foot);
    btn.addEventListener("click", () => playCard(card, hash));
    prog.addEventListener("click", (e) => {
      const r = prog.getBoundingClientRect();
      playCard(card, hash);
      player.currentTime = ((e.clientX - r.left) / r.width) * dur;
      lastT = player.currentTime;
    });
    card.dataset.hash = hash;
    return card;
  }
  const heardEnough = (hash) => TEST || (listened[hash] || 0) >= MIN_LISTEN * survey.clips[hash].duration - 0.3;

  // -------------------------------------------------------------- screens
  let survey, state;

  function screen(...nodes) {
    player.pause();
    current = null;
    app.innerHTML = "";
    app.append(...nodes);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function topbar() {
    const tb = document.getElementById("topbar");
    if (state.step !== "trials") { tb.hidden = true; return; }
    tb.hidden = false;
    const total = state.plan.reduce((s, p) => s + p.trials.length, 0);
    const done = state.plan.slice(0, state.part).reduce((s, p) => s + p.trials.length, 0) + state.trial;
    document.getElementById("part-label").textContent = state.plan[state.part].part;
    document.getElementById("count").textContent = `${done + 1} / ${total}`;
    document.getElementById("bar-fill").style.width = `${(done / total) * 100}%`;
  }

  // Onboarding: welcome -> how it works -> privacy & consent -> about you.
  const ONBOARD = ["welcome", "how", "consent", "about"];
  function dots(step) {
    const d = h("div", "dots");
    ONBOARD.forEach((x, i) => d.append(h("span", i <= ONBOARD.indexOf(step) ? "on" : "")));
    return d;
  }
  function nav(nextLabel, onNext, back) {
    const go = h("button", "btn primary", nextLabel);
    go.addEventListener("click", onNext);
    const kids = [];
    if (back) {
      const b = h("button", "btn ghost back", "Indietro");
      b.addEventListener("click", typeof back === "function" ? back : () => { state.step = back; save(); route(); });
      kids.push(b);
    }
    kids.push(go);
    return { row: row("actions", ...kids), go };
  }
  const goto = (step) => () => { state.step = step; save(); route(); };

  function welcome() {
    const s = h("section", "card-screen");
    s.append(
      dots("welcome"),
      h("p", "eyebrow", "Studio di ascolto · circa 20 minuti"),
      h("h1", "s-title", "Aiutaci a scoprire come suona la musica composta da un computer"),
      h("p", "lede", "Abbiamo sviluppato dei modelli che compongono musica e vorremmo sapere che impressione fa a chi la ascolta. Non servono conoscenze musicali: basta ascoltare e dirci cosa ne pensi."),
      h("div", "steps3",
        '<div><b>1</b><span>Ascolta</span><small>brevi clip da 3 a 20 secondi</small></div>' +
        '<div><b>2</b><span>Confronta</span><small>qualche versione alla volta</small></div>' +
        '<div><b>3</b><span>Scegli</span><small>clicca la risposta che preferisci, oppure salta</small></div>'),
    );
    if (CFG.contact) s.append(h("p", "hint contact", `Hai domande sullo studio? Scrivi a <a href="mailto:${CFG.contact}">${CFG.contact}</a>`));
    s.append(nav("Iniziamo", goto("how")).row);
    screen(s);
  }

  function how() {
    const s = h("section", "card-screen");
    s.append(
      dots("how"),
      h("h2", "s-title", "Come sono fatte le clip"),
      h("p", "lede", "Ogni clip si apre con qualche secondo di un brano musicale reale: l'<b>incipit</b>, identico in tutte le versioni di una stessa domanda. Poi la musica <b>prosegue</b>: è questa continuazione che ti chiediamo di valutare."),
      h("div", "anatomy",
        '<div class="an-bar"><div class="an-prompt">incipit<small>uguale in ogni versione</small></div>' +
        '<div class="an-cont">continuazione<small>la parte da valutare</small></div></div>'),
      h("p", "hint", "Prova: premi play e segui la barra. La scritta sopra la barra ti segnala quando inizia la continuazione."),
    );
    const ex = state.exampleClip && survey.clips[state.exampleClip] ? state.exampleClip : null;
    if (ex) s.append(row("sgrid one", clipCard(ex, "Esempio")));
    s.append(nav("Ho capito", goto("consent"), "welcome").row);
    screen(s);
  }

  function consentPage() {
    const s = h("section", "card-screen");
    s.append(
      dots("consent"),
      h("h2", "s-title", "Prima di iniziare"),
      h("ul", "facts",
        "<li>Meglio ascoltare in <b>cuffia</b>, possibilmente in un ambiente tranquillo.</li>" +
        "<li>Lo studio è <b>anonimo</b>. Conserviamo solo le tue risposte, un codice partecipante casuale e due domande facoltative su di te.</li>" +
        "<li>Le risposte saranno usate per una tesi di laurea magistrale e potranno comparire, in forma aggregata, in pubblicazioni scientifiche.</li>" +
        "<li>Puoi <b>saltare</b> qualunque domanda e interrompere quando vuoi. Le risposte vengono inviate solo alla fine.</li>" +
        (CFG.contact ? `<li>Per domande: <a href="mailto:${CFG.contact}">${CFG.contact}</a></li>` : "")),
    );
    const consent = h("label", "check", `<input type="checkbox" id="consent"> Ho almeno 18 anni e accetto di partecipare.`);
    const { row: r, go } = nav("Continua", () => { state.consent = new Date().toISOString(); goto("about")(); }, "how");
    go.disabled = !state.consent;
    consent.querySelector("input").checked = !!state.consent;
    consent.querySelector("input").addEventListener("change", (e) => { go.disabled = !e.target.checked; });
    s.append(consent, r);
    screen(s);
  }

  function about() {
    const s = h("section", "card-screen");
    s.append(dots("about"), h("h2", "s-title", "Parlaci un po' di te"),
      h("p", "lede", "Facoltativo, ma ci aiuta a interpretare meglio le risposte."));
    const q = (name, label, opts) => {
      const box = h("fieldset", "q");
      box.append(h("legend", null, label));
      const chips = h("div", "chips");
      for (const [o, label] of opts) {
        const b = h("button", "chip", label);
        b.type = "button";
        b.addEventListener("click", () => {
          state.background[name] = o;
          chips.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === b));
          save();
        });
        if (state.background[name] === o) b.classList.add("on");
        chips.append(b);
      }
      box.append(chips);
      return box;
    };
    s.append(
      q("training", "Formazione musicale", [["None", "Nessuna"], ["Some (a few years, hobby)", "Un po' (qualche anno, per passione)"], ["Extensive (performer, music degree, producer)", "Approfondita (musicista, studi in conservatorio, produttore)"]]),
      q("device", "Come stai ascoltando?", [["Headphones", "In cuffia"], ["External speakers", "Con casse esterne"], ["Laptop / phone speakers", "Dagli altoparlanti del portatile / telefono"]]),
    );
    s.append(nav("Iniziamo l'ascolto", () => { state.step = "trials"; state.part = 0; state.trial = -1; save(); route(); }, "consent").row);
    screen(s);
  }

  const INTROS = {
    intro1: {
      kicker: "Parte 1 di 3", title: "Quale continuazione suona meglio?",
      text: "Ascolterai quattro versioni (<b>da A a D</b>) dello stesso incipit, ognuna con una continuazione diversa. Ascoltale tutte, poi indica quella che ti sembra <b>più musicale</b> e quella che ti sembra <b>meno musicale</b>.",
      tip: "Sta a te decidere cosa sia “musicale”: naturale, coerente, una continuazione convincente dell'incipit.",
      preview: '<div class="pv-label">Più musicale</div><div class="chips"><span class="chip">A</span><span class="chip on">B</span><span class="chip">C</span><span class="chip">D</span></div>',
    },
    intro2a: {
      kicker: "Parte 2 di 3", title: "Noti qualche differenza?",
      text: "Ora ascolterai un brano di <b>riferimento</b> e una <b>seconda versione</b> dello stesso brano, che potrebbe essere stata modificata in un solo aspetto. La domanda ti indica a cosa prestare attenzione:",
      list: "<li><b>Intensità</b>, da <i>piano</i> (delicato) a <i>forte</i> (intenso)</li>" +
            "<li><b>Durata delle note</b>, da <i>staccato</i> (brevi, separate) a <i>legato</i> (lunghe, unite)</li>" +
            "<li><b>Altezza</b>, da suoni gravi (profondi) a suoni acuti (brillanti)</li>",
      tip: "Spesso la differenza è minima, o non c'è affatto: “nessuna differenza” è una risposta del tutto valida.",
      preview: '<div class="pv-label">La seconda versione è…</div><div class="chips"><span class="chip">decisamente più piano</span><span class="chip">leggermente più piano</span><span class="chip on">nessuna differenza</span><span class="chip">leggermente più forte</span><span class="chip">decisamente più forte</span></div>',
    },
    intro2b: {
      kicker: "Parte 3 di 3", title: "Stessa modifica, versioni diverse",
      text: "Ultima parte. Tutte le versioni di una domanda hanno subito la stessa modifica (per esempio, sono state rese tutte più forti). Indica quella che ti sembra <b>più musicale</b> e quella che ti sembra <b>meno musicale</b>.",
      tip: "Conta quanto suona bene, non quanto si sente la modifica.",
      preview: '<div class="pv-label">Meno musicale</div><div class="chips"><span class="chip">A</span><span class="chip">B</span><span class="chip on">C</span><span class="chip">D</span></div>',
    },
  };

  function partIntro() {
    const it = INTROS[state.plan[state.part].intro];
    const s = h("section", "card-screen");
    s.append(
      h("p", "eyebrow", it.kicker),
      h("h2", "s-title", it.title),
      h("p", "lede", it.text),
      ...(it.list ? [h("ul", "facts attrs", it.list)] : []),
      h("div", "preview", it.preview),
      h("ul", "facts tips",
        `<li>${it.tip}</li>` +
        "<li>Puoi riascoltare le clip tutte le volte che vuoi e cliccare sulla barra per spostarti avanti o indietro.</li>" +
        "<li>Valuta la <b>continuazione</b>, cioè la parte che segue l'incipit.</li>" +
        "<li>Nel dubbio, puoi sempre saltare la domanda.</li>"),
    );
    const back = () => {
      if (state.part === 0) { state.step = "about"; }
      else { state.part -= 1; state.trial = state.plan[state.part].trials.length - 1; }
      save(); route();
    };
    s.append(nav("Comincia", () => { state.trial = 0; save(); route(); }, back).row);
    screen(s);
  }

  function trialScreen() {
    const p = state.plan[state.part];
    const t = p.trials[state.trial];
    const pool = survey.pool[t.kind][t.pool_index];
    for (const k in listened) delete listened[k];
    const prev = state.responses.find((r) => r.part === state.part && r.trial === state.trial);
    if (prev) Object.assign(listened, prev.listened);   // already heard: stays unlocked
    const started = performance.now();
    const s = h("section", "trial");
    const answer = prev && !prev.answer.skipped ? { ...prev.answer } : {};
    const submit = h("button", "btn primary", "Avanti");
    submit.disabled = true;
    let ready = () => false;

    if (t.kind === "unguided" || t.kind === "catch_unguided" || t.kind === "compare") {
      s.append(h("h2", "t-q", t.kind === "compare"
        ? `In tutte queste versioni è stata modificata <b>${survey.attr_name[pool.attribute]}</b>, nello stesso senso. ${t.order.length === 2 ? "Quale ti sembra più musicale?" : "Quale ti sembra più musicale, e quale meno?"}`
        : "Quale continuazione ti sembra più musicale, e quale meno?"));
      const grid = h("div", "sgrid");
      t.order.forEach((hash, i) => grid.append(clipCard(hash, `Versione ${LETTERS[i]}`)));
      s.append(grid);
      const pick = (name, label) => {
        const box = h("fieldset", "q");
        box.append(h("legend", null, label));
        const row = h("div", "chips");
        t.order.forEach((hash, i) => {
          const b = h("button", "chip letter", LETTERS[i]);
          b.type = "button";
          if (answer[name] === hash) b.classList.add("on");
          b.addEventListener("click", () => {
            answer[name] = hash;
            row.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === b));
            update();
          });
          row.append(b);
        });
        box.append(row);
        return box;
      };
      if (t.order.length === 2) {
        // Two versions: one forced choice; the other one is recorded as "worst".
        const box = pick("best", "Quale ti sembra più musicale?");
        box.addEventListener("click", () => { if (answer.best) answer.worst = t.order.find((x) => x !== answer.best); update(); });
        s.append(row("answers", box));
      } else {
        s.append(row("answers", pick("best", "Più musicale"), pick("worst", "Meno musicale")));
      }
      ready = () => answer.best && answer.worst && answer.best !== answer.worst;
    } else {
      const attr = pool.attribute;
      s.append(h("h2", "t-q", `Rispetto al riferimento, com'è <b>${survey.attr_name[attr]}</b> nella seconda versione?`));
      const grid = h("div", "sgrid two");
      grid.append(clipCard(pool.reference, "Riferimento", "ref"), clipCard(pool.test, "Seconda versione"));
      s.append(grid);
      const box = h("fieldset", "q");
      box.append(h("legend", null, "La seconda versione è…"));
      const scaleRow = h("div", "scale5");
      survey.scale[attr].forEach((lab, i) => {
        const b = h("button", "chip", lab);
        b.type = "button";
        if (answer.scale === i - 2) b.classList.add("on");
        b.addEventListener("click", () => {
          answer.scale = i - 2;   // -2 .. +2
          scaleRow.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === b));
          update();
        });
        scaleRow.append(b);
      });
      box.append(scaleRow);
      s.append(row("answers", box));
      ready = () => answer.scale != null;
    }

    // Optional: which versions sounded cacophonous? (multi-select, per clip)
    const cacoClips = t.order || [pool.reference, pool.test];
    const cacoNames = t.order ? t.order.map((_, i) => LETTERS[i]) : ["Riferimento", "Seconda versione"];
    if (t.order === null && pool.reference === pool.test) cacoClips.pop();   // same clip twice
    answer.cacophonous = answer.cacophonous || [];
    const caco = h("fieldset", "q caco");
    caco.append(h("legend", null, "Facoltativo: qualche versione ti è sembrata <b>cacofonica</b> (stridente, caotica, con note quasi a caso)? Selezionala."));
    const cacoRow = h("div", "chips");
    cacoClips.forEach((hash, i) => {
      const b = h("button", "chip tick", cacoNames[i]);
      b.type = "button";
      if (answer.cacophonous.includes(hash)) b.classList.add("on");
      b.addEventListener("click", () => {
        const on = !answer.cacophonous.includes(hash);
        answer.cacophonous = on ? [...answer.cacophonous, hash] : answer.cacophonous.filter((x) => x !== hash);
        b.classList.toggle("on", on);
      });
      cacoRow.append(b);
    });
    caco.append(cacoRow);
    // The separator lives on a wrapper: a fieldset's own top border is drawn
    // through its legend, which showed up as a stray dashed line after the text.
    s.querySelector(".answers").append(row("caco-wrap", caco));

    const help = pool.attribute ? ` <span class="help">${survey.attr_help[pool.attribute]}</span>` : "";
    s.querySelector(".t-q").after(h("p", "t-hint", (t.kind.includes("change")
      ? "Entrambe le versioni partono dallo stesso incipit. Concentrati sulla <b>continuazione</b>, dopo la parte grigia della barra."
      : "Tutte le versioni partono dallo stesso incipit (la parte grigia della barra). Valuta solo la <b>continuazione</b>.") + help));
    const hashes = t.order || [pool.reference, pool.test];
    const status = h("p", "hint", "");
    function update() {
      const pending = [...new Set(hashes)].filter((x) => !heardEnough(x)).length;
      app.querySelectorAll(".scard").forEach((c) => c.classList.toggle("heard-ok", heardEnough(c.dataset.hash)));
      app.querySelectorAll(".answers .chip").forEach((c) => { c.disabled = pending > 0; });
      status.textContent = pending ? `${pending === 1 ? "Ti manca ancora una clip" : `Ti mancano ancora ${pending} clip`} da ascoltare per sbloccare le risposte (oppure salta la domanda).` : "";
      submit.disabled = pending > 0 || !ready();
    }
    onListen = update;
    const skip = h("button", "btn ghost", "Salta la domanda");
    skip.addEventListener("click", () => record({ skipped: true }));
    submit.addEventListener("click", () => record({ ...answer }));
    function record(ans) {
      // Going back and answering again replaces the earlier answer.
      state.responses = state.responses.filter((r) => !(r.part === state.part && r.trial === state.trial));
      state.responses.push({
        part: state.part, trial: state.trial, kind: t.kind, pool_index: t.pool_index,
        order: hashes, answer: ans,
        listened: Object.fromEntries(hashes.map((x) => [x, +(listened[x] || 0).toFixed(1)])),
        ms: Math.round(performance.now() - started),
      });
      state.trial += 1;
      save();
      route();
    }
    const back = h("button", "btn ghost back", "Indietro");
    back.addEventListener("click", () => { state.trial -= 1; save(); route(); });
    s.append(status, row("actions", back, skip, submit));
    screen(s);
    update();
  }

  function finish() {
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", "Abbiamo quasi finito"),
      h("p", "lede", "Vuoi aggiungere qualcosa? (facoltativo) Per esempio a cosa hai fatto caso durante l'ascolto, o se qualcosa ti è sembrato strano."));
    const ta = h("textarea", "comments");
    ta.value = state.comments || "";
    ta.addEventListener("input", () => { state.comments = ta.value; save(); });
    const copy = h("div", "copybox");
    copy.append(
      h("label", "check", `<input type="checkbox" id="wantcopy"> Voglio ricevere una copia delle mie risposte via email`),
      h("input", "email"),
      h("p", "hint", "Facoltativo. Useremo il tuo indirizzo solo per inviarti la copia: non verrà salvato insieme alle risposte."));
    const emailIn = copy.querySelector("input.email");
    emailIn.type = "email"; emailIn.placeholder = "tu@esempio.it"; emailIn.hidden = true;
    copy.querySelector("#wantcopy").addEventListener("change", (e) => { emailIn.hidden = !e.target.checked; if (e.target.checked) emailIn.focus(); });
    const go = h("button", "btn primary", "Invia le risposte");
    const msg = h("p", "hint", "");
    go.addEventListener("click", async () => {
      state.finished = new Date().toISOString();
      save();
      const payload = { ...state, version: survey.version, ua: navigator.userAgent, summary: summaryText(), lang: "it" };
      const wantCopy = copy.querySelector("#wantcopy").checked;
      if (wantCopy) {
        if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(emailIn.value.trim())) {
          msg.textContent = "Inserisci un indirizzo email valido, oppure togli la spunta dalla richiesta di copia.";
          return;
        }
        payload.copy_to = emailIn.value.trim();   // used for sending only; the script drops it before storing
      }
      if (!CFG.endpoint) {
        downloadJSON(payload);
        state.step = "done"; save(); route();
        return;
      }
      go.disabled = true;
      msg.textContent = "Invio in corso…";
      try {
        await fetch(CFG.endpoint, { method: "POST", mode: "no-cors", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(payload) });
        state.step = "done"; state.sent = true; save(); route();
      } catch (e) {
        go.disabled = false;
        msg.innerHTML = `Invio non riuscito (${e}). Riprova, oppure <a href="#" id="dl">scarica le tue risposte</a> e inviale per email.`;
        msg.querySelector("#dl").addEventListener("click", (ev) => { ev.preventDefault(); downloadJSON(payload); });
      }
    });
    s.append(ta, copy, msg, row("actions", go));
    screen(s);
  }

  function summaryText() {
    const lines = [`Studio di ascolto — le tue risposte (partecipante ${state.id})`, ""];
    state.plan.forEach((p, pi) => {
      lines.push(p.part);
      p.trials.forEach((t, ti) => {
        const r = state.responses.find((x) => x.part === pi && x.trial === ti);
        const pool = survey.pool[t.kind][t.pool_index];
        const name = (hash) => t.order ? `Versione ${LETTERS[t.order.indexOf(hash)]}` : (hash === pool.test && hash !== pool.reference ? "Seconda versione" : "Riferimento");
        let a;
        if (!r) a = "senza risposta";
        else if (r.answer.skipped) a = "saltata";
        else if (r.answer.scale != null) a = `seconda versione: ${survey.scale[pool.attribute][r.answer.scale + 2]}`;
        else a = `più musicale: ${name(r.answer.best)}` + (t.order.length > 2 ? `, meno musicale: ${name(r.answer.worst)}` : "");
        if (r && r.answer.cacophonous && r.answer.cacophonous.length) a += ` · cacofonica: ${r.answer.cacophonous.map(name).join(", ")}`;
        lines.push(`  D${ti + 1}: ${a}`);
      });
      lines.push("");
    });
    if (state.comments) lines.push(`Commenti: ${state.comments}`);
    lines.push("", "Grazie per aver partecipato!");
    return lines.join("\n");
  }

  function downloadJSON(obj) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" }));
    a.download = `survey_${obj.id}.json`;
    a.click();
  }

  function done() {
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", "Grazie!"),
      h("p", "lede", CFG.endpoint ? "Le tue risposte sono state registrate. Ora puoi chiudere la pagina." :
        "Modalità di prova: nessun endpoint configurato, quindi le risposte sono state scaricate come file invece di essere inviate."),
      h("p", "hint", `Codice partecipante: ${state.id}`));
    screen(s);
  }

  function route() {
    topbar();
    if (state.step === "welcome") return welcome();
    if (state.step === "how") return how();
    if (state.step === "consent") return consentPage();
    if (state.step === "about") return about();
    if (state.step === "finish") return finish();
    if (state.step === "done") return done();
    // trials
    const p = state.plan[state.part];
    if (state.trial < 0) return partIntro();
    if (state.trial >= p.trials.length) {
      state.part += 1; state.trial = -1;
      if (state.part >= state.plan.length) state.step = "finish";
      save();
      return route();
    }
    trialScreen();
  }

  fetch("../survey.json").then((r) => r.json()).then((s) => {
    survey = Object.assign(s, IT);
    const saved = JSON.parse(localStorage.getItem(STORE) || "null");
    if (saved && saved.version === s.version && saved.step !== "done") {
      state = saved;
    } else {
      const id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).slice(0, 13);
      state = { id, version: s.version, started: new Date().toISOString(), step: "welcome",
                background: {}, plan: makePlan(s, id), part: 0, trial: -1, responses: [] };
      const used = new Set(state.plan[0].trials.filter((t) => t.kind === "unguided").map((t) => t.pool_index));
      const spare = s.pool.unguided.findIndex((_, i) => !used.has(i));
      if (spare >= 0) state.exampleClip = s.pool.unguided[spare].options[0];
      save();
    }
    route();
  }).catch((e) => { app.innerHTML = `<p class="intro">Impossibile caricare lo studio (${e}).</p>`; });
})();
