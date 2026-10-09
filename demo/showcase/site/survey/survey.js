// Blind listening survey. Trials are sampled per participant from survey.json
// (built by demo/survey/build_survey.py); clips are opaque hashes, options are
// shown as A, B, C, D in random order. Responses go to SURVEY_CONFIG.endpoint.
(() => {
  const CFG = window.SURVEY_CONFIG || {};
  const STORE = "ebt-survey-v1";
  const app = document.getElementById("app");
  const player = document.getElementById("player");
  const LETTERS = ["A", "B", "C", "D", "E", "F"];
  const MIN_LISTEN = 0.7;   // share of a clip that must be heard before answering
  // ?test on localhost skips the listening requirement (for automated checks only).
  const TEST = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) && new URLSearchParams(location.search).has("test");

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
      { part: "Part 1 · Unguided", intro: "intro1", trials: unguided },
      { part: "Part 2 · Perceived change", intro: "intro2a", trials: changeTrials },
      { part: "Part 3 · Comparing versions", intro: "intro2b", trials: compare },
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
      player.src = `audio/${hash}.mp3`;
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
      const d = player.duration || survey.clips[current.hash].duration || 1;
      current.card.querySelector(".prog-fill").style.width = `${Math.min(100, (t / d) * 100)}%`;
      current.card.querySelector(".time").textContent = `${fmt(t)} / ${fmt(d)}`;
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
    btn.setAttribute("aria-label", "Play " + label);
    top.append(btn, h("div", "clip-label", label), h("span", "heard", ""));
    const prog = h("div", "prog");
    const pr = h("div", "prog-prompt");
    pr.style.width = `${Math.min(100, (c.prompt_end / c.duration) * 100)}%`;
    pr.title = "prompt";
    prog.append(pr, h("div", "prog-fill"));
    const foot = h("div", "clip-foot");
    foot.append(h("span", "time", fmt(c.duration)), h("span", "metric", "prompt ▸ continuation"));
    card.append(top, prog, foot);
    btn.addEventListener("click", () => playCard(card, hash));
    prog.addEventListener("click", (e) => {
      const r = prog.getBoundingClientRect();
      playCard(card, hash);
      player.currentTime = ((e.clientX - r.left) / r.width) * c.duration;
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

  function welcome() {
    const s = h("section", "card-screen");
    s.append(
      h("p", "eyebrow", "Listening study · about 20 minutes"),
      h("h1", "s-title", "How does machine-generated music sound to you?"),
      h("p", "lede", "You will hear short pieces of music. Each one starts with a few seconds taken from a real piece (the <em>prompt</em>), followed by a <em>continuation</em> generated by a computer model or written by a person. You will compare versions and tell us what you hear. There are no right or wrong answers."),
      h("ul", "facts",
        "<li><b>Headphones</b> are recommended; please use a quiet place.</li>" +
        "<li>The study is <b>anonymous</b>: we store only your answers, a random participant code and the optional background questions.</li>" +
        "<li>Your answers are used for a master's thesis and may be reported in aggregate in publications.</li>" +
        "<li>You can stop at any time; nothing is sent until the end.</li>" +
        (CFG.contact ? `<li>Questions: <a href="mailto:${CFG.contact}">${CFG.contact}</a></li>` : "")),
    );
    const consent = h("label", "check", `<input type="checkbox" id="consent"> I am 18 or older and agree to take part.`);
    const go = h("button", "btn primary", "Start");
    go.disabled = true;
    consent.querySelector("input").addEventListener("change", (e) => { go.disabled = !e.target.checked; });
    go.addEventListener("click", () => { state.consent = new Date().toISOString(); state.step = "about"; save(); route(); });
    s.append(consent, row("actions", go));
    screen(s);
  }

  function about() {
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", "About you"), h("p", "lede", "Two quick questions (optional, but they help us interpret the answers)."));
    const q = (name, label, opts) => {
      const box = h("fieldset", "q");
      box.append(h("legend", null, label));
      const row = h("div", "chips");
      for (const o of opts) {
        const b = h("button", "chip", o);
        b.type = "button";
        b.addEventListener("click", () => {
          state.background[name] = o;
          row.querySelectorAll(".chip").forEach((x) => x.classList.toggle("on", x === b));
          save();
        });
        if (state.background[name] === o) b.classList.add("on");
        row.append(b);
      }
      box.append(row);
      return box;
    };
    s.append(
      q("training", "Musical training", ["None", "Some (a few years, hobby)", "Extensive (performer, music degree, producer)"]),
      q("device", "You are listening on", ["Headphones", "External speakers", "Laptop / phone speakers"]),
    );
    const go = h("button", "btn primary", "Continue");
    go.addEventListener("click", () => { state.step = "trials"; state.part = 0; state.trial = -1; save(); route(); });
    s.append(row("actions", go));
    screen(s);
  }

  const INTROS = {
    intro1: ["Part 1 · Which continuation sounds best?",
      "Each question plays four versions (A–D) that start with the same prompt and then continue differently. Listen to all four, then choose the one that sounds <b>most musical</b> (natural, coherent, a good continuation of the prompt) and the one that sounds <b>least musical</b>."],
    intro2a: ["Part 2 · Do you hear a difference?",
      "Each question plays a <b>reference</b> and a <b>second version</b> of the same piece. Some versions were changed in one property: loudness, note length or pitch. The question names the property; tell us whether, and in which direction, the second version differs from the reference. If you don't hear a difference, say so — that is a useful answer."],
    intro2b: ["Part 3 · Same goal, different versions",
      "Each question plays several versions of the same prompt that were all changed in the same way (for example, all made louder). Choose the one that sounds <b>most musical</b> and the one that sounds <b>least musical</b>."],
  };

  function partIntro() {
    const [title, text] = INTROS[state.plan[state.part].intro];
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", title), h("p", "lede", text),
      h("p", "hint", "Answer buttons unlock once you have listened to most of every clip. Click the bar under a clip to jump within it."));
    const go = h("button", "btn primary", "Begin");
    go.addEventListener("click", () => { state.trial = 0; save(); route(); });
    s.append(row("actions", go));
    screen(s);
  }

  function trialScreen() {
    const p = state.plan[state.part];
    const t = p.trials[state.trial];
    const pool = survey.pool[t.kind][t.pool_index];
    for (const k in listened) delete listened[k];
    const started = performance.now();
    const s = h("section", "trial");
    const answer = {};
    const submit = h("button", "btn primary", "Next");
    submit.disabled = true;
    let ready = () => false;

    if (t.kind === "unguided" || t.kind === "catch_unguided" || t.kind === "compare") {
      s.append(h("h2", "t-q", t.kind === "compare"
        ? `These versions were all changed in <b>${survey.attr_name[pool.attribute]}</b> in the same direction. ${t.order.length === 2 ? "Which sounds more musical?" : "Which sounds most and least musical?"}`
        : "Which continuation sounds most and least musical?"));
      const grid = h("div", "sgrid");
      t.order.forEach((hash, i) => grid.append(clipCard(hash, `Version ${LETTERS[i]}`)));
      s.append(grid);
      const pick = (name, label) => {
        const box = h("fieldset", "q");
        box.append(h("legend", null, label));
        const row = h("div", "chips");
        t.order.forEach((hash, i) => {
          const b = h("button", "chip letter", LETTERS[i]);
          b.type = "button";
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
        const box = pick("best", "Which sounds more musical?");
        box.addEventListener("click", () => { if (answer.best) answer.worst = t.order.find((x) => x !== answer.best); update(); });
        s.append(row("answers", box));
      } else {
        s.append(row("answers", pick("best", "Most musical"), pick("worst", "Least musical")));
      }
      ready = () => answer.best && answer.worst && answer.best !== answer.worst;
    } else {
      const attr = pool.attribute;
      s.append(h("h2", "t-q", `Compared with the reference, how does the second version sound in <b>${survey.attr_name[attr]}</b>?`));
      const grid = h("div", "sgrid two");
      grid.append(clipCard(pool.reference, "Reference", "ref"), clipCard(pool.test, "Second version"));
      s.append(grid);
      const box = h("fieldset", "q");
      box.append(h("legend", null, "The second version is…"));
      const scaleRow = h("div", "scale5");
      survey.scale[attr].forEach((lab, i) => {
        const b = h("button", "chip", lab);
        b.type = "button";
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

    const hashes = t.order || [pool.reference, pool.test];
    const status = h("p", "hint", "");
    function update() {
      const pending = [...new Set(hashes)].filter((x) => !heardEnough(x)).length;
      app.querySelectorAll(".scard").forEach((c) => c.classList.toggle("heard-ok", heardEnough(c.dataset.hash)));
      app.querySelectorAll(".answers .chip").forEach((c) => { c.disabled = pending > 0; });
      status.textContent = pending ? `Listen to ${pending === 1 ? "one more clip" : `${pending} more clips`} to unlock the answers.` : "";
      submit.disabled = pending > 0 || !ready();
    }
    onListen = update;
    submit.addEventListener("click", () => {
      state.responses.push({
        part: state.part, kind: t.kind, pool_index: t.pool_index,
        order: hashes, answer: { ...answer },
        listened: Object.fromEntries(hashes.map((x) => [x, +(listened[x] || 0).toFixed(1)])),
        ms: Math.round(performance.now() - started),
      });
      state.trial += 1;
      save();
      route();
    });
    s.append(status, row("actions", submit));
    screen(s);
    update();
  }

  function finish() {
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", "Almost done"),
      h("p", "lede", "Anything you'd like to tell us? (optional) For example what you listened for, or whether something sounded odd."));
    const ta = h("textarea", "comments");
    ta.value = state.comments || "";
    ta.addEventListener("input", () => { state.comments = ta.value; save(); });
    const go = h("button", "btn primary", "Submit answers");
    const msg = h("p", "hint", "");
    go.addEventListener("click", async () => {
      state.finished = new Date().toISOString();
      save();
      const payload = { ...state, version: survey.version, ua: navigator.userAgent };
      if (!CFG.endpoint) {
        downloadJSON(payload);
        state.step = "done"; save(); route();
        return;
      }
      go.disabled = true;
      msg.textContent = "Sending…";
      try {
        await fetch(CFG.endpoint, { method: "POST", mode: "no-cors", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(payload) });
        state.step = "done"; state.sent = true; save(); route();
      } catch (e) {
        go.disabled = false;
        msg.innerHTML = `Could not send (${e}). Please try again, or <a href="#" id="dl">download your answers</a> and email them.`;
        msg.querySelector("#dl").addEventListener("click", (ev) => { ev.preventDefault(); downloadJSON(payload); });
      }
    });
    s.append(ta, msg, row("actions", go));
    screen(s);
  }

  function downloadJSON(obj) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" }));
    a.download = `survey_${obj.id}.json`;
    a.click();
  }

  function done() {
    const s = h("section", "card-screen");
    s.append(h("h2", "s-title", "Thank you!"),
      h("p", "lede", CFG.endpoint ? "Your answers have been recorded. You can close this page." :
        "Test mode: no endpoint is configured, so your answers were downloaded as a file instead of being sent."),
      h("p", "hint", `Participant code: ${state.id}`));
    screen(s);
  }

  function route() {
    topbar();
    if (state.step === "welcome") return welcome();
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

  fetch("survey.json").then((r) => r.json()).then((s) => {
    survey = s;
    const saved = JSON.parse(localStorage.getItem(STORE) || "null");
    if (saved && saved.version === s.version && saved.step !== "done") {
      state = saved;
    } else {
      const id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)).slice(0, 13);
      state = { id, version: s.version, started: new Date().toISOString(), step: "welcome",
                background: {}, plan: makePlan(s, id), part: 0, trial: -1, responses: [] };
      save();
    }
    route();
  }).catch((e) => { app.innerHTML = `<p class="intro">Could not load the study (${e}).</p>`; });
})();
