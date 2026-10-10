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
      const d = survey.clips[current.hash].audio || player.duration || 1;
      current.card.querySelector(".prog-fill").style.width = `${Math.min(100, (t / d) * 100)}%`;
      current.card.querySelector(".time").textContent = `${fmt(t)} / ${fmt(d)}`;
      const inPrompt = t < survey.clips[current.hash].prompt_end;
      const ph = current.card.querySelector(".phase");
      ph.textContent = inPrompt ? "now: prompt" : "now: continuation";
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
    btn.setAttribute("aria-label", "Play " + label);
    top.append(btn, h("div", "clip-label", label), h("span", "heard", ""));
    const dur = c.audio || c.duration;
    const share = Math.min(100, (c.prompt_end / dur) * 100);
    // Labels sit above the bar so "prompt" is always readable, however short it is.
    const labels = h("div", "prog-labels");
    const lp = h("span", "lab-prompt", "prompt");
    const lc = h("span", "lab-cont", '<span class="long">continuation · judge this</span><span class="short">judge this</span>');
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
    // Progress is kept across reloads; this discards it and starts a new participant.
    if (state.step !== "welcome" && state.step !== "done") {
      const again = h("button", "restart", "Start over");
      again.addEventListener("click", () => {
        if (!confirm("Start the survey again from the beginning? Your answers so far will be deleted.")) return;
        localStorage.removeItem(STORE);
        location.reload();
      });
      app.append(again);
    }
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
      const b = h("button", "btn ghost back", "Back");
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
      h("p", "eyebrow", "Listening study · about 20 minutes"),
      h("h1", "s-title", "Help us find out how computer-made music sounds"),
      h("p", "lede", "We built models that write music, and we'd like to know how their music sounds to real people. No musical knowledge needed: just listen and tell us what you hear."),
      h("div", "steps3",
        '<div><b>1</b><span>Listen</span><small>short clips of 3–20 seconds</small></div>' +
        '<div><b>2</b><span>Compare</span><small>a few versions at a time</small></div>' +
        '<div><b>3</b><span>Choose</span><small>click what fits best, or skip</small></div>'),
    );
    if (CFG.contact) s.append(h("p", "hint contact", `Questions about the study? Write to <a href="mailto:${CFG.contact}">${CFG.contact}</a>`));
    s.append(nav("Let's go", goto("how")).row);
    screen(s);
  }

  function how() {
    const s = h("section", "card-screen");
    s.append(
      dots("how"),
      h("h2", "s-title", "How each clip works"),
      h("p", "lede", "Every clip starts with a few seconds of a real piece of music: the <b>prompt</b>. It is the same in all versions of a question. After it, the music <b>continues</b>, and that is the part we ask you to judge."),
      h("div", "anatomy",
        '<div class="an-bar"><div class="an-prompt">prompt<small>same in every version</small></div>' +
        '<div class="an-cont">continuation<small>what you judge</small></div></div>'),
      h("p", "hint", "Try it: press play and watch the bar. The label under the bar tells you when the continuation starts."),
    );
    const ex = state.exampleClip && survey.clips[state.exampleClip] ? state.exampleClip : null;
    if (ex) s.append(row("sgrid one", clipCard(ex, "Example")));
    s.append(nav("Got it", goto("consent"), "welcome").row);
    screen(s);
  }

  function consentPage() {
    const s = h("section", "card-screen");
    s.append(
      dots("consent"),
      h("h2", "s-title", "Before we start"),
      h("ul", "facts",
        "<li><b>Headphones</b> work best; a quiet place helps.</li>" +
        "<li>The study is <b>anonymous</b>. We store only your answers, a random participant code and two optional background questions.</li>" +
        "<li>Answers are used for a master's thesis and may appear in aggregate in publications.</li>" +
        "<li>You can <b>skip</b> any question, and stop at any time. Nothing is sent until the end.</li>" +
        (CFG.contact ? `<li>Questions: <a href="mailto:${CFG.contact}">${CFG.contact}</a></li>` : "")),
    );
    const consent = h("label", "check", `<input type="checkbox" id="consent"> I am 18 or older and agree to take part.`);
    const { row: r, go } = nav("Continue", () => { state.consent = new Date().toISOString(); goto("about")(); }, "how");
    go.disabled = !state.consent;
    consent.querySelector("input").checked = !!state.consent;
    consent.querySelector("input").addEventListener("change", (e) => { go.disabled = !e.target.checked; });
    s.append(consent, r);
    screen(s);
  }

  function about() {
    const s = h("section", "card-screen");
    s.append(dots("about"), h("h2", "s-title", "A little about you"),
      h("p", "lede", "Optional, but it helps us interpret the answers."));
    const q = (name, label, opts) => {
      const box = h("fieldset", "q");
      box.append(h("legend", null, label));
      const chips = h("div", "chips");
      for (const o of opts) {
        const b = h("button", "chip", o);
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
      q("training", "Musical training", ["None", "Some (a few years, hobby)", "Extensive (performer, music degree, producer)"]),
      q("device", "You are listening on", ["Headphones", "External speakers", "Laptop / phone speakers"]),
    );
    s.append(nav("Start listening", () => { state.step = "trials"; state.part = 0; state.trial = -1; save(); route(); }, "consent").row);
    screen(s);
  }

  const INTROS = {
    intro1: {
      kicker: "Part 1 of 3", title: "Which continuation sounds best?",
      text: "You'll hear four versions, <b>A to D</b>, of the same prompt, each continued differently. Listen to all four, then pick the one that sounds <b>most musical</b> and the one that sounds <b>least musical</b>.",
      tip: "“Musical” is up to you: natural, coherent, a convincing continuation of the prompt.",
      preview: '<div class="pv-label">Most musical</div><div class="chips"><span class="chip">A</span><span class="chip on">B</span><span class="chip">C</span><span class="chip">D</span></div>',
    },
    intro2a: {
      kicker: "Part 2 of 3", title: "Do you hear a difference?",
      text: "Now you'll hear a <b>reference</b> and a <b>second version</b> of the same music. The second version may have been changed in one way. The question tells you which one to listen for:",
      list: "<li><b>Loudness</b>, from <i>piano</i> (soft) to <i>forte</i> (loud)</li>" +
            "<li><b>Note length</b>, from <i>staccato</i> (short, detached) to <i>legato</i> (long, connected)</li>" +
            "<li><b>Pitch</b>, from low (deep) to high (bright) tones</li>",
      tip: "Often the change is subtle, or there is none. “No difference” is a perfectly good answer.",
      preview: '<div class="pv-label">The second version is…</div><div class="chips"><span class="chip">clearly softer</span><span class="chip">slightly softer</span><span class="chip on">no difference</span><span class="chip">slightly louder</span><span class="chip">clearly louder</span></div>',
    },
    intro2b: {
      kicker: "Part 3 of 3", title: "Same change, different versions",
      text: "Last part. All versions in a question were changed in the same way (for example, all made louder). Pick the one that sounds <b>most musical</b>, and the one that sounds <b>least musical</b>.",
      tip: "You're judging how good it sounds, not how strong the change is.",
      preview: '<div class="pv-label">Least musical</div><div class="chips"><span class="chip">A</span><span class="chip">B</span><span class="chip on">C</span><span class="chip">D</span></div>',
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
        "<li>Replay clips as often as you like; click the bar to jump around.</li>" +
        "<li>Judge the <b>continuation</b>, the part after the prompt.</li>" +
        "<li>Not sure? You can skip any question.</li>"),
    );
    const back = () => {
      if (state.part === 0) { state.step = "about"; }
      else { state.part -= 1; state.trial = state.plan[state.part].trials.length - 1; }
      save(); route();
    };
    s.append(nav("Begin", () => { state.trial = 0; save(); route(); }, back).row);
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
    const cacoNames = t.order ? t.order.map((_, i) => LETTERS[i]) : ["Reference", "Second version"];
    if (t.order === null && pool.reference === pool.test) cacoClips.pop();   // same clip twice
    answer.cacophonous = answer.cacophonous || [];
    const caco = h("fieldset", "q caco");
    caco.append(h("legend", null, "Optional: did any version sound <b>cacophonous</b> (harsh, chaotic, random-sounding notes)? Tick those that did."));
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
      ? "Both start with the same prompt. Listen for the difference in the <b>continuation</b>, after the grey part of the bar."
      : "All versions start with the same prompt (grey part of the bar). Judge only the <b>continuation</b> that follows.") + help));
    const hashes = t.order || [pool.reference, pool.test];
    const status = h("p", "hint", "");
    function update() {
      const pending = [...new Set(hashes)].filter((x) => !heardEnough(x)).length;
      app.querySelectorAll(".scard").forEach((c) => c.classList.toggle("heard-ok", heardEnough(c.dataset.hash)));
      app.querySelectorAll(".answers .chip").forEach((c) => { c.disabled = pending > 0; });
      status.textContent = pending ? `Listen to ${pending === 1 ? "one more clip" : `${pending} more clips`} to unlock the answers, or skip this question.` : "";
      submit.disabled = pending > 0 || !ready();
    }
    onListen = update;
    const skip = h("button", "btn ghost", "Skip question");
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
    const back = h("button", "btn ghost back", "Back");
    back.addEventListener("click", () => { state.trial -= 1; save(); route(); });
    s.append(status, row("actions", back, skip, submit));
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
    const copy = h("div", "copybox");
    copy.append(
      h("label", "check", `<input type="checkbox" id="wantcopy"> Email me a copy of my answers`),
      h("input", "email"),
      h("p", "hint", "Optional. Your email address is used only to send you the copy; it is not stored with your answers."));
    const emailIn = copy.querySelector("input.email");
    emailIn.type = "email"; emailIn.placeholder = "you@example.com"; emailIn.hidden = true;
    copy.querySelector("#wantcopy").addEventListener("change", (e) => { emailIn.hidden = !e.target.checked; if (e.target.checked) emailIn.focus(); });
    const go = h("button", "btn primary", "Submit answers");
    const msg = h("p", "hint", "");
    go.addEventListener("click", async () => {
      state.finished = new Date().toISOString();
      save();
      const payload = { ...state, version: survey.version, ua: navigator.userAgent, summary: summaryText() };
      const wantCopy = copy.querySelector("#wantcopy").checked;
      if (wantCopy) {
        if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(emailIn.value.trim())) {
          msg.textContent = "Please enter a valid email address, or untick the copy option.";
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
    s.append(ta, copy, msg, row("actions", go));
    screen(s);
  }

  function summaryText() {
    const lines = [`Listening study — your answers (participant ${state.id})`, ""];
    state.plan.forEach((p, pi) => {
      lines.push(p.part);
      p.trials.forEach((t, ti) => {
        const r = state.responses.find((x) => x.part === pi && x.trial === ti);
        const pool = survey.pool[t.kind][t.pool_index];
        const name = (hash) => t.order ? `Version ${LETTERS[t.order.indexOf(hash)]}` : (hash === pool.test && hash !== pool.reference ? "Second version" : "Reference");
        let a;
        if (!r) a = "not answered";
        else if (r.answer.skipped) a = "skipped";
        else if (r.answer.scale != null) a = `second version: ${survey.scale[pool.attribute][r.answer.scale + 2]}`;
        else a = `most musical: ${name(r.answer.best)}` + (t.order.length > 2 ? `, least musical: ${name(r.answer.worst)}` : "");
        if (r && r.answer.cacophonous && r.answer.cacophonous.length) a += ` · cacophonous: ${r.answer.cacophonous.map(name).join(", ")}`;
        lines.push(`  Q${ti + 1}: ${a}`);
      });
      lines.push("");
    });
    if (state.comments) lines.push(`Comments: ${state.comments}`);
    lines.push("", "Thank you for taking part!");
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
    s.append(h("h2", "s-title", "Thank you!"),
      h("p", "lede", CFG.endpoint ? "Your answers have been recorded. You can close this page." :
        "Test mode: no endpoint is configured, so your answers were downloaded as a file instead of being sent."),
      h("p", "hint", `Participant code: ${state.id}`));
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

  fetch("survey.json").then((r) => r.json()).then((s) => {
    survey = s;
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
  }).catch((e) => { app.innerHTML = `<p class="intro">Could not load the study (${e}).</p>`; });
})();
