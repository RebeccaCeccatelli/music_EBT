"use strict";
(function () {
  const C = window.CATALOG;
  const TOK = { remi: "REMI", ant: "Anticipation" };
  const SET = { test: "Test prompts", tune: "Tune prompts", song: "100 songs" };
  const SET1 = { test: "Test prompt", tune: "Tune prompt", song: "Song" };
  const ATTR = {
    velocity: ["Velocity", "piano ↔ forte", "softer", "louder"],
    duration: ["Duration", "staccato ↔ legato", "shorter", "longer"],
    pitch_register: ["Pitch register", "low ↔ high", "lower", "higher"],
  };
  const UNG = { original: "Original", ebt: "EBT", llama: "Llama", gpt2: "GPT-2" };
  const SDS = ["-2", "-1", "-0.5", "+0.5", "+1", "+2"];

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtT = (s) => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
  };

  // ---------- index ----------
  const info = {};       // id -> description
  const num = {};        // prompt key -> number within its tok/set
  const bySet = {};
  Object.keys(C.prompts).forEach((k) => {
    const p = C.prompts[k];
    (bySet[p.tok + "|" + p.set] = bySet[p.tok + "|" + p.set] || []).push(k);
  });
  Object.values(bySet).forEach((ks) => {
    ks.sort((a, b) => C.prompts[a].pid - C.prompts[b].pid);
    ks.forEach((k, i) => { num[k] = i + 1; });
  });
  Object.keys(C.unguided).forEach((key) => {
    Object.keys(C.unguided[key]).forEach((sys) => {
      C.unguided[key][sys].forEach(([id, label]) => { info[id] = { kind: "u", key, sys, label }; });
    });
  });
  Object.keys(C.guided).forEach((key) => {
    Object.keys(C.guided[key]).forEach((attr) => {
      Object.keys(C.guided[key][attr]).forEach((sys) => {
        const cell = C.guided[key][attr][sys];
        if (cell.ref) info[cell.ref] = { kind: "g", key, attr, sys, ref: true };
        Object.keys(cell.grid).forEach((st) => {
          Object.keys(cell.grid[st]).forEach((sd) => { info[cell.grid[st][sd]] = { kind: "g", key, attr, sys, st, sd }; });
        });
      });
    });
  });

  function promptLabel(key) {
    const p = C.prompts[key];
    return SET1[p.set] + " " + num[key] + " · " + p.name;
  }
  function strengthText(sys, st) {
    const sc = C.scale[sys] || "strength";
    return sc === "λ" ? "λ " + st : sc === "N" ? "N = " + st : "strength " + st;
  }
  function describe(id) {
    const i = info[id];
    if (!i) return id;
    const p = C.prompts[i.key];
    const head = TOK[p.tok] + " · " + promptLabel(i.key);
    if (i.kind === "u") return head + " · " + (UNG[i.sys] || i.sys) + (i.sys === "original" ? "" : " · " + i.label);
    const sys = C.sys_label[i.sys] || i.sys;
    const a = ATTR[i.attr][0];
    if (i.ref) return head + " · " + sys + " · " + a + " · unguided reference";
    return head + " · " + sys + " · " + a + " · " + strengthText(i.sys, i.st) + " · target " + i.sd + " SD";
  }

  // ---------- state ----------
  const saved = store.get("cliplib.view", {});
  const S = Object.assign({ tab: "unguided", tok: "remi", uset: "test", gset: "test", attr: "velocity",
    off: [], q: "", sel: {} }, saved);
  let stars = store.get("cliplib.stars", null);
  if (!stars) {
    stars = {};
    C.picks.forEach((id) => { stars[id] = { note: "", from: "earlier picks" }; });
  }
  const saveView = () => store.set("cliplib.view", S);
  const saveStars = () => { store.set("cliplib.stars", stars); $("starCount").textContent = Object.keys(stars).length; };

  // ---------- audio ----------
  const audio = new Audio();
  audio.preload = "none";
  let cur = null;
  function play(id) {
    if (cur === id) { audio.paused ? audio.play() : audio.pause(); return; }
    const c = C.clips[id];
    if (!c) return;
    cur = id;
    audio.src = "audio/" + c[0];
    audio.play().catch(() => {});
    $("player").hidden = false;
    $("pTitle").textContent = describe(id);
    $("pPrompt").style.width = Math.min(100, (100 * c[2]) / Math.max(c[1], 0.01)) + "%";
    $("pPrompt").title = "prompt (given to the model) — the generated continuation follows";
    syncPlay();
  }
  function syncPlay() {
    const on = !audio.paused;
    document.querySelectorAll(".play[data-id]").forEach((b) => {
      const me = b.dataset.id === cur && on;
      b.classList.toggle("on", me);
      b.textContent = me ? "❚❚" : "▶";
    });
    $("pPlay").textContent = on ? "❚❚" : "▶";
    syncStars();
  }
  function syncStars() {
    document.querySelectorAll(".star[data-id]").forEach((b) => {
      const on = !!stars[b.dataset.id];
      b.classList.toggle("on", on);
      b.textContent = on ? "★" : "☆";
      b.title = on ? "In your collection (click to remove)" : "Add to collection";
    });
    const pon = cur && stars[cur];
    $("pStar").classList.toggle("on", !!pon);
    $("pStar").textContent = pon ? "★" : "☆";
  }
  audio.addEventListener("play", syncPlay);
  audio.addEventListener("pause", syncPlay);
  audio.addEventListener("ended", syncPlay);
  audio.addEventListener("timeupdate", () => {
    const d = audio.duration || (C.clips[cur] || [0, 1])[1];
    $("pFill").style.width = (100 * audio.currentTime) / d + "%";
    $("pTime").textContent = fmtT(audio.currentTime) + " / " + fmtT(d);
  });
  $("pPlay").onclick = () => { if (cur) play(cur); };
  $("pStar").onclick = () => { if (cur) toggleStar(cur); };
  $("pBar").onclick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (audio.duration) audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
  };
  function toggleStar(id) {
    if (stars[id]) delete stars[id]; else stars[id] = { note: "", t: Date.now() };
    saveStars();
    syncStars();
    if (S.tab === "collection") render();
  }

  const playBtn = (id) => `<button class="play" data-id="${esc(id)}" aria-label="Play">▶</button>`;
  const starBtn = (id) => `<button class="star" data-id="${esc(id)}">☆</button>`;

  // ---------- controls ----------
  function seg(name, opts, val) {
    return `<div class="seg">` + opts.map(([v, l]) =>
      `<button data-set="${name}" data-val="${esc(v)}" class="${v === val ? "on" : ""}">${esc(l)}</button>`).join("") + `</div>`;
  }
  function guidedSystems() {
    const out = new Set();
    Object.keys(C.guided).forEach((k) => {
      if (C.prompts[k].tok !== S.tok || C.prompts[k].set !== S.gset) return;
      Object.keys((C.guided[k] || {})[S.attr] || {}).forEach((s) => out.add(s));
    });
    return C.sys_order.filter((s) => out.has(s));
  }
  function renderControls() {
    const el = $("controls");
    if (S.tab === "collection") { el.innerHTML = ""; return; }
    let h = `<div class="ctl"><span>Tokenization</span>${seg("tok", [["remi", "REMI"], ["ant", "Anticipation"]], S.tok)}</div>`;
    if (S.tab === "unguided") {
      h += `<div class="ctl"><span>Prompts</span>${seg("uset", [["test", SET.test], ["tune", SET.tune], ["song", SET.song]], S.uset)}</div>`;
    } else {
      const attrs = S.tok === "ant" ? ["duration", "pitch_register"] : ["velocity", "duration", "pitch_register"];
      if (!attrs.includes(S.attr)) S.attr = attrs[0];
      h += `<div class="ctl"><span>Attribute</span>${seg("attr", attrs.map((a) => [a, ATTR[a][0] + " (" + ATTR[a][1] + ")"]), S.attr)}</div>`;
      h += `<div class="ctl"><span>Prompts</span>${seg("gset", [["test", SET.test], ["tune", SET.tune]], S.gset)}</div>`;
      h += `<div class="ctl"><span>Techniques</span>` + guidedSystems().map((s) =>
        `<label class="chk"><input type="checkbox" data-sys="${s}" ${S.off.includes(s) ? "" : "checked"}>${esc(C.sys_label[s] || s)}</label>`).join("") + `</div>`;
    }
    h += `<div class="ctl"><input type="search" id="q" placeholder="Filter prompts (name, instrument, id)" value="${esc(S.q)}"></div>`;
    el.innerHTML = h;
    $("q").oninput = (e) => { S.q = e.target.value; saveView(); renderMain(); };
  }
  $("controls").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-set]");
    if (!b) return;
    S[b.dataset.set] = b.dataset.val;
    saveView();
    render();
  });
  $("controls").addEventListener("change", (e) => {
    const s = e.target.dataset && e.target.dataset.sys;
    if (!s) return;
    S.off = e.target.checked ? S.off.filter((x) => x !== s) : S.off.concat([s]);
    saveView();
    renderMain();
  });

  // ---------- views ----------
  function promptKeys() {
    const set = S.tab === "unguided" ? S.uset : S.gset;
    const src = S.tab === "unguided" ? C.unguided : C.guided;
    const q = S.q.trim().toLowerCase();
    return (bySet[S.tok + "|" + set] || []).filter((k) => {
      if (!src[k]) return false;
      if (S.tab === "guided" && !src[k][S.attr]) return false;
      if (!q) return true;
      const p = C.prompts[k];
      return (p.name + " " + p.inst + " " + p.pid + " " + num[k]).toLowerCase().includes(q);
    });
  }
  function idsOf(key) {
    const out = [];
    if (S.tab === "unguided") Object.values(C.unguided[key] || {}).forEach((l) => l.forEach(([id]) => out.push(id)));
    else Object.values((C.guided[key] || {})[S.attr] || {}).forEach((c) => {
      if (c.ref) out.push(c.ref);
      Object.values(c.grid).forEach((r) => Object.values(r).forEach((id) => out.push(id)));
    });
    return out;
  }
  function renderMain() {
    const m = $("main");
    if (S.tab === "collection") { m.innerHTML = collectionView(); syncPlay(); return; }
    const keys = promptKeys();
    const selKey = S.tab + "|" + S.tok + "|" + (S.tab === "unguided" ? S.uset : S.gset);
    let sel = S.sel[selKey];
    if (!keys.includes(sel)) sel = keys[0];
    S.sel[selKey] = sel;
    const list = keys.map((k) => {
      const p = C.prompts[k];
      const n = idsOf(k).filter((id) => stars[id]).length;
      return `<button data-key="${esc(k)}" class="${k === sel ? "on" : ""}"><span class="n">${num[k]}</span>` +
        `<span class="nm">${esc(p.name)}<span class="in">${esc(p.inst || "prompt " + p.pid)}</span></span>` +
        `<span class="st">${n ? "★" + n : ""}</span></button>`;
    }).join("");
    m.innerHTML = `<div class="split"><nav class="plist" id="plist">${list || '<div class="empty">No prompts match.</div>'}</nav>` +
      `<div class="detail">${sel ? (S.tab === "unguided" ? unguidedView(sel) : guidedView(sel)) : ""}</div></div>`;
    syncPlay();
  }
  function head(key, extra) {
    const p = C.prompts[key];
    return `<div class="phead"><h2>${esc(SET1[p.set])} ${num[key]} · ${esc(p.name)}</h2>` +
      `<p class="muted">${esc(TOK[p.tok])} · prompt id ${p.pid}${p.inst ? " · " + esc(p.inst) : ""}${extra || ""}</p></div>`;
  }
  function row(id, label) {
    const c = C.clips[id];
    return `<div class="row">${playBtn(id)}<span class="lab" title="${esc(label)}">${esc(label)}</span>` +
      `<span class="dur">${c ? fmtT(c[1]) : ""}</span>${starBtn(id)}</div>`;
  }
  function unguidedView(key) {
    const u = C.unguided[key];
    const cols = C.ung_sys.filter((s) => u[s] && u[s].length).map((s) =>
      `<div class="ucol"><h4>${esc(UNG[s] || s)} <small>(${u[s].length})</small></h4>` +
      u[s].map(([id, l]) => row(id, l)).join("") + `</div>`).join("");
    return head(key, " · the grey part of the progress bar is the prompt") + `<div class="card"><div class="ucols">${cols}</div></div>`;
  }
  function guidedView(key) {
    const g = (C.guided[key] || {})[S.attr] || {};
    const a = ATTR[S.attr];
    const cards = C.sys_order.filter((s) => g[s] && !S.off.includes(s)).map((s) => {
      const cell = g[s];
      const sts = Object.keys(cell.grid).sort((x, y) => parseFloat(x) - parseFloat(y));
      const sc = C.scale[s] || "strength";
      const th = SDS.map((sd, i) =>
        `<th class="${sd[0] === "-" ? "down" : "up"}">${sd[0] === "-" ? "▼ " : "▲ "}${sd.replace("-", "−")} SD` +
        `${i === 0 ? "<br><small>" + a[2] + "</small>" : i === 5 ? "<br><small>" + a[3] + "</small>" : ""}</th>`).join("");
      const body = sts.map((st) => `<tr><th>${esc(sc === "λ" ? "λ " + st : sc === "N" ? "N " + st : st)}</th>` +
        SDS.map((sd, i) => {
          const id = cell.grid[st][sd];
          return `<td class="${i === 3 ? "gap" : ""}">${id ? `<span class="cell">${playBtn(id)}${starBtn(id)}</span>` : '<span class="none">—</span>'}</td>`;
        }).join("") + `</tr>`).join("");
      return `<div class="card"><h3>${esc(C.sys_label[s] || s)}<small>rows: ${sc === "λ" ? "strength λ" : sc === "N" ? "N (number of samples)" : "strength"} · columns: target</small></h3>` +
        (cell.ref ? `<div class="reflink">${playBtn(cell.ref)}<span>unguided reference (same run, no steering)</span>${starBtn(cell.ref)}</div>` : "") +
        `<div class="gwrap"><table class="grid"><thead><tr><th></th>${th}</tr></thead><tbody>${body}</tbody></table></div></div>`;
    }).join("");
    return head(key, " · " + a[0] + " (" + a[1] + ")") + (cards || '<div class="empty">All techniques are hidden by the filter.</div>');
  }
  function collectionView() {
    const ids = Object.keys(stars).filter((id) => C.clips[id]);
    ids.sort((x, y) => describe(x).localeCompare(describe(y), undefined, { numeric: true }));
    const rows = ids.map((id) => `<div class="crow">${playBtn(id)}<span class="desc"><a href="#" data-goto="${esc(id)}">${esc(describe(id))}</a>` +
      `<small>${esc(id)}</small></span><input data-note="${esc(id)}" placeholder="note" value="${esc(stars[id].note || "")}">${starBtn(id)}</div>`).join("");
    return `<div class="coll"><div class="toolbar">` +
      `<button class="btn primary" id="exp">Export collection (JSON)</button>` +
      `<label class="btn">Import JSON<input type="file" id="imp" accept=".json,application/json" hidden></label>` +
      `<button class="btn" id="restore">Add back earlier picks (${C.picks.length})</button>` +
      `<span class="muted">${ids.length} clips · saved in this browser only; export to keep or send them</span></div>` +
      `<div class="card">${rows || '<div class="empty">Nothing collected yet. Use ☆ next to any clip.</div>'}</div></div>`;
  }
  $("main").addEventListener("click", (e) => {
    const p = e.target.closest(".play[data-id]");
    if (p) { play(p.dataset.id); return; }
    const s = e.target.closest(".star[data-id]");
    if (s) { toggleStar(s.dataset.id); return; }
    const k = e.target.closest(".plist button[data-key]");
    if (k) {
      S.sel[S.tab + "|" + S.tok + "|" + (S.tab === "unguided" ? S.uset : S.gset)] = k.dataset.key;
      saveView();
      renderMain();
      return;
    }
    const g = e.target.closest("[data-goto]");
    if (g) { e.preventDefault(); goTo(g.dataset.goto); return; }
    if (e.target.id === "exp") exportJSON();
    if (e.target.id === "restore") {
      C.picks.forEach((id) => { if (!stars[id]) stars[id] = { note: "", from: "earlier picks" }; });
      saveStars(); render();
    }
  });
  $("main").addEventListener("input", (e) => {
    const id = e.target.dataset && e.target.dataset.note;
    if (id && stars[id]) { stars[id].note = e.target.value; saveStars(); }
  });
  $("main").addEventListener("change", (e) => {
    if (e.target.id !== "imp" || !e.target.files[0]) return;
    e.target.files[0].text().then((t) => {
      const d = JSON.parse(t);
      (d.clips || d).forEach((c) => {
        const id = typeof c === "string" ? c : c.id;
        if (C.clips[id]) stars[id] = { note: (c && c.note) || (stars[id] && stars[id].note) || "" };
      });
      saveStars(); render();
    }).catch(() => alert("Could not read that file."));
  });
  function goTo(id) {
    const i = info[id];
    if (!i) return;
    const p = C.prompts[i.key];
    S.tok = p.tok;
    if (i.kind === "u") { S.tab = "unguided"; S.uset = p.set; }
    else { S.tab = "guided"; S.gset = p.set; S.attr = i.attr; S.off = S.off.filter((x) => x !== i.sys); }
    S.q = "";
    S.sel[S.tab + "|" + S.tok + "|" + p.set] = i.key;
    saveView();
    render();
    window.scrollTo(0, 0);
  }
  function exportJSON() {
    const out = { exported: new Date().toISOString(), clips: Object.keys(stars).filter((id) => C.clips[id]).map((id) =>
      ({ id, note: stars[id].note || "", description: describe(id), audio: "audio/" + C.clips[id][0] })) };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
    a.download = "clip_collection.json";
    a.click();
  }

  // ---------- tabs ----------
  function render() {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === S.tab));
    renderControls();
    renderMain();
  }
  $("tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (!b) return;
    S.tab = b.dataset.tab;
    saveView();
    render();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || /INPUT|TEXTAREA/.test(e.target.tagName) || !cur) return;
    e.preventDefault();
    play(cur);
  });

  $("summary").textContent = Object.keys(C.clips).length.toLocaleString() + " clips · " +
    "every unguided and guided clip rendered from the final sweeps (EBT, Llama and GPT-2, both tokenizations)";
  saveStars();
  render();
})();
