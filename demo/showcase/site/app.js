// Static showcase: renders data.json (built by ../build_showcase.py).
(() => {
  const player = document.getElementById("player");
  const clips = [];          // {el, canvas, data, range, base}
  let current = null;

  const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  const noteName = (p) => NAMES[Math.round(p) % 12] + (Math.floor(Math.round(p) / 12) - 1);
  const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  const ICON_PLAY = '<svg viewBox="0 0 16 16"><path fill="currentColor" d="M4 2.5v11l9.5-5.5z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 16 16"><path fill="currentColor" d="M3.5 2h3v12h-3zM9.5 2h3v12h-3z"/></svg>';

  const h = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };

  function fmtMetric(key, v, unit) {
    if (v == null) return "–";
    if (key === "pitch") return `${noteName(v)} <span style="opacity:.6">(${v.toFixed(1)})</span>`;
    if (key === "note_beats" || key === "dissonance") return v.toFixed(2) + unit;
    return Math.round(v) + unit;
  }

  // ------------------------------------------------------------ piano roll
  function exampleRange(ex) {
    let tMax = 0, pMin = 127, pMax = 0;
    for (const c of ex.clips) {
      for (const [t, d, p, , drum] of c.notes) {
        tMax = Math.max(tMax, t + d);
        if (!drum) { pMin = Math.min(pMin, p); pMax = Math.max(pMax, p); }
      }
    }
    if (pMin > pMax) { pMin = 36; pMax = 84; }
    return { tMax: tMax + 0.25, pMin: pMin - 2, pMax: pMax + 2 };
  }

  function drawBase(clip) {
    const { canvas, data, range } = clip;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, hgt = canvas.clientHeight;
    if (!w) return;
    canvas.width = w * dpr; canvas.height = hgt * dpr;
    const off = document.createElement("canvas");
    off.width = canvas.width; off.height = canvas.height;
    const g = off.getContext("2d");
    g.scale(dpr, dpr);

    const x = (t) => (t / range.tMax) * w;
    const drumBand = data.notes.some((n) => n[4]) ? 14 : 0;
    const pitchH = hgt - drumBand - 6;
    const y = (p) => 3 + (1 - (p - range.pMin) / (range.pMax - range.pMin)) * pitchH;
    const rowH = Math.max(2, Math.min(5, pitchH / (range.pMax - range.pMin)));

    // octave guides
    g.strokeStyle = "rgba(255,255,255,0.045)"; g.lineWidth = 1;
    for (let p = Math.ceil(range.pMin / 12) * 12; p <= range.pMax; p += 12) {
      g.beginPath(); g.moveTo(0, y(p) + 0.5); g.lineTo(w, y(p) + 0.5); g.stroke();
    }
    // prompt region
    if (data.prompt_end > 0) {
      const px = x(data.prompt_end);
      g.fillStyle = "rgba(255,255,255,0.07)";
      g.fillRect(0, 0, px, hgt);
      g.setLineDash([3, 3]); g.strokeStyle = "rgba(255,255,255,0.28)";
      g.beginPath(); g.moveTo(px + 0.5, 0); g.lineTo(px + 0.5, hgt); g.stroke();
      g.setLineDash([]);
      g.fillStyle = "rgba(255,255,255,0.45)";
      g.font = "600 9px ui-monospace, Menlo, monospace";
      if (px > 46) g.fillText("PROMPT", 6, 12);
    }
    // notes
    for (const [t, d, p, vel, drum] of data.notes) {
      const a = 0.25 + 0.75 * Math.pow(vel / 127, 1.4);
      if (drum) {
        g.fillStyle = `rgba(242,193,78,${a * 0.9})`;
        g.fillRect(x(t), hgt - drumBand + 2 + (p % 3) * 3.5, Math.max(2, x(t + Math.min(d, 0.1)) - x(t)), 2.5);
      } else {
        const inPrompt = t < data.prompt_end - 1e-3;
        g.fillStyle = inPrompt ? `rgba(200,205,220,${a * 0.75})` : `rgba(150,135,255,${a})`;
        const xx = x(t), ww = Math.max(2, x(t + d) - xx - 0.5);
        g.beginPath();
        g.roundRect ? g.roundRect(xx, y(p) - rowH / 2, ww, rowH, 1.5) : g.rect(xx, y(p) - rowH / 2, ww, rowH);
        g.fill();
      }
    }
    clip.base = off;
    drawFrame(clip, clip === current ? player.currentTime : null);
  }

  function drawFrame(clip, t) {
    const { canvas, base, range } = clip;
    if (!base) return;
    const g = canvas.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.drawImage(base, 0, 0);
    if (t != null) {
      const dpr = window.devicePixelRatio || 1;
      const px = (t / range.tMax) * canvas.width;
      const grad = g.createLinearGradient(px - 40 * dpr, 0, px, 0);
      grad.addColorStop(0, "rgba(63,208,201,0)"); grad.addColorStop(1, "rgba(63,208,201,0.16)");
      g.fillStyle = grad; g.fillRect(px - 40 * dpr, 0, 40 * dpr, canvas.height);
      g.fillStyle = "#3fd0c9"; g.fillRect(px, 0, 1.5 * dpr, canvas.height);
    }
  }

  // ---------------------------------------------------------------- audio
  function setPlaying(clip, on) {
    clip.el.classList.toggle("playing", on);
    clip.btn.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
    clip.btn.setAttribute("aria-label", (on ? "Pause " : "Play ") + clip.name);
  }

  function play(clip, at) {
    if (current && current !== clip) {
      setPlaying(current, false);
      drawFrame(current, null);
      current.time.textContent = fmtTime(current.duration);
    }
    if (current !== clip) {
      current = clip;
      player.src = clip.data.audio;
    }
    if (at != null) player.currentTime = at;
    player.play();
  }

  function toggle(clip) {
    if (current === clip && !player.paused) player.pause();
    else play(clip);
  }

  player.addEventListener("play", () => current && setPlaying(current, true));
  player.addEventListener("pause", () => current && setPlaying(current, false));
  player.addEventListener("ended", () => {
    if (!current) return;
    setPlaying(current, false);
    drawFrame(current, null);
  });
  (function tick() {
    if (current && !player.paused) {
      drawFrame(current, player.currentTime);
      current.time.textContent = `${fmtTime(player.currentTime)} / ${fmtTime(current.duration)}`;
    }
    requestAnimationFrame(tick);
  })();
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && current && e.target === document.body) {
      e.preventDefault(); toggle(current);
    }
  });

  // ---------------------------------------------------------------- clips
  function makeClip(data, range, opts = {}) {
    const el = h("div", "clip");
    if (data.tag === "ours") el.classList.add("ours");
    if (opts.col) el.classList.add("col-" + opts.col);

    const top = h("div", "clip-top");
    const btn = h("button", "play", ICON_PLAY);
    top.append(btn);
    if (opts.label) top.append(h("div", "clip-label", opts.label));
    if (data.placeholder) {
      el.classList.add("placeholder");
      top.append(h("span", "tag ph", "placeholder"));
    } else if (data.tag) {
      const cls = { ours: "ours", human: "human", baseline: "base", "too strong": "warn" }[data.tag] || "";
      top.append(h("span", "tag " + cls, data.tag));
    }
    const canvas = h("canvas", "roll");
    const foot = h("div", "clip-foot");
    const time = h("span", "time");
    foot.append(time);
    if (opts.metricHTML) foot.append(h("span", "metric", opts.metricHTML));
    el.append(top, canvas, foot);

    const duration = data.notes.reduce((m, n) => Math.max(m, n[0] + n[1]), 0);
    time.textContent = fmtTime(duration);
    const clip = { el, canvas, btn, time, data, range, duration, name: opts.label || data.id };
    btn.addEventListener("click", () => toggle(clip));
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      play(clip, Math.max(0, ((e.clientX - r.left) / r.width) * range.tMax));
    });
    clips.push(clip);
    return el;
  }

  function metricCell(ex, clip, ref) {
    if (clip.placeholder) return "";
    const v = clip.metrics && clip.metrics[ex.metric];
    let html = `${ex.metric_label} <b>${fmtMetric(ex.metric, v, ex.metric_unit)}</b>`;
    if (ref != null && v != null && clip !== ref && !ref.placeholder && ref.metrics[ex.metric] !== v) {
      const r = ref.metrics[ex.metric];
      const up = v > r;
      html += ` <span class="${up ? "up" : "down"}">${up ? "▲" : "▼"}</span>`;
    }
    return html;
  }

  // --------------------------------------------------------------- blocks
  const fmtSd = (sd) => `${sd > 0 ? "+" : "−"}${Math.abs(sd)} SD`;

  function blockCard(b, sub) {
    const card = h("article", "example");
    const head = h("div", "example-head");
    const titles = h("div", "titles");
    titles.append(h("h3", null, b.title || b.prompt_name || ""));
    if (b.subtitle) titles.append(h("p", "subtitle", b.subtitle));
    head.append(titles);
    if (b.title && b.prompt_name) head.append(h("span", "pid", `prompt: ${b.prompt_name}${sub ? " · " + sub : ""}`));
    card.append(head);
    return card;
  }

  function clipsBlock(b) {
    const card = blockCard(b);
    const grid = h("div", "grid");
    const range = exampleRange(b);
    const ref = b.clips[0];
    const shown = b.metrics || (b.metric ? [b] : []);
    for (const c of b.clips) {
      grid.append(makeClip(c, range, {
        label: c.label,
        metricHTML: shown.map((m) => metricCell(m, c, ref)).join("<br>") || null,
      }));
    }
    card.append(grid);
    return card;
  }

  function steerBlock(b) {
    const card = blockCard(b);
    const range = exampleRange(b);
    const table = h("div", "steer");
    const COLS = [["down", `▼ push down`], ["unguided", "unguided"], ["up", `▲ push up`]];
    table.append(h("div"));
    for (const [k, lab] of COLS) table.append(h("div", "colhead " + k, lab));

    const rows = [...new Set(b.clips.map((c) => c.row))];
    const values = {};
    for (const row of rows) {
      const rc = Object.fromEntries(b.clips.filter((c) => c.row === row).map((c) => [c.col, c]));
      table.append(h("div", "rowhead", `${row}<small>${rc.unguided.row_sub || ""}</small>`));
      for (const [k] of COLS) {
        table.append(makeClip(rc[k], range, {
          label: { down: "Down", unguided: "Unguided", up: "Up" }[k],
          col: k,
          metricHTML: b.metric ? metricCell(b, rc[k], rc.unguided) : null,
        }));
      }
      if (!rc.unguided.placeholder) values[row] = Object.fromEntries(COLS.map(([k]) => [k, rc[k].metrics[b.metric]]));
    }
    card.append(table);
    if (b.metric && Object.keys(values).length) card.append(scaleBlock(b, values));
    return card;
  }

  function intensityBlock(b) {
    const card = blockCard(b);
    const range = exampleRange(b);
    const ref = b.clips.find((c) => c.col === "unguided");
    const sds = [...new Set(b.clips.filter((c) => c !== ref).map((c) => c.sd))];
    const lambdas = [...new Set(b.clips.filter((c) => c !== ref).map((c) => c.strength))];
    const table = h("div", "intensity");
    table.style.setProperty("--cols", sds.length);

    table.append(h("div", "colhead", "λ"));
    for (const sd of sds) table.append(h("div", "colhead " + (sd > 0 ? "up" : "down"), (sd > 0 ? "▲ " : "▼ ") + fmtSd(sd)));

    // Reference row: the unguided continuation every cell is compared to.
    table.append(h("div", "rowhead", `unguided<small>reference</small>`));
    table.append(makeClip(ref, range, {
      label: "Unguided",
      metricHTML: b.metric ? metricCell(b, ref, null) : null,
    }));
    const hint = h("div", "ref-hint", "Every cell below continues the same prompt with the EBT. Moving right pushes harder toward a bigger target; moving down raises the guidance strength.");
    hint.style.gridColumn = `span ${sds.length - 1}`;
    table.append(hint);

    const r0 = ref.metrics[b.metric];
    const devs = b.clips.filter((c) => c !== ref && c.metrics[b.metric] != null).map((c) => Math.abs(c.metrics[b.metric] - r0));
    const maxDev = Math.max(...devs, 1e-9);
    for (const lam of lambdas) {
      const cells = b.clips.filter((c) => c.strength === lam);
      const op = cells.some((c) => c.operating);
      table.append(h("div", "rowhead lam", `λ = ${lam}${op ? '<small class="op">operating point</small>' : ""}`));
      for (const sd of sds) {
        const c = cells.find((x) => x.sd === sd);
        const el = makeClip(c, range, {
          label: fmtSd(sd),
          col: sd > 0 ? "up" : "down",
          metricHTML: b.metric ? metricCell(b, c, ref) : null,
        });
        const v = c.metrics[b.metric];
        if (v != null && !c.placeholder) {
          const k = Math.abs(v - r0) / maxDev;
          el.style.setProperty("--int", k.toFixed(3));
          el.classList.add(v > r0 ? "tint-up" : "tint-down");
        }
        table.append(el);
      }
    }
    const scroll = h("div", "intensity-scroll");
    scroll.append(table);
    card.append(scroll);
    return card;
  }

  function scaleBlock(ex, values) {
    const all = Object.values(values).flatMap((o) => Object.values(o)).filter((v) => v != null);
    const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 1;
    const pos = (v) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100;
    const wrap = h("div", "scale");
    for (const [row, v] of Object.entries(values)) {
      const r = h("div", "scale-row");
      r.append(h("div", "lab", row));
      const track = h("div", "track");
      for (const [k, color] of [["down", "rgba(90,169,255,.45)"], ["up", "rgba(255,138,92,.5)"]]) {
        if (v[k] == null || v.unguided == null) continue;
        const s = h("div", "span");
        s.style.left = pos(Math.min(v[k], v.unguided)) + "%";
        s.style.width = Math.abs(pos(v.unguided) - pos(v[k])) + "%";
        s.style.background = color;
        track.append(s);
      }
      for (const k of ["down", "unguided", "up"]) {
        if (v[k] == null) continue;
        const d = h("div", "dot " + k); d.style.left = pos(v[k]) + "%";
        d.title = `${k}: ${v[k]}`;
        track.append(d);
      }
      r.append(track);
      wrap.append(r);
    }
    wrap.append(h("p", "scale-caption", `measured ${ex.metric_label} of the continuation: ` +
      `<span style="color:var(--down)">●</span> pushed down · <span style="color:var(--text)">●</span> unguided · ` +
      `<span style="color:var(--up)">●</span> pushed up`));
    return wrap;
  }

  function noteBlock(b) {
    const card = h("aside", "note");
    card.append(h("h4", null, b.title || ""), h("p", null, b.text || ""));
    return card;
  }

  const BLOCKS = { clips: clipsBlock, steer: steerBlock, intensity: intensityBlock, note: noteBlock };

  // ----------------------------------------------------------------- tabs
  function renderNode(node, depth) {
    const frag = h("div", "node depth-" + depth);
    if (node.intro && depth > 0) frag.append(h("p", "intro", node.intro));
    for (const b of node.blocks || []) frag.append(BLOCKS[b.type](b));
    if (node.tabs && node.tabs.length) {
      const bar = h("div", "tabs " + (depth === 0 ? "tabs-main" : "tabs-sub"));
      bar.setAttribute("role", "tablist");
      const panels = node.tabs.map((t) => {
        const p = renderNode(t, depth + 1);
        p.setAttribute("role", "tabpanel");
        return p;
      });
      const buttons = node.tabs.map((t, i) => {
        const btn = h("button", "tab", `<span>${t.label}</span>${t.sublabel ? `<small>${t.sublabel}</small>` : ""}`);
        btn.setAttribute("role", "tab");
        btn.addEventListener("click", () => select(i));
        bar.append(btn);
        return btn;
      });
      function select(i) {
        buttons.forEach((b, j) => { b.classList.toggle("active", j === i); b.setAttribute("aria-selected", j === i); });
        panels.forEach((p, j) => { p.hidden = j !== i; });
        requestAnimationFrame(() => clips.forEach((c) => { if (panels[i].contains(c.el)) drawBase(c); }));
      }
      frag.append(bar, ...panels);
      select(0);
    }
    return frag;
  }

  function render(data) {
    const m = data.meta;
    document.getElementById("title").textContent = m.title;
    document.getElementById("subtitle").textContent = m.subtitle;
    document.getElementById("disclaimer").textContent = m.disclaimer;
    document.getElementById("author").textContent = m.author;
    document.getElementById("repo").href = m.repo;

    const toc = document.getElementById("toc");
    const main = document.getElementById("sections");
    for (const sec of data.sections) {
      const a = h("a", null, `<span>${sec.kicker}</span>${sec.title}`);
      a.href = "#" + sec.id;
      toc.append(a);

      const s = h("section", "section");
      s.id = sec.id;
      const head = h("div", "section-head");
      head.append(h("div", "kicker", sec.kicker), h("h2", null, sec.title), h("p", "intro", sec.intro));
      s.append(head, renderNode(sec, 0));
      main.append(s);
    }
    requestAnimationFrame(() => clips.forEach(drawBase));
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => clips.forEach(drawBase), 120);
  });

  fetch("data.json").then((r) => r.json()).then(render).catch((err) => {
    console.error(err);
    document.getElementById("sections").innerHTML =
      `<p class="intro">Could not load data.json (${err}). If you opened this file directly, ` +
      `serve the folder instead: <code>python -m http.server</code>.</p>`;
  });
})();
