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

  // Horizontally scrolling grid with a visible hint (label + position bar + right-edge fade),
  // shown only when the grid is wider than the screen: on a phone, sideways content is not
  // otherwise obvious. Tapping the bar jumps to that position.
  function hscroll(content, cls) {
    const wrap = h("div", "hscroll");
    const hint = h("div", "hscroll-hint");
    const label = h("span", "hscroll-label", "Swipe sideways to see more &rarr;");
    const track = h("div", "hscroll-track");
    const thumb = h("div", "hscroll-thumb");
    track.append(thumb);
    hint.append(label, track);
    const scroller = h("div", cls);
    scroller.append(content);
    wrap.append(hint, scroller);
    const update = () => {
      const max = scroller.scrollWidth - scroller.clientWidth;
      const can = max > 4;
      wrap.classList.toggle("can-scroll", can);
      if (!can) return;
      const frac = scroller.clientWidth / scroller.scrollWidth;
      const pos = scroller.scrollLeft / max;
      thumb.style.width = (frac * 100).toFixed(2) + "%";
      thumb.style.left = (pos * (1 - frac) * 100).toFixed(2) + "%";
      const atEnd = pos > 0.98;
      wrap.classList.toggle("at-end", atEnd);
      wrap.classList.toggle("at-start", pos < 0.02);
      label.innerHTML = atEnd ? "&larr; Swipe back" : "Swipe sideways to see more &rarr;";
    };
    scroller.addEventListener("scroll", update, { passive: true });
    track.addEventListener("click", (e) => {
      const r = track.getBoundingClientRect();
      const pos = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      scroller.scrollTo({ left: pos * (scroller.scrollWidth - scroller.clientWidth), behavior: "smooth" });
    });
    if (window.ResizeObserver) new ResizeObserver(update).observe(scroller);
    window.addEventListener("resize", update);
    requestAnimationFrame(update);
    return wrap;
  }

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
    const clip = { el, canvas, btn, time, data, range, duration, name: (opts.label || "").replace(/<[^>]*>/g, "").trim() || data.id };
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
  // Plain-language names for an ordered set of strengths; the exact value stays in a tooltip.
  // Strength levels are ranked within one technique (λ 0.02 and PPLM 16 are not on one scale).
  const RANKS = { 1: ["normal"], 2: ["gentle", "strong"], 3: ["gentle", "medium", "strong"],
    4: ["gentle", "medium", "strong", "very strong"], 5: ["very gentle", "gentle", "medium", "strong", "very strong"] };
  const strengthNames = (n) => RANKS[n] || Array.from({ length: n }, (_, i) => `level ${i + 1}`);
  // Target in the attribute's own words, e.g. −2 SD on velocity -> "much softer".
  const targetWord = (sd, b) => `${{ 0.5: "a bit ", 1: "", 2: "much " }[Math.abs(sd)] ?? ""}${sd < 0 ? b.down : b.up}`;
  const fmtSd = (sd) => `${sd > 0 ? "+" : "−"}${Math.abs(sd)} SD`;

  function blockCard(b, sub) {
    const card = h("article", "example");
    const head = h("div", "example-head");
    const titles = h("div", "titles");
    const name = b.title || b.prompt_name || "";
    titles.append(h("h3", null, b._num ? `<span class="num">${b._num}</span>${name}` : name));
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
    const ref = b.no_ref ? null : b.clips[0];   // no_ref: show metric values without ▲/▼
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
    // Default columns: down / unguided / up; combined control passes its own (unguided, 1 SD, 2 SD).
    const COLS = b.cols || [["down", `▼ push down`], ["unguided", "unguided"], ["up", `▲ push up`]];
    const cellLabel = (k, lab) => ({ down: "Down", unguided: "Unguided", up: "Up" }[k] || lab);
    const shown = b.metrics || (b.metric ? [b] : []);
    table.append(h("div"));
    for (const [k, lab] of COLS) table.append(h("div", "colhead " + k, lab));

    const rows = [...new Set(b.clips.map((c) => c.row))];
    for (const row of rows) {
      const rowClips = b.clips.filter((c) => c.row === row);
      const ref = rowClips.find((c) => c.col === "unguided");
      // A row with several strengths (e.g. PPLM low / medium / high) gets a switch inside
      // each steered card; every version is laid out in place and only the chosen one shows.
      const strengths = [...new Set(rowClips.filter((c) => c !== ref).map((c) => c.strength))];
      const names = strengthNames(strengths.length);
      const first = (rowClips.find((c) => c.default && c !== ref) || {}).strength;
      const sel = {};   // column -> chosen strength
      table.append(h("div", "rowhead", `${row}<small>${ref.row_sub || ""}</small>`));
      for (const [k, lab] of COLS) {
        const cell = h("div", "steer-cell");
        const versions = {};   // strength -> card element
        for (const c of k === "unguided" ? [ref] : rowClips.filter((x) => x.col === k)) {
          const el = makeClip(c, range, {
            label: cellLabel(k, lab),
            col: k,
            metricHTML: shown.map((m) => metricCell(m, c, ref)).join("<br>") || null,
          });
          if (k !== "unguided") {
            versions[c.strength] = el;
            if (!c.default) el.style.display = "none";
          }
          cell.append(el);
        }
        if (k !== "unguided") sel[k] = first;
        if (k !== "unguided" && strengths.length > 1) {
          const show = (st) => {
            sel[k] = st;
            for (const [x, el] of Object.entries(versions)) {
              if (x !== st && current && current.el === el) player.pause();
              el.style.display = x === st ? "" : "none";
              el.querySelectorAll(".st-btn").forEach((btn) => btn.classList.toggle("on", btn.dataset.st === st));
            }
            requestAnimationFrame(() => clips.forEach((c) => { if (c.el === versions[st]) drawBase(c); }));
          };
          // Each version carries the same switch, so it stays put while the card swaps.
          for (const el of Object.values(versions)) {
            const sw = h("div", "strength-switch");
            sw.append(h("span", "sw-label", "push"));
            strengths.forEach((st, i) => {
              const btn = h("button", "st-btn" + (st === first ? " on" : ""), names[i]);
              btn.type = "button";
              btn.dataset.st = st;
              btn.title = `${ref.row} strength ${st}`;
              btn.addEventListener("click", () => show(st));
              sw.append(btn);
            });
            el.querySelector(".clip-top").after(sw);
          }
        }
        table.append(cell);
      }
      // Same-height invisible switch on the unguided card keeps the piano rolls aligned.
      const sample = strengths.length > 1 && table.querySelector(".steer-cell:last-child .strength-switch");
      if (sample) {
        const refEl = clips.find((c) => c.data === ref).el;
        const ghost = sample.cloneNode(true);
        ghost.classList.add("ghost");
        ghost.setAttribute("aria-hidden", "true");
        refEl.querySelector(".clip-top").after(ghost);
      }
    }
    card.append(table);
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

    const who = b.method === "pplm" ? "PPLM (Llama)" : "the EBT";

    // Reference first: the unguided continuation every cell is compared to.
    table.append(h("div", "rowhead", `unguided<small>reference</small>`));
    table.append(makeClip(ref, range, {
      label: "Unguided",
      metricHTML: b.metric ? metricCell(b, ref, null) : null,
    }));
    const hint = h("div", "ref-hint", `Every cell below continues the same prompt with ${who}. Moving away from the centre asks for a bigger change (the target, in SD); moving down pushes harder to get there (the strength).`);
    hint.style.gridColumn = `span ${sds.length - 1}`;
    table.append(hint);

    // Direction groups over the target columns, then the column heads.
    const nDown = sds.filter((sd) => sd < 0).length;
    table.append(h("div", "colgroup-spacer"));
    for (const [cls, n, lab] of [["down", nDown, "▼ push down"], ["up", sds.length - nDown, "▲ push up"]]) {
      if (!n) continue;
      const g = h("div", "colgroup " + cls, lab);
      g.style.gridColumn = `span ${n}`;
      table.append(g);
    }
    table.append(h("div", "colhead first", "strength"));
    for (const sd of sds) table.append(h("div", "colhead " + (sd > 0 ? "up" : "down"), fmtSd(sd)));

    const r0 = ref.metrics[b.metric];
    const devs = b.clips.filter((c) => c !== ref && c.metrics[b.metric] != null).map((c) => Math.abs(c.metrics[b.metric] - r0));
    const maxDev = Math.max(...devs, 1e-9);
    const lamNames = strengthNames(lambdas.length);
    lambdas.forEach((lam, li) => {
      const cells = b.clips.filter((c) => c.strength === lam);
      const op = cells.some((c) => c.operating);
      const head = h("div", "rowhead lam", `${lamNames[li]}${op ? '<small class="op">recommended</small>' : ""}`);
      head.title = b.method === "pplm" ? `PPLM strength ${lam}` : `λ = ${lam}`;
      table.append(head);
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
    });
    card.append(hscroll(table, "intensity-scroll"));
    return card;
  }

  // Curated picks for one prompt. Rows = techniques (EBT, PPLM, Tilt, Best-of-N); columns = ▼ down,
  // unguided, ▲ up. The one unguided reference (the EBT's) sits in the middle and spans all rows.
  // Each down/up cell holds all of that technique's picks in that direction, with selectors for the
  // target (SD) and the strength (λ / baseline strength) when there is more than one choice.
  function ladderBlock(b) {
    const card = blockCard(b);
    // Pitch axis shared by the whole card (so higher/lower stays visible); time axis per clip, so each
    // roll fills its box instead of ending early when another clip in the card is longer.
    const shared = exampleRange(b);
    const rangeOf = (c) => ({ ...shared, tMax: exampleRange({ clips: [c] }).tMax });
    const ref = b.clips.find((c) => c.col === "ref");
    const picks = b.clips.filter((c) => c.col !== "ref");
    const maxSd = Math.max(...picks.map((c) => Math.abs(c.sd)), 1);
    const rows = [...new Set(picks.map((c) => c.row))];   // display order (EBT first)
    const table = h("div", "ladder");

    const place = (el, row, col) => { el.style.gridRow = row; el.style.gridColumn = col; return el; };
    table.append(place(h("div", "corner"), 1, 1));
    table.append(place(h("div", "colhead down", `▼ ${b.down}`), 1, 2));
    table.append(place(h("div", "colhead ref", "unguided<small>reference</small>"), 1, 3));
    table.append(place(h("div", "colhead up", `▲ ${b.up}`), 1, 4));

    const refCell = place(h("div", "lad-ref"), `2 / span ${rows.length}`, 3);
    if (ref) refCell.append(makeClip(ref, rangeOf(ref), {
      label: "Unguided", metricHTML: b.metric ? metricCell(b, ref, null) : null,
    }));
    refCell.append(h("p", "lad-ref-note", "The EBT's unguided continuation: listen to it first, then compare each row."));
    table.append(refCell);

    const meter = (c) => c.levels > 1
      ? ` <span class="meter" title="strength ${c.level} of ${c.levels} used for this technique in this tab">` +
        Array.from({ length: c.levels }, (_, i) => `<i class="${i < c.level ? "on" : ""}"></i>`).join("") + "</span>"
      : "";
    const sdText = (sd) => `${sd > 0 ? "+" : "−"}${Math.abs(sd)} SD`;
    // Plain words on the buttons; the exact setting (λ 0.02, strength 16) stays in a tooltip.
    // Best-of-N keeps its number, which already means something: "best of 8 tries".
    const isBon = (c) => c.label.startsWith("best of");
    const strengthName = (c) => isBon(c) ? `${c.strength} tries` : (strengthNames(c.levels || 1)[(c.level || 1) - 1]);
    const strengthScale = (c) => isBon(c) ? "best of" : "push";
    // One setting only for this technique in this tab (e.g. PPLM 32): no rank to name, the exact value on hover.
    const strengthHTML = (c) => isBon(c) ? ` · <span class="exact" title="N = ${c.strength}">${c.label} tries</span>`
      : (c.levels || 1) > 1 ? ` · <span class="exact" title="${c.label}">${strengthName(c)} push</span>`
      : "";

    rows.forEach((row, ri) => {
      const rc = picks.filter((c) => c.row === row);
      const ours = rc[0].tag === "ours";
      table.append(place(h("div", "rowhead" + (ours ? " ours" : ""), `${row}<small>${rc[0].row_sub || ""}</small>`), ri + 2, 1));
      for (const [dir, col] of [[-1, 2], [1, 4]]) {
        const cell = place(h("div", "lad-cell " + (dir < 0 ? "down" : "up")), ri + 2, col);
        const here = rc.filter((c) => Math.sign(c.sd) === dir)
          .sort((x, y) => Math.abs(x.sd) - Math.abs(y.sd) || parseFloat(x.strength) - parseFloat(y.strength));
        if (!here.length) {
          cell.classList.add("none");
          cell.append(h("span", "lad-none", "nothing to show for this technique in this direction"));
          table.append(cell);
          continue;
        }
        // target buttons run outward: −2 −1 −0.5 on the softer side, +0.5 +1 +2 on the louder side
        const sds = [...new Set(here.map((c) => c.sd))].sort((x, y) => x - y);
        const els = new Map();
        for (const c of here) {
          const el = makeClip(c, rangeOf(c), {
            label: `<span title="${sdText(c.sd)} · ${c.label}">${dir < 0 ? "▼" : "▲"} ${targetWord(c.sd, b)} <span class="sdv">${sdText(c.sd)}</span>${strengthHTML(c)}</span>${meter(c)}`,
            col: dir < 0 ? "down" : "up",
            metricHTML: b.metric ? metricCell(b, c, ours && ref ? ref : null) : null,
          });
          el.style.setProperty("--int", (Math.abs(c.sd) / maxSd).toFixed(3));
          el.classList.add(dir > 0 ? "tint-up" : "tint-down");
          el.style.display = "none";
          els.set(c, el);
        }
        let selSd = dir < 0 ? sds[0] : sds[sds.length - 1];   // start at the biggest target: the clearest change
        let selSt = null;
        const sdBar = h("div", "strength-switch");
        const stBar = h("div", "strength-switch");
        const show = () => {
          const opts = here.filter((c) => c.sd === selSd);
          if (!opts.some((c) => c.strength === selSt)) selSt = opts[0].strength;
          for (const [c, el] of els) {
            const on = c.sd === selSd && c.strength === selSt;
            if (!on && current && current.el === el) player.pause();
            el.style.display = on ? "" : "none";
          }
          sdBar.querySelectorAll(".st-btn").forEach((x) => x.classList.toggle("on", Number(x.dataset.v) === selSd));
          stBar.replaceChildren(h("span", "sw-label", strengthScale(opts[0])));
          stBar.style.display = opts.length > 1 ? "" : "none";
          for (const c of opts) {
            const btn = h("button", "st-btn" + (c.strength === selSt ? " on" : ""), strengthName(c));
            btn.type = "button";
            btn.title = c.label;
            btn.addEventListener("click", () => { selSt = c.strength; show(); });
            stBar.append(btn);
          }
          requestAnimationFrame(() => clips.forEach((x) => { if (x.el.style.display !== "none" && [...els.values()].includes(x.el)) drawBase(x); }));
        };
        // controls and clips share one framed box, so the selectors clearly belong to this cell only
        const box = h("div", "lad-box " + (dir < 0 ? "down" : "up"));
        const ctrl = h("div", "lad-ctrl");
        if (sds.length > 1) {
          sdBar.append(h("span", "sw-label", "asked for"));
          for (const sd of sds) {
            const btn = h("button", "st-btn", `${targetWord(sd, b)} <small>${sdText(sd)}</small>`);
            btn.type = "button";
            btn.dataset.v = sd;
            btn.addEventListener("click", () => { selSd = sd; show(); });
            sdBar.append(btn);
          }
          ctrl.append(sdBar);
        }
        ctrl.append(stBar);
        if (here.length > 1) box.append(ctrl);   // several targets and/or several strengths to choose from
        box.append(...els.values());
        cell.append(box);
        show();
        table.append(cell);
      }
    });
    card.append(hscroll(table, "ladder-scroll"));
    return card;
  }

  function noteBlock(b) {
    if (b.collapsible) {   // a "learn more" card: one line until clicked
      const card = h("details", "note learn");
      const sum = h("summary", null,
        `<span class="learn-icon">i</span><span class="learn-text"><b>${b.title || ""}</b>` +
        `${b.teaser ? `<small>${b.teaser}</small>` : ""}</span><span class="learn-cta">Learn more</span>`);
      card.append(sum, h("p", null, b.text || ""));
      card.addEventListener("toggle", () => {
        sum.querySelector(".learn-cta").textContent = card.open ? "Close" : "Learn more";
      });
      return card;
    }
    const card = h("aside", "note");
    card.append(h("h4", null, b.title || ""), h("p", null, b.text || ""));
    return card;
  }

  // Several blocks in one shaded card, e.g. all the baselines of a tab.
  function groupBlock(b) {
    const card = blockCard(b);
    card.classList.add("group");
    for (const c of b.blocks || []) {
      const el = BLOCKS[c.type](c);
      el.classList.add("nested");
      card.append(el);
    }
    return card;
  }

  const BLOCKS = { clips: clipsBlock, steer: steerBlock, intensity: intensityBlock, ladder: ladderBlock, note: noteBlock, group: groupBlock };

  // ----------------------------------------------------------------- tabs
  // Above each block title: which tabs it belongs to, and whether it is a baseline or ours.
  function withContext(el, b, path) {
    const titles = el.querySelector(".titles");
    if (!titles || (!path.length && !b.role)) return el;
    const line = h("div", "context");
    if (b.role) line.append(h("span", "role " + b.role, b.role === "ours" ? "ours" : "baseline"));
    if (path.length) line.append(h("span", "crumb", path.join(" · ")));
    titles.prepend(line);
    return el;
  }

  function renderNode(node, depth, path = []) {
    const frag = h("div", "node depth-" + depth);
    if (node.intro && depth > 0) frag.append(h("p", "intro", node.intro));
    // collapse_after: show the first n blocks; the rest sit behind a "Show more" button.
    const extra = [];
    // numbered: examples (not notes) get 1, 2, 3 … in their tab
    if (node.numbered) {
      let k = 0;
      for (const b of node.blocks || []) if (b.type !== "note") b._num = ++k;
    }
    let nEx = 0;   // examples so far (notes don't count towards collapse_after)
    (node.blocks || []).forEach((b) => {
      const el = withContext(BLOCKS[b.type](b), b, path);
      if (b.type !== "note" && node.collapse_after != null && nEx++ >= node.collapse_after) {
        el.style.display = "none";
        extra.push(el);
      }
      frag.append(el);
    });
    if (extra.length) {
      const more = `Show ${extra.length} more example${extra.length > 1 ? "s" : ""}`;
      const btn = h("button", "show-more", `${more} <span>▾</span>`);
      btn.type = "button";
      btn.addEventListener("click", () => {
        const open = extra[0].style.display === "none";
        if (!open && current && extra.some((el) => el.contains(current.el))) player.pause();
        extra.forEach((el) => { el.style.display = open ? "" : "none"; });
        btn.innerHTML = open ? "Show fewer <span>▴</span>" : `${more} <span>▾</span>`;
        if (open) requestAnimationFrame(() => clips.forEach((c) => { if (extra.some((el) => el.contains(c.el))) drawBase(c); }));
        else btn.scrollIntoView({ block: "nearest" });
      });
      frag.append(btn);
    }
    if (node.tabs && node.tabs.length) {
      const bar = h("div", "tabs " + (depth === 0 ? "tabs-main" : "tabs-sub"));
      bar.setAttribute("role", "tablist");
      let helpBox = null;
      if (node.tabs_label) {
        const lab = h("span", "tabs-label", node.tabs_label);
        if (node.tabs_help) {
          // "?" next to the label opens a short explanation below the tab bar.
          const q = h("button", "help-q", "?");
          q.type = "button";
          q.setAttribute("aria-label", `What is ${node.tabs_label.toLowerCase()}?`);
          q.setAttribute("aria-expanded", "false");
          helpBox = h("div", "help-box", node.tabs_help);
          helpBox.style.display = "none";
          const set = (open) => {
            helpBox.style.display = open ? "" : "none";
            q.setAttribute("aria-expanded", String(open));
            q.classList.toggle("on", open);
          };
          q.addEventListener("click", (e) => { e.stopPropagation(); set(helpBox.style.display === "none"); });
          document.addEventListener("click", (e) => { if (!helpBox.contains(e.target)) set(false); });
          document.addEventListener("keydown", (e) => { if (e.key === "Escape") set(false); });
          lab.append(q);
        }
        bar.append(lab);
      }
      const panels = node.tabs.map((t) => {
        const p = renderNode(t, depth + 1, [...path, t.label]);
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
      frag.append(bar, ...(helpBox ? [helpBox] : []), ...panels);
      select(0);
    }
    return frag;
  }

  function render(data) {
    const m = data.meta;
    document.getElementById("title").textContent = m.title;
    document.getElementById("subtitle").innerHTML = m.subtitle;  // author-written markup: bold names, trait tags
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
