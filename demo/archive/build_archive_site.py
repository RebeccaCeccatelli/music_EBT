"""Listening Archive as a standalone static site (Netlify), identical in look to the claude.ai pages.

    python demo/archive/build_archive_site.py      # -> music_EBT_logs/survey/archive_site/ + ~/transfer/archive.zip

Sources (the published page code and clip indexes, kept on scratch):
  survey/picker_hub/index.html                  -> index.html                      (Listening Archive)
  survey/unguided_picker/{index.html,data/}      -> unguided/                       (Unguided Outputs)
  survey/guided_pickers_v2/{remi,ant}/           -> single-remi/, single-anticipation/
  survey/combined_pickers/remi/                  -> combined-remi/
Changes from the artifact versions:
  - wrapped in the same document skeleton claude.ai adds at publish time, so the pages render the same;
  - audio packed into ~1 MB MP3 chunks (audio/<n>.mp3; raw MP3 bytes concatenated, each clip = [chunk, offset,
    length] in the index), clips of one prompt card kept together. ~450 files instead of ~13,000, so a Netlify
    drag-and-drop deploy works; playing a card loads one or two chunks;
  - clip indexes stripped of cluster paths (midi, prompt_midi) and packing fields; "Copy saved" lists clip ids;
  - hub links are relative; ☆ Save keeps each visitor's list in their own browser (no claude.ai database).
The author's own saved lists (from the claude.ai pages) are exported in demo/archive/saved/.
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

SURVEY = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs/survey"
OUT = SURVEY / "archive_site"
ZIP = Path.home() / "transfer/archive.zip"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "icons"))
from make_icons import head_tags  # noqa: E402
ICON_TAGS = head_tags("archive")  # browser-tab icon (a record), inlined so every sub-page can use it
SKELETON_HEAD = ('<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,'
                 'initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light;box-sizing:border-box;'
                 'padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
                 'html{scroll-padding-top:env(safe-area-inset-top,0px)}body{margin:0;padding:0;font:14px '
                 '-apple-system,BlinkMacSystemFont,sans-serif;background:#fff;color:#000}img{max-width:100%}'
                 '[hidden]:not([hidden=until-found i]){display:none!important}</style>' + ICON_TAGS + '</head><body>\n')
SKELETON_TAIL = "\n</body></html>\n"

PAGES = {  # out dir -> (source dir, mp3 dir, kind)
    "unguided": (SURVEY / "unguided_picker", SURVEY / "candidates/mp3", "unguided"),
    "single-remi": (SURVEY / "guided_pickers_v2/remi", SURVEY / "guided/mp3", "guided"),
    "single-anticipation": (SURVEY / "guided_pickers_v2/ant", SURVEY / "guided/mp3", "guided"),
    "combined-remi": (SURVEY / "combined_pickers/remi", SURVEY / "combined/remi/mp3", "guided"),
    "combined-anticipation": (SURVEY / "combined_pickers/ant", SURVEY / "combined/ant/mp3", "guided"),
}
HUB_LINKS = {
    "https://claude.ai/artifact/AFp5Khpb96uB7VjECoxjVd": "unguided/",
    "https://claude.ai/artifact/6XdgVLkt6zXenTd7qg9eoM": "single-remi/",
    "https://claude.ai/artifact/6VcNq6B9Tu7uuDi7PYp11k": "single-anticipation/",
    "https://claude.ai/artifact/FPDfpVuB7cTm4NtQ9LHcb1": "combined-remi/",
    "https://claude.ai/artifact/6LGgTSvwzxg6KShVKwYzUG": "combined-anticipation/",
}
DROP = {"midi", "prompt_midi", "at", "table"}
CHUNK = 1_000_000
CHUNK_JS = ('''const audioCache = {};
function chunk(name) {
  if (!audioCache[name]) audioCache[name] = fetch(name).then(r => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .catch(e => { delete audioCache[name]; throw e; });
  return audioCache[name];
}
async function srcFor(id) {
  if (urlCache[id]) return urlCache[id];
  const [name, off, len] = index.clips[id].at;
  const buf = await chunk(name);
  return (urlCache[id] = URL.createObjectURL(new Blob([buf.slice(off, off + len)], { type: "audio/mpeg" })));
}
''')


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:70], s.count(old))
    return s.replace(old, new)


def page(src, kind):
    s = src.read_text()
    if s.lstrip().lower().startswith("<!doctype"):   # some copies already carry the publish skeleton
        s = s[s.index("<body>") + len("<body>"):]
        s = s.rstrip()
        if s.endswith("</body></html>"):
            s = s[:-len("</body></html>")]
    if kind == "guided":
        a = s.index("function chunk(name) {")
        b = s.index("// Picks made before the merge")
        s = s[:a] + CHUNK_JS + s[b:]
        s = rep(s, ", c.strength ?? \"\", c.midi].join", ", c.strength ?? \"\", id].join")
        s = rep(s, "strength\\tmidi", "strength\\tclip id")
    else:
        a = s.index("async function chunkFor(id) {")
        b = s.index("function stopCurrent() {")
        s = s[:a] + CHUNK_JS + "\n" + s[b:]
        s = rep(s, ", c.strength ?? \"\", c.midi].join", ", c.strength ?? \"\", id].join")
        s = rep(s, "strength\\tmidi", "strength\\tclip id")
    s = rep(s, "// Picks are also kept in the artifact's database (doc survey/picks), so Claude can read them back.",
            "// Saved clips are kept in this browser (localStorage).")
    assert "claude.ai" not in s and "/home/" not in s
    return SKELETON_HEAD + s.strip() + SKELETON_TAIL


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    hub = (SURVEY / "picker_hub/index.html").read_text()
    for a, b in HUB_LINKS.items():
        hub = rep(hub, f'href="{a}"', f'href="{b}"')
    assert "claude.ai" not in hub
    (OUT / "index.html").write_text(SKELETON_HEAD + hub.strip() + SKELETON_TAIL)

    total = 0
    for name, (src, mp3, kind) in PAGES.items():
        if not (src / "data/index.json").exists():
            print(f"{name}: not built yet, skipped")
            continue
        d = OUT / name
        (d / "data").mkdir(parents=True)
        (d / "audio").mkdir()
        (d / "index.html").write_text(page(src / "index.html", kind))
        idx = json.loads((src / "data/index.json").read_text())
        clips = {k: {f: v for f, v in c.items() if f not in DROP} for k, c in idx["clips"].items()}
        # pack: clips of the same prompt card next to each other
        order = sorted(clips, key=lambda k: (clips[k].get("section", ""), clips[k].get("split") or "",
                                              clips[k].get("prompt", 0), clips[k].get("system", ""), k))
        n, buf = 0, bytearray()

        def flush():
            nonlocal n, buf
            if buf:
                (d / "audio" / f"{n}.mp3").write_bytes(bytes(buf))
                n, buf = n + 1, bytearray()
        for cid in order:
            fid = cid.split("~", 1)[1] if kind == "guided" else cid
            b = (mp3 / f"{fid}.mp3").read_bytes()
            if buf and len(buf) + len(b) > CHUNK:
                flush()
            clips[cid]["at"] = [f"audio/{n}.mp3", len(buf), len(b)]
            buf += b
        flush()
        idx["clips"] = {k: clips[k] for k in idx["clips"]}
        idx.pop("chunks", None)
        idx["format"] = "mp3-concat"
        (d / "data/index.json").write_text(json.dumps(idx, separators=(",", ":"), ensure_ascii=False))
        total += n
        assert "/home/" not in (d / "data/index.json").read_text()
        print(f"{name}: {len(clips)} clips in {n} audio chunks")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"{total} audio chunks, {files} files in all, {size / 1e6:.0f} MB")

    ZIP.unlink(missing_ok=True)
    subprocess.run(["nice", "ionice", "-c3", "zip", "-q", "-r", "-0", str(ZIP), "archive_site"],
                   cwd=OUT.parent, check=True)
    print("wrote", ZIP, f"{ZIP.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
